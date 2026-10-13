#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-Grade Real-Time Cross-Day Evaluation Pipeline
Model: Logistic Regression (LR)
Training Domain: Friday DDoS (CICIDS2017)
Testing Domain : Wednesday Cross-Day Scenario

Protocol:
1. Classification Metrics (Accuracy, Precision, Recall, F1, F2, MCC, ROC-AUC) calculated on 100% of Wednesday test data.
2. Real-Time Single-Sample Latency profiled strictly with Batch Size = 1 on 50,000 samples.
3. Accurate Tail Latency Percentiles (Mean, P50, P90, P95, P99).
4. System Resource Profiles (RAM RSS Footprint & CPU Utilization over Time).
5. Thesis Visual Standard: Times New Roman, Size 10, 300 DPI, Units, Legends, and Captions.
"""

import os
import gc
import json
import time
import warnings
import psutil
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. Thesis Plotting Configuration (Strict Guidelines)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_FILENAME = "lr_model.joblib"
SCALER_FILENAME = "scaler_lr.joblib"
METADATA_FILENAME = "lr_metadata.json"

MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(RESULTS_DIR, SCALER_FILENAME)
METADATA_PATH = os.path.join(RESULTS_DIR, METADATA_FILENAME)

CHUNK_SIZE = 25000
BENCHMARK_LATENCY_SAMPLES = 50000

FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

CANONICAL_RAW_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]


# ============================================================
# 2. Helpers & Data Streaming
# ============================================================
def out(filename):
    return os.path.join(RESULTS_DIR, filename)


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def find_file(filename):
    candidate_paths = [
        os.path.join(RESULTS_DIR, filename),
        os.path.join("/mnt/data", filename),
        os.path.join("/app", filename),
        filename,
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] File not found: {filename}. Checked paths: {candidate_paths}")


def resolve_csv_columns(file_path):
    header_df = pd.read_csv(file_path, nrows=0)
    actual_columns = list(header_df.columns)
    normalized_to_actual = {str(col).strip(): col for col in actual_columns}

    missing_cols = [col for col in CANONICAL_RAW_COLUMNS if col not in normalized_to_actual]
    if missing_cols:
        raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")

    return [normalized_to_actual[col] for col in CANONICAL_RAW_COLUMNS]


def iter_prepared_chunks(file_path, chunk_size=CHUNK_SIZE):
    usecols = resolve_csv_columns(file_path)

    csv_iterator = pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunk_size,
        engine="c",
        low_memory=True,
    )

    for raw_chunk in csv_iterator:
        raw_chunk.columns = raw_chunk.columns.astype(str).str.strip()

        fwd_packets = pd.to_numeric(raw_chunk["Total Fwd Packets"], errors="coerce")
        bwd_packets = pd.to_numeric(raw_chunk["Total Backward Packets"], errors="coerce")
        avg_packet_size = pd.to_numeric(raw_chunk["Average Packet Size"], errors="coerce")
        flow_rate = pd.to_numeric(raw_chunk["Flow Packets/s"], errors="coerce")
        flow_duration = pd.to_numeric(raw_chunk["Flow Duration"], errors="coerce")
        labels = raw_chunk["Label"].astype(str).str.strip().str.upper()

        prepared = pd.DataFrame({
            "Packet_Count": (fwd_packets + bwd_packets).astype(np.float32),
            "Packet_Size": avg_packet_size.astype(np.float32),
            "Flow_Rate": flow_rate.astype(np.float32),
            "Flow_Duration": flow_duration.astype(np.float32),
            "Label": labels,
        })

        prepared.replace([np.inf, -np.inf], np.nan, inplace=True)
        prepared.dropna(subset=FEATURE_COLUMNS + ["Label"], inplace=True)

        if len(prepared) == 0:
            del raw_chunk, prepared
            gc.collect()
            continue

        X_chunk = prepared[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        y_chunk = (prepared["Label"].to_numpy() != "BENIGN").astype(np.uint8)

        del raw_chunk, prepared
        gc.collect()

        yield X_chunk, y_chunk


def get_dataset_statistics(file_path):
    n_clean = 0
    n_benign = 0
    n_attack = 0

    print(f">> [INFO] Inspecting class distribution in: {os.path.basename(file_path)}")
    for _, y_chunk in iter_prepared_chunks(file_path):
        n_clean += len(y_chunk)
        n_benign += int((y_chunk == 0).sum())
        n_attack += int((y_chunk == 1).sum())
        del y_chunk
        gc.collect()

    return {
        "n_rows_clean": int(n_clean),
        "n_benign": int(n_benign),
        "n_attack": int(n_attack),
    }


def get_memory_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


# ============================================================
# 3. Evaluation & Threshold Calibration
# ============================================================
def evaluate_predictions(y_true, y_prob, threshold, roc_auc_val=None):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_value = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_val) if roc_auc_val is not None else float(roc_auc_score(y_true, y_prob)),
        "false_positive_rate": float(fpr_value),
        "false_negative_rate": float(fnr_value),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


def get_calibrated_threshold(model, scaler, friday_file):
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if "best_threshold" in meta:
                    val_thresh = meta["best_threshold"]
                    val_f2 = meta.get("calibration", {}).get("f2_score", 0.0)
                    print(f">> [METADATA] Loaded calibrated threshold: {val_thresh:.4f} (Friday validation F2: {val_f2:.4f})")
                    return val_thresh, val_f2, None
        except Exception:
            pass

    print(">> [CALIBRATION] Calibrating decision threshold on Friday validation set (last 20%)...")
    friday_stats = get_dataset_statistics(friday_file)
    total_samples = friday_stats["n_rows_clean"]
    validation_start_index = int(total_samples * 0.80)

    val_y_list, val_prob_list = [], []
    processed = 0

    for X_chunk, y_chunk in iter_prepared_chunks(friday_file):
        chunk_start = processed
        chunk_end = processed + len(y_chunk)
        processed = chunk_end

        if chunk_end <= validation_start_index:
            continue

        local_start = max(0, validation_start_index - chunk_start)
        X_val_scaled = scaler.transform(X_chunk[local_start:])
        probs = model.predict_proba(X_val_scaled)[:, 1].astype(np.float32)

        val_y_list.append(y_chunk[local_start:].copy())
        val_prob_list.append(probs)

    y_val = np.concatenate(val_y_list).astype(np.uint8)
    prob_val = np.concatenate(val_prob_list).astype(np.float32)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_th = 0.50
    best_f2 = -1.0

    for th in thresholds:
        preds = (prob_val >= th).astype(np.uint8)
        score = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_th = float(th)

    print(f">> [CALIBRATION] Optimal Threshold (F2 max): {best_th:.4f} (Val F2: {best_f2:.4f})")
    return best_th, float(best_f2), friday_stats


# ============================================================
# 4. Pure Real-Time Latency Benchmark (Batch Size = 1)
# ============================================================
def benchmark_single_sample_latency(model, scaler, wednesday_file, n_benchmark_samples=BENCHMARK_LATENCY_SAMPLES):
    print(f">> [BENCHMARK] Profiling pure single-sample inference latency (Batch Size = 1, N = {n_benchmark_samples:,})...")

    collected_samples = []
    for X_chunk, _ in iter_prepared_chunks(wednesday_file):
        collected_samples.append(X_chunk)
        if sum(len(c) for c in collected_samples) >= n_benchmark_samples:
            break

    X_benchmark = np.vstack(collected_samples)[:n_benchmark_samples]
    total_eval = len(X_benchmark)

    # Pre-scale features to isolate pure model inference invocation
    X_benchmark_scaled = scaler.transform(X_benchmark)

    # Warm-up (200 invocations)
    for i in range(min(200, total_eval)):
        _ = model.predict_proba(X_benchmark_scaled[i : i + 1])

    latencies_ms = np.empty(total_eval, dtype=np.float64)

    # Pure Model Inference Timer Window
    for i in range(total_eval):
        x_single = X_benchmark_scaled[i : i + 1]

        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()

        latencies_ms[i] = (t1 - t0) * 1000.0

    latency_stats = {
        "n_samples_evaluated": int(total_eval),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0,
    }

    return latency_stats


# ============================================================
# 5. Full Streaming Evaluation on Wednesday (100% of Data)
# ============================================================
def evaluate_wednesday_streaming(model, scaler, wednesday_file):
    print(">> [EVALUATION] Executing streaming evaluation across 100% of Wednesday dataset...")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    prob_parts = []
    memory_trace_mb = []
    cpu_trace_pct = []

    total_samples = 0
    total_time_sec = 0.0

    for X_chunk, y_chunk in iter_prepared_chunks(wednesday_file):
        mem_before = get_memory_usage_mb()
        t0 = time.perf_counter()

        X_chunk_scaled = scaler.transform(X_chunk)
        probs = model.predict_proba(X_chunk_scaled)[:, 1].astype(np.float32)

        t1 = time.perf_counter()
        mem_after = get_memory_usage_mb()

        y_parts.append(y_chunk)
        prob_parts.append(probs)

        memory_trace_mb.append(max(mem_before, mem_after))
        cpu_trace_pct.append(process.cpu_percent(interval=None))

        total_samples += len(y_chunk)
        total_time_sec += (t1 - t0)

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities_test = np.concatenate(prob_parts).astype(np.float32)

    del y_parts, prob_parts
    gc.collect()

    stream_stats = {
        "total_inference_time_sec": float(total_time_sec),
        "n_inference_samples": int(total_samples),
        "peak_memory_mb": float(np.max(memory_trace_mb)) if memory_trace_mb else 0.0,
        "avg_memory_mb": float(np.mean(memory_trace_mb)) if memory_trace_mb else 0.0,
        "peak_cpu_percent": float(np.max(cpu_trace_pct)) if cpu_trace_pct else 0.0,
        "avg_cpu_percent": float(np.mean(cpu_trace_pct)) if cpu_trace_pct else 0.0,
    }

    profile_data = {
        "memory_trace_mb": np.array(memory_trace_mb, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace_pct, dtype=np.float32),
    }

    return y_test, probabilities_test, stream_stats, profile_data


# ============================================================
# 6. Standardized Thesis Visualizations
# ============================================================
def save_comprehensive_metrics_bar(metrics, threshold_label, output_path, is_calibrated=False):
    """
    Standard Thesis Bar Chart:
    Metrics: Accuracy, Precision, Recall, F1-Score, F2-Score, MCC, ROC-AUC
    Font: Times New Roman 10pt, 300 DPI, Units: Score [0.0 - 1.0], Legend & Caption.
    """
    ensure_parent_dir(output_path)

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    vals = [metrics.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    bar_color = "#59A14F" if is_calibrated else "#2b5c8f"
    legend_label = f"Calibrated Threshold ({threshold_label})" if is_calibrated else f"Standard Threshold ({threshold_label})"

    bars = ax.bar(metric_names, vals, color=bar_color, edgecolor="black", linewidth=0.8, width=0.52, label=legend_label)

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=10)
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + 0.02, 1.08),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    caption_text = f"Figure: Comprehensive classification performance metrics on Wednesday cross-day dataset ({threshold_label})."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_realtime_latency_percentiles_plot(latency_stats, output_path):
    percentiles_keys = ["p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    percentile_labels = ["P50 (Median)", "P90", "P95", "P99 (Tail)"]
    percentile_vals = [latency_stats[k] for k in percentiles_keys]

    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    bars = ax.bar(percentile_labels, percentile_vals, color="#7A68A6", edgecolor="black", linewidth=0.8, width=0.5, label="Pure Model Latency")

    offset = max(percentile_vals) * 0.03 if max(percentile_vals) > 0 else 0.001
    for b, v in zip(bars, percentile_vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + offset, f"{v:.4f} ms", ha="center", fontsize=10)

    ax.set_ylabel("Pure Inference Latency (ms/sample)", fontsize=10)
    ax.set_xlabel("Tail Latency Percentiles", fontsize=10)
    ax.set_ylim(0, max(percentile_vals) * 1.25)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption_text = "Figure: Single-sample pure inference latency percentiles for real-time edge evaluation (Batch Size = 1)."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_single_confusion_matrix(y_true, y_pred, title, caption_label, output_path, cmap="Blues"):
    ensure_parent_dir(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax,
        annot_kws={"size": 10}
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label (Class)", fontsize=10)
    ax.set_ylabel("True Label (Class)", fontsize=10)

    caption_text = f"Figure: Confusion matrix of cross-day evaluation on Wednesday dataset under {caption_label}."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_resource_usage_plots(profile_data, memory_output_path, cpu_output_path):
    """
    Visualizes Memory (RSS) Footprint and CPU Utilization over evaluation chunks.
    Standardized for edge-deployment analysis (e.g. Raspberry Pi RAM constraints).
    """
    ensure_parent_dir(memory_output_path)
    ensure_parent_dir(cpu_output_path)

    mem_trace = profile_data["memory_trace_mb"]
    cpu_trace = profile_data["cpu_trace_pct"]
    chunks = np.arange(1, len(mem_trace) + 1)

    # 1. RAM / Memory Usage Plot
    fig_mem, ax_mem = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax_mem.plot(chunks, mem_trace, color="#1f77b4", marker="o", markersize=3, linewidth=1.5, label="Measured RSS Memory")
    
    # Reference line for standard embedded device (1 GB RAM budget)
    ax_mem.axhline(1024, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7, label="1 GB RAM Threshold")
    
    ax_mem.set_ylabel("Resident Set Size (MB)", fontsize=10)
    ax_mem.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=10)
    ax_mem.set_ylim(0, max(np.max(mem_trace) * 1.3, 1100))
    ax_mem.legend(loc="upper right", frameon=True, fontsize=10)
    ax_mem.grid(True, linestyle="--", alpha=0.5)

    caption_mem = "Figure: Memory consumption profile (Resident Set Size in MB) during continuous streaming inference."
    fig_mem.text(0.5, 0.01, caption_mem, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(memory_output_path, dpi=300)
    plt.close(fig_mem)

    # 2. CPU Utilization Plot
    fig_cpu, ax_cpu = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax_cpu.plot(chunks, cpu_trace, color="#ff7f0e", marker="s", markersize=3, linewidth=1.5, label="Process CPU Load (%)")
    
    avg_cpu = float(np.mean(cpu_trace))
    ax_cpu.axhline(avg_cpu, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"Mean CPU Load ({avg_cpu:.1f}%)")

    ax_cpu.set_ylabel("Processor Utilization (%)", fontsize=10)
    ax_cpu.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=10)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.legend(loc="upper right", frameon=True, fontsize=10)
    ax_cpu.grid(True, linestyle="--", alpha=0.5)

    caption_cpu = "Figure: Processor load profile over streaming inference chunks across the Wednesday test dataset."
    fig_cpu.text(0.5, 0.01, caption_cpu, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(cpu_output_path, dpi=300)
    plt.close(fig_cpu)


# ============================================================
# 7. Main Execution Pipeline
# ============================================================
def main():
    print("=" * 78)
    print(" LOGISTIC REGRESSION CROSS-DAY EVALUATION (REAL-TIME STREAMING)")
    print("=" * 78)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    friday_file = find_file(FRIDAY_FILENAME)
    wednesday_file = find_file(WEDNESDAY_FILENAME)
    model_path = find_file(MODEL_FILENAME)
    scaler_path = find_file(SCALER_FILENAME)

    print(f">> Model file : {model_path}")
    print(f">> Scaler file: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 1. Friday Validation Split Calibration
    optimal_threshold, val_f2, friday_stats = get_calibrated_threshold(model, scaler, friday_file)

    # 2. Pure Real-Time Latency Benchmark (Batch Size = 1, N = 50,000)
    latency_stats = benchmark_single_sample_latency(model, scaler, wednesday_file, n_benchmark_samples=BENCHMARK_LATENCY_SAMPLES)

    # 3. 100% Wednesday Dataset Streaming Evaluation
    y_test, probabilities_test, stream_stats, profile_data = evaluate_wednesday_streaming(model, scaler, wednesday_file)

    # Compute ROC-AUC directly on 100% data
    roc_auc_val = float(roc_auc_score(y_test, probabilities_test))

    # Compute Default and Calibrated Metrics
    metrics_default = evaluate_predictions(y_test, probabilities_test, threshold=0.50, roc_auc_val=roc_auc_val)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, threshold=optimal_threshold, roc_auc_val=roc_auc_val)

    print("\n>> Generating publication-grade thesis figures...")

    # Output filepaths
    path_bar_default = out("lr_wednesday_metrics_default.png")
    path_bar_calibrated = out("lr_wednesday_metrics_calibrated.png")
    path_cm_default = out("lr_wednesday_confusion_matrix_default.png")
    path_cm_calibrated = out("lr_wednesday_confusion_matrix_calibrated.png")
    path_latency = out("lr_latency_percentiles.png")
    path_memory = out("lr_memory_usage_over_time.png")
    path_cpu = out("lr_cpu_usage_over_time.png")

    # Generate Figures
    save_comprehensive_metrics_bar(metrics_default, "T = 0.50", path_bar_default, is_calibrated=False)
    save_comprehensive_metrics_bar(metrics_calibrated, f"T = {optimal_threshold:.4f}", path_bar_calibrated, is_calibrated=True)
    save_single_confusion_matrix(y_test, metrics_default["y_pred"], "Confusion Matrix (T = 0.50)", "default threshold", path_cm_default, cmap="Blues")
    save_single_confusion_matrix(y_test, metrics_calibrated["y_pred"], f"Confusion Matrix (Calibrated T = {optimal_threshold:.4f})", "calibrated threshold", path_cm_calibrated, cmap="Greens")
    save_realtime_latency_percentiles_plot(latency_stats, path_latency)
    save_resource_usage_plots(profile_data, path_memory, path_cpu)

    # Export JSON Payload
    metrics_def_clean = {k: v for k, v in metrics_default.items() if k != "y_pred"}
    metrics_cal_clean = {k: v for k, v in metrics_calibrated.items() if k != "y_pred"}

    results_payload = {
        "model": "Logistic Regression",
        "features": FEATURE_COLUMNS,
        "friday_calibration": {"optimal_threshold": float(optimal_threshold), "f2_score": float(val_f2)},
        "pure_realtime_latency_batch_size_1": latency_stats,
        "wednesday_stream_stats": stream_stats,
        "metrics_default_threshold_0_50": metrics_def_clean,
        "metrics_calibrated_threshold": metrics_cal_clean,
    }

    json_path = out("lr_wednesday_crossday_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    # Print Formatted Results
    print("\n" + "=" * 78)
    print(" LR CROSS-DAY EVALUATION SUMMARY")
    print("=" * 78)
    print(f"{'Metric':<22}{'Default (T=0.50)':>24}{f'Calibrated (T={optimal_threshold:.4f})':>27}")
    print("-" * 78)
    for m in ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]:
        print(f"{m.upper():<22}{metrics_default[m]:>24.4f}{metrics_calibrated[m]:>27.4f}")
    print("-" * 78)
    print(" HARDWARE RESOURCE UTILIZATION:")
    print(f"   Peak RAM (RSS):       {stream_stats['peak_memory_mb']:.2f} MB")
    print(f"   Average RAM (RSS):    {stream_stats['avg_memory_mb']:.2f} MB")
    print(f"   Peak CPU Load:        {stream_stats['peak_cpu_percent']:.2f}%")
    print(f"   Average CPU Load:     {stream_stats['avg_cpu_percent']:.2f}%")
    print("-" * 78)
    print(" PURE REAL-TIME INFERENCE LATENCY (BATCH SIZE = 1):")
    print(f"   Mean Latency: {latency_stats['mean_latency_ms']:.4f} ms/sample")
    print(f"   P50 (Median): {latency_stats['p50_latency_ms']:.4f} ms/sample")
    print(f"   P90:          {latency_stats['p90_latency_ms']:.4f} ms/sample")
    print(f"   P95:          {latency_stats['p95_latency_ms']:.4f} ms/sample")
    print(f"   P99:          {latency_stats['p99_latency_ms']:.4f} ms/sample")
    print(f"   Throughput:   {latency_stats['throughput_samples_per_sec']:.2f} samples/sec")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
