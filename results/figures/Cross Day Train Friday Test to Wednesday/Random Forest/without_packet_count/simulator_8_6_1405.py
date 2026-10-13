#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAM-Optimized Fair Cross-Day Evaluation (Without Packet_Count Feature)
Train domain: Friday DDoS
Test domain : Wednesday

Ablation version with Real-Time Latency Benchmark:
- Excludes "Packet_Count" feature group
- Friday threshold calibration by F2-score on remaining 3 features
- Fast chunked cross-day prediction (vectorized)
- Dedicated single-sample (Batch Size = 1) streaming benchmark for latency profiling (P50, P90, P95, P99)
- Separate metric plots for Default (T=0.50) and Calibrated thresholds
- Uniform thesis-ready plots (Times New Roman, 10pt font, 300 DPI, standardized units/legends)
"""

import os
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil

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
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# 1. Paths and Global Configuration (Thesis Standards: 10pt, Times New Roman, 300 DPI)
# ============================================================

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_FILENAME = "random_forest_friday_model_without_packet_count.pkl"

MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)
CHUNK_SIZE = 25000
BENCHMARK_SAMPLE_COUNT = 5000  # Number of single samples for high-resolution per-packet latency

FEATURE_COLUMNS = [
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

CANONICAL_RAW_COLUMNS = [
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


# ============================================================
# 2. Helpers
# ============================================================

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
    raise FileNotFoundError(
        f"\n[ERROR] File not found: {filename}\nChecked paths:\n- " + "\n- ".join(candidate_paths)
    )


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def resolve_csv_columns(file_path):
    header_df = pd.read_csv(file_path, nrows=0)
    actual_columns = list(header_df.columns)

    normalized_to_actual = {}
    for actual_name in actual_columns:
        normalized_to_actual[str(actual_name).strip()] = actual_name

    missing_columns = [col for col in CANONICAL_RAW_COLUMNS if col not in normalized_to_actual]
    if missing_columns:
        raise ValueError(
            "\n[ERROR] Required columns were not found in the CSV header.\n"
            f"Missing canonical columns: {missing_columns}\n"
            f"Available columns:\n{actual_columns}"
        )

    usecols = [normalized_to_actual[col] for col in CANONICAL_RAW_COLUMNS]
    return usecols


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

        avg_packet_size = pd.to_numeric(raw_chunk["Average Packet Size"], errors="coerce")
        flow_rate = pd.to_numeric(raw_chunk["Flow Packets/s"], errors="coerce")
        flow_duration = pd.to_numeric(raw_chunk["Flow Duration"], errors="coerce")
        labels = raw_chunk["Label"].astype(str).str.strip().str.upper()

        prepared = pd.DataFrame({
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

    print(f"[INFO] Counting valid samples in: {os.path.basename(file_path)}")

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

def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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
        "false_positive_rate": float(fpr_value),
        "false_negative_rate": float(fnr_value),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "y_pred": y_pred,
    }


def calibrate_threshold_on_friday_validation(model, friday_file):
    print("\n[INFO] Starting threshold calibration on Friday validation data...")

    friday_stats = get_dataset_statistics(friday_file)
    total_samples = friday_stats["n_rows_clean"]

    if total_samples < 2:
        raise ValueError("[ERROR] Friday dataset has insufficient valid samples.")

    validation_start_index = int(total_samples * 0.80)

    print(
        f"[INFO] Friday clean samples: {total_samples:,} | "
        f"Temporal validation samples: {total_samples - validation_start_index:,}"
    )

    validation_y_parts = []
    validation_prob_parts = []
    processed_samples = 0

    for X_chunk, y_chunk in iter_prepared_chunks(friday_file):
        chunk_start = processed_samples
        chunk_end = processed_samples + len(y_chunk)
        processed_samples = chunk_end

        if chunk_end <= validation_start_index:
            del X_chunk, y_chunk
            gc.collect()
            continue

        local_start = max(0, validation_start_index - chunk_start)
        X_validation = X_chunk[local_start:]
        y_validation = y_chunk[local_start:]

        if len(y_validation) > 0:
            probabilities = model.predict_proba(X_validation)[:, 1].astype(np.float32)
            validation_y_parts.append(y_validation.copy())
            validation_prob_parts.append(probabilities)
            del probabilities

        del X_chunk, y_chunk, X_validation, y_validation
        gc.collect()

    if not validation_y_parts:
        raise ValueError("[ERROR] Friday validation partition could not be created.")

    y_validation_all = np.concatenate(validation_y_parts).astype(np.uint8)
    prob_validation_all = np.concatenate(validation_prob_parts).astype(np.float32)

    del validation_y_parts, validation_prob_parts
    gc.collect()

    if len(np.unique(y_validation_all)) < 2:
        raise ValueError("[ERROR] Friday temporal validation partition contains only one class.")

    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.50
    best_f2 = -1.0

    for threshold in thresholds:
        y_pred = (prob_validation_all >= threshold).astype(np.uint8)
        current_f2 = fbeta_score(y_validation_all, y_pred, beta=2, zero_division=0)
        if current_f2 > best_f2:
            best_f2 = current_f2
            best_threshold = float(threshold)

    print(
        f"[INFO] Friday calibration finished: "
        f"best threshold = {best_threshold:.2f}, validation F2-score = {best_f2:.6f}"
    )

    del y_validation_all, prob_validation_all
    gc.collect()

    return best_threshold, float(best_f2), friday_stats


# ============================================================
# 4. Fast Cross-Day Streaming & Dedicated Real-Time Latency Benchmark
# ============================================================

def evaluate_wednesday_fast_streaming(model, wednesday_file):
    """Vectorized chunk evaluation for rapid full-dataset classification and resource monitoring."""
    print("\n[INFO] Starting Fast Streaming Evaluation on Wednesday test data...")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    probability_parts = []
    memory_trace_mb = []
    cpu_trace_pct = []
    chunk_latency_trace_ms = []
    benchmark_samples = []

    for chunk_id, (X_chunk, y_chunk) in enumerate(iter_prepared_chunks(wednesday_file), start=1):
        mem_before = get_memory_usage_mb()

        if sum(len(c) for c in benchmark_samples) < BENCHMARK_SAMPLE_COUNT:
            remain = BENCHMARK_SAMPLE_COUNT - sum(len(c) for c in benchmark_samples)
            benchmark_samples.append(X_chunk[:remain])

        t0 = time.perf_counter()
        probs = model.predict_proba(X_chunk)[:, 1].astype(np.float32)
        t1 = time.perf_counter()

        dt_chunk_ms = (t1 - t0) * 1000.0
        avg_chunk_sample_lat_ms = dt_chunk_ms / len(y_chunk)

        mem_after = get_memory_usage_mb()
        cpu_pct = process.cpu_percent(interval=None)

        y_parts.append(y_chunk.copy())
        probability_parts.append(probs)

        memory_trace_mb.append(max(mem_before, mem_after))
        cpu_trace_pct.append(cpu_pct)
        chunk_latency_trace_ms.append(avg_chunk_sample_lat_ms)

        print(
            f"[INFO] Chunk {chunk_id:03d} | "
            f"Samples: {len(y_chunk):,} | "
            f"Vector Latency: {avg_chunk_sample_lat_ms:.5f} ms/sample | "
            f"Memory: {max(mem_before, mem_after):.2f} MB | "
            f"CPU: {cpu_pct:.2f}%"
        )

        del X_chunk, y_chunk, probs
        gc.collect()

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities_test = np.concatenate(probability_parts).astype(np.float32)

    if benchmark_samples:
        X_benchmark = np.concatenate(benchmark_samples, axis=0)[:BENCHMARK_SAMPLE_COUNT]
    else:
        X_benchmark = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)

    del y_parts, probability_parts, benchmark_samples
    gc.collect()

    print(f"\n[INFO] Running Single-Sample Real-Time Benchmark on {len(X_benchmark):,} samples (Batch Size = 1)...")
    single_sample_latencies = np.zeros(len(X_benchmark), dtype=np.float64)

    for i in range(min(50, len(X_benchmark))):
        _ = model.predict_proba(X_benchmark[i:i+1])

    t_bench_start = time.perf_counter()
    for idx in range(len(X_benchmark)):
        x_single = X_benchmark[idx:idx+1]
        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()
        single_sample_latencies[idx] = (t1 - t0) * 1000.0
    total_pure_bench_time = time.perf_counter() - t_bench_start

    latency_stats = {
        "batch_size": 1,
        "benchmark_samples": int(len(X_benchmark)),
        "total_inference_time_sec": float(total_pure_bench_time),
        "average_latency_ms": float(np.mean(single_sample_latencies)),
        "latency_p50_ms": float(np.percentile(single_sample_latencies, 50)),
        "latency_p90_ms": float(np.percentile(single_sample_latencies, 90)),
        "latency_p95_ms": float(np.percentile(single_sample_latencies, 95)),
        "latency_p99_ms": float(np.percentile(single_sample_latencies, 99)),
        "n_total_test_samples": int(len(y_test)),
        "chunk_size": int(CHUNK_SIZE),
        "peak_memory_mb": float(np.max(memory_trace_mb)) if memory_trace_mb else 0.0,
        "avg_cpu_percent": float(np.mean(cpu_trace_pct)) if cpu_trace_pct else 0.0,
    }

    profile_data = {
        "memory_trace_mb": np.array(memory_trace_mb, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace_pct, dtype=np.float32),
        "latency_trace_ms": np.array(chunk_latency_trace_ms, dtype=np.float32),
        "sample_latencies_ms": single_sample_latencies,
    }

    del X_benchmark
    gc.collect()

    return y_test, probabilities_test, latency_stats, profile_data


def profile_prediction_runs(model, wednesday_file, repeats=3, max_samples=1000):
    collected_samples = []
    for X_chunk, _ in iter_prepared_chunks(wednesday_file):
        collected_samples.append(X_chunk)
        if sum(len(c) for c in collected_samples) >= max_samples:
            break

    if not collected_samples:
        return np.array([], dtype=np.float32)

    X_bench = np.concatenate(collected_samples, axis=0)[:max_samples]
    run_times_sec = []

    for run_id in range(repeats):
        t0 = time.perf_counter()
        for i in range(len(X_bench)):
            _ = model.predict_proba(X_bench[i:i+1])
        dt = time.perf_counter() - t0
        run_times_sec.append(dt)
        print(f"[INFO] Benchmark Run {run_id+1}/{repeats} ({len(X_bench)} single samples): {dt:.4f} s")

    del collected_samples, X_bench
    gc.collect()

    return np.array(run_times_sec, dtype=np.float32)


# ============================================================
# 5. Plotting (Thesis Standardized: Times New Roman, 10pt, 300 DPI)
# ============================================================

def apply_thesis_style(ax):
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_single_metrics_bar_plot(metrics, roc_auc_val, threshold_label, color, output_path):
    """Generates a standalone, dedicated bar plot for a single threshold configuration."""
    ensure_parent_dir(output_path)

    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    vals = [metrics[k] for k in metric_keys] + [roc_auc_val]

    x = np.arange(len(metric_labels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    bars = ax.bar(
        x,
        vals,
        width,
        label=threshold_label,
        color=color,
        edgecolor="black",
        linewidth=0.7,
    )

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score (Unitless [0.0–1.0])", fontsize=10)
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    apply_thesis_style(ax)

    caption_text = f"Figure: Cross-day classification metrics on Wednesday test dataset under {threshold_label}."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_all_metrics_bar_plot(metrics_default, metrics_calibrated, roc_auc_val, output_path):
    ensure_parent_dir(output_path)

    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]

    vals_def = [metrics_default[k] for k in metric_keys] + [roc_auc_val]
    vals_cal = [metrics_calibrated[k] for k in metric_keys] + [roc_auc_val]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.8, 4.3), dpi=300)
    bars1 = ax.bar(
        x - width / 2,
        vals_def,
        width,
        label="Default Threshold (T = 0.50)",
        color="#3470a3",
        edgecolor="black",
        linewidth=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        vals_cal,
        width,
        label=f"Calibrated Threshold (T = {metrics_calibrated['threshold']:.2f})",
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.7,
    )

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score (Unitless [0.0–1.0])", fontsize=10)
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    apply_thesis_style(ax)

    caption_text = (
        "Figure: Cross-day classification metrics comparison on Wednesday test dataset "
        "under default and calibrated thresholds."
    )
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_single_confusion_matrix(y_true, y_pred, title, output_path, cmap="Blues"):
    ensure_parent_dir(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.6, 3.8), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        annot_kws={"size": 10},
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.tick_params(axis="both", labelsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix_comparison(y_test, probabilities, threshold_default, threshold_calibrated, output_path):
    pred_default = (probabilities >= threshold_default).astype(np.uint8)
    pred_calibrated = (probabilities >= threshold_calibrated).astype(np.uint8)

    cm_default = confusion_matrix(y_test, pred_default, labels=[0, 1])
    cm_calibrated = confusion_matrix(y_test, pred_calibrated, labels=[0, 1])

    ensure_parent_dir(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=300)

    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0], annot_kws={"size": 10})
    axes[0].set_title(f"(a) Standard Threshold (T = {threshold_default:.2f})", fontsize=10)
    axes[0].set_xlabel("Predicted Class", fontsize=10)
    axes[0].set_ylabel("True Class", fontsize=10)
    axes[0].set_xticklabels(["Benign", "Attack"], fontsize=10)
    axes[0].set_yticklabels(["Benign", "Attack"], rotation=0, fontsize=10)

    sns.heatmap(cm_calibrated, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1], annot_kws={"size": 10})
    axes[1].set_title(f"(b) Calibrated Threshold (T = {threshold_calibrated:.2f})", fontsize=10)
    axes[1].set_xlabel("Predicted Class", fontsize=10)
    axes[1].set_ylabel("True Class", fontsize=10)
    axes[1].set_xticklabels(["Benign", "Attack"], fontsize=10)
    axes[1].set_yticklabels(["Benign", "Attack"], rotation=0, fontsize=10)

    caption_text = "Figure: Confusion matrix comparison on Wednesday test dataset under standard and calibrated thresholds."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_pr_figure(y_test, probabilities, output_path):
    ensure_parent_dir(output_path)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_score_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=300)

    axes[0].plot(fpr, tpr, color="#D55E00", lw=1.8, label=f"RF (AUC = {roc_auc_score_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Baseline")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
    axes[0].set_title("(a) Receiver Operating Characteristic (ROC)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)

    axes[1].plot(recall, precision, color="#0072B2", lw=1.8, label=f"RF (AP = {average_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)

    caption_text = "Figure: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves for Wednesday dataset."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(roc_auc_score_val), float(average_precision), fpr, tpr, precision, recall


def save_latency_percentiles(sample_latencies_ms, output_path):
    p50, p90, p95, p99 = np.percentile(sample_latencies_ms, [50, 90, 95, 99])
    mean_val = np.mean(sample_latencies_ms)

    labels = ["Mean", "P50 (Median)", "P90", "P95", "P99"]
    values = [mean_val, p50, p90, p95, p99]

    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6.8, 3.9), dpi=300)
    bars = ax.bar(labels, values, color="#5c6f84", width=0.55, edgecolor="black", linewidth=0.7)

    offset = max(values) * 0.02 if max(values) > 0 else 0.001
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:.4f}", ha="center", fontsize=9)

    ax.set_ylabel("Inference Latency (ms/sample)", fontsize=10)
    ax.set_xlabel("Latency Statistic", fontsize=10)
    ax.set_title("Pure Real-Time Single-Sample Latency Profile", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    apply_thesis_style(ax)

    caption_text = "Figure: Single-sample pure inference latency statistics for real-time edge evaluation (batch_size = 1)."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(p50), float(p90), float(p95), float(p99)


def save_memory_usage_plot(memory_trace_mb, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    ax.plot(
        np.linspace(0, 1, len(memory_trace_mb)),
        memory_trace_mb,
        color="#C44E52",
        linewidth=1.5,
        label="RSS Memory Footprint",
    )
    ax.axhline(y=1024.0, color="r", linestyle="--", linewidth=1.2, label="1 GB Edge RAM Budget")
    ax.set_title("Edge Memory Footprint Profile Over Evaluation", fontsize=10)
    ax.set_xlabel("Normalized Timeline", fontsize=10)
    ax.set_ylabel("Memory Consumption (MB)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    apply_thesis_style(ax)

    caption_text = "Figure: Resident memory (RSS) footprint during streaming cross-day evaluation."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_cpu_usage_plot(cpu_trace_pct, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    ax.plot(
        np.linspace(0, 1, len(cpu_trace_pct)),
        cpu_trace_pct,
        color="#55A868",
        linewidth=1.5,
        label="Process CPU Utilization",
    )
    ax.set_title("Processor Load Profile Over Evaluation", fontsize=10)
    ax.set_xlabel("Normalized Timeline", fontsize=10)
    ax.set_ylabel("CPU Utilization (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    apply_thesis_style(ax)

    caption_text = "Figure: CPU utilization percentage profile over the streaming evaluation duration."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_probability_distribution(y_true, probabilities, threshold, output_path):
    ensure_parent_dir(output_path)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    sns.histplot(
        x=probabilities,
        hue=y_true,
        bins=50,
        kde=True,
        stat="density",
        alpha=0.6,
        palette={0: "#34495e", 1: "#e74c3c"},
        hue_order=[0, 1],
        ax=ax,
    )
    ax.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold (T = {threshold:.4f})",
    )
    ax.set_title(f"Attack Probability Distribution (T = {threshold:.4f})", fontsize=10)
    ax.set_xlabel("Predicted Attack Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    apply_thesis_style(ax)

    caption_text = "Figure: Predicted attack probability distribution by class with calibrated decision threshold."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_probability_distribution_comparison(y_true, probabilities, threshold_default, threshold_calibrated, output_path):
    ensure_parent_dir(output_path)
    df = pd.DataFrame({"Probability": probabilities, "Label": y_true})

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=300, sharey=True)

    sns.histplot(
        data=df,
        x="Probability",
        hue="Label",
        bins=50,
        kde=True,
        alpha=0.5,
        palette={0: "#34495e", 1: "#e74c3c"},
        ax=axes[0],
        stat="density",
        hue_order=[0, 1],
    )
    axes[0].axvline(x=threshold_default, color="black", linestyle="--", linewidth=1.2, label=f"T = {threshold_default:.2f}")
    axes[0].set_title(f"(a) Standard Threshold (T = {threshold_default:.2f})", fontsize=10)
    axes[0].set_xlabel("Predicted Probability", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    apply_thesis_style(axes[0])

    sns.histplot(
        data=df,
        x="Probability",
        hue="Label",
        bins=50,
        kde=True,
        alpha=0.5,
        palette={0: "#34495e", 1: "#e74c3c"},
        ax=axes[1],
        stat="density",
        hue_order=[0, 1],
    )
    axes[1].axvline(x=threshold_calibrated, color="black", linestyle="--", linewidth=1.2, label=f"T = {threshold_calibrated:.2f}")
    axes[1].set_title(f"(b) Calibrated Threshold (T = {threshold_calibrated:.2f})", fontsize=10)
    axes[1].set_xlabel("Predicted Probability", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    apply_thesis_style(axes[1])

    caption_text = "Figure: Comparison of predicted attack probability distributions under standard and calibrated thresholds."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_latency_stability(latency_trace_ms, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=300)
    ax.plot(latency_trace_ms, color="#C44E52", linewidth=1.2, label="Measured Chunk Latency")
    ax.axhline(y=np.mean(latency_trace_ms), color="#2ca02c", linestyle="--", label="Mean Latency")
    ax.set_xlabel("Evaluation Chunk Index", fontsize=10)
    ax.set_ylabel("Latency (ms/sample)", fontsize=10)
    ax.set_title("Latency Stability Over Evaluation Chunks", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    apply_thesis_style(ax)

    caption_text = "Figure: Chunk-level latency stability during cross-day streaming evaluation."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_prediction_time_runs(run_times_sec, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax.plot(np.arange(1, len(run_times_sec) + 1), run_times_sec, marker="o", linewidth=1.5, color="#4C72B0")
    ax.set_xlabel("Benchmark Run Index", fontsize=10)
    ax.set_ylabel("Total Execution Time (s)", fontsize=10)
    ax.set_title("Prediction Time Across Benchmark Iterations", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    apply_thesis_style(ax)

    caption_text = "Figure: Total execution time across repeated batch-size-1 benchmark runs."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 6. JSON / Console
# ============================================================

def save_json(payload, output_path):
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


def print_final_results(metrics_default, metrics_calibrated, latency_stats, roc_auc_score_val, average_precision):
    print("\n" + "=" * 78)
    print(" FINAL ABLATED RANDOM FOREST (NO PACKET_COUNT) REAL-TIME EVALUATION")
    print(" Training Dataset: Friday DDoS | Testing Dataset: Wednesday (Batch Size = 1 Benchmark)")
    print("=" * 78)

    print(f"{'Metric':<22}{'Standard (T=0.50)':>24}{'Calibrated Threshold':>27}")
    print("-" * 78)

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
        ("F2-score", "f2"),
        ("Specificity", "specificity"),
        ("MCC", "mcc"),
    ]

    for display_name, key in metric_names:
        print(
            f"{display_name:<22}"
            f"{metrics_default[key]:>24.6f}"
            f"{metrics_calibrated[key]:>27.6f}"
        )

    print("-" * 78)
    print(f"ROC-AUC:                         {roc_auc_score_val:.6f}")
    print(f"Average Precision (PR-AUC):      {average_precision:.6f}")
    print(f"Inference Mode:                  Real-Time Streaming (Batch Size = 1 Profiled)")
    print(f"Benchmark Execution Time:        {latency_stats['total_inference_time_sec']:.4f} s")
    print(f"Mean Latency:                    {latency_stats['average_latency_ms']:.5f} ms/sample")
    print(f"P50 Latency (Median):            {latency_stats['latency_p50_ms']:.5f} ms/sample")
    print(f"P90 Latency:                     {latency_stats['latency_p90_ms']:.5f} ms/sample")
    print(f"P95 Latency:                     {latency_stats['latency_p95_ms']:.5f} ms/sample")
    print(f"P99 Latency (Tail SLA):          {latency_stats['latency_p99_ms']:.5f} ms/sample")
    print(f"Peak Memory Usage:               {latency_stats['peak_memory_mb']:.2f} MB")
    print(f"Average CPU Usage:               {latency_stats['avg_cpu_percent']:.2f} %")
    print("=" * 78 + "\n")


# ============================================================
# 7. Main Execution Flow
# ============================================================

def main():
    print("[INFO] Starting Optimized Cross-Day Ablation Real-Time Pipeline...")

    friday_file = find_file(FRIDAY_FILENAME)
    wednesday_file = find_file(WEDNESDAY_FILENAME)

    if not os.path.exists(MODEL_PATH):
        model_path = find_file(MODEL_FILENAME)
    else:
        model_path = MODEL_PATH

    print(f"[INFO] Results directory : {RESULTS_DIR}")
    print(f"[INFO] Friday dataset    : {friday_file}")
    print(f"[INFO] Wednesday dataset : {wednesday_file}")
    print(f"[INFO] Model             : {model_path}")

    print("\n[INFO] Loading trained Random Forest model (Ablated)...")
    model = joblib.load(model_path)

    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
        print("[INFO] Random Forest n_jobs set to 1 for precise edge evaluation.")

    optimal_threshold, friday_validation_f2, friday_stats = calibrate_threshold_on_friday_validation(model, friday_file)
    gc.collect()

    print("\n[INFO] Counting Wednesday clean samples...")
    wednesday_stats = get_dataset_statistics(wednesday_file)

    print(
        f"[INFO] Wednesday clean samples: {wednesday_stats['n_rows_clean']:,} | "
        f"Benign: {wednesday_stats['n_benign']:,} | "
        f"Attack: {wednesday_stats['n_attack']:,}"
    )

    y_test, probabilities_test, latency_stats, profile_data = evaluate_wednesday_fast_streaming(model, wednesday_file)

    default_threshold = 0.50
    metrics_default = evaluate_predictions(y_test, probabilities_test, default_threshold)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, optimal_threshold)

    print("\n[INFO] Running repeated timing benchmark...")
    run_times_sec = profile_prediction_runs(model, wednesday_file, repeats=3, max_samples=1000)

    print("\n[INFO] Generating unified manuscript-ready figures...")

    paths = {
        "confusion_matrix_default": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrix_default_no_packet_count.png"),
        "confusion_matrix_calibrated": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrix_calibrated_no_packet_count.png"),
        "confusion_matrix_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrices_comparison_no_packet_count.png"),
        "roc_pr_curves": os.path.join(RESULTS_DIR, "rf_wednesday_roc_pr_curves_no_packet_count.png"),
        "metrics_default": os.path.join(RESULTS_DIR, "rf_wednesday_metrics_default_no_packet_count.png"),
        "metrics_calibrated": os.path.join(RESULTS_DIR, "rf_wednesday_metrics_calibrated_no_packet_count.png"),
        "metrics_all_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_all_metrics_comparison_no_packet_count.png"),
        "probability_distribution_default": os.path.join(RESULTS_DIR, "rf_probability_distribution_default_no_packet_count.png"),
        "probability_distribution_calibrated": os.path.join(RESULTS_DIR, "rf_probability_distribution_calibrated_no_packet_count.png"),
        "probability_distribution_comparison": os.path.join(RESULTS_DIR, "rf_probability_distribution_comparison_no_packet_count.png"),
        "memory_usage": os.path.join(RESULTS_DIR, "rf_memory_usage_over_time_no_packet_count.png"),
        "cpu_usage": os.path.join(RESULTS_DIR, "rf_cpu_usage_over_time_no_packet_count.png"),
        "latency_percentiles": os.path.join(RESULTS_DIR, "rf_latency_percentiles_no_packet_count.png"),
        "latency_stability": os.path.join(RESULTS_DIR, "rf_latency_stability_over_time_no_packet_count.png"),
        "prediction_time_runs": os.path.join(RESULTS_DIR, "rf_prediction_time_runs_no_packet_count.png"),
    }

    roc_auc_score_val, average_precision, fpr, tpr, precision_curve, recall_curve = save_roc_pr_figure(
        y_test, probabilities_test, paths["roc_pr_curves"]
    )

    save_single_confusion_matrix(
        y_test,
        metrics_default["y_pred"],
        "Confusion Matrix (T = 0.50)",
        paths["confusion_matrix_default"],
        cmap="Blues",
    )

    save_single_confusion_matrix(
        y_test,
        metrics_calibrated["y_pred"],
        f"Confusion Matrix (Calibrated T = {optimal_threshold:.2f})",
        paths["confusion_matrix_calibrated"],
        cmap="Greens",
    )

    save_confusion_matrix_comparison(
        y_test, probabilities_test, default_threshold, optimal_threshold, paths["confusion_matrix_comparison"]
    )

    # 1. Standalone Bar Plot: Default Threshold (T = 0.50)
    save_single_metrics_bar_plot(
        metrics=metrics_default,
        roc_auc_val=roc_auc_score_val,
        threshold_label="Default Threshold (T = 0.50)",
        color="#3470a3",
        output_path=paths["metrics_default"],
    )

    # 2. Standalone Bar Plot: Calibrated Threshold (T = optimal)
    save_single_metrics_bar_plot(
        metrics=metrics_calibrated,
        roc_auc_val=roc_auc_score_val,
        threshold_label=f"Calibrated Threshold (T = {optimal_threshold:.2f})",
        color="#2ca02c",
        output_path=paths["metrics_calibrated"],
    )

    # 3. Comparative Bar Plot: Both Thresholds Side-by-Side
    save_all_metrics_bar_plot(
        metrics_default,
        metrics_calibrated,
        roc_auc_score_val,
        paths["metrics_all_comparison"],
    )

    save_probability_distribution(
        y_test,
        probabilities_test,
        default_threshold,
        paths["probability_distribution_default"],
    )
    save_probability_distribution(
        y_test,
        probabilities_test,
        optimal_threshold,
        paths["probability_distribution_calibrated"],
    )
    save_probability_distribution_comparison(
        y_test,
        probabilities_test,
        default_threshold,
        optimal_threshold,
        paths["probability_distribution_comparison"],
    )

    save_memory_usage_plot(profile_data["memory_trace_mb"], paths["memory_usage"])
    save_cpu_usage_plot(profile_data["cpu_trace_pct"], paths["cpu_usage"])
    save_latency_percentiles(profile_data["sample_latencies_ms"], paths["latency_percentiles"])
    save_latency_stability(profile_data["latency_trace_ms"], paths["latency_stability"])
    save_prediction_time_runs(run_times_sec, paths["prediction_time_runs"])

    metrics_default_json = {k: v for k, v in metrics_default.items() if k != "y_pred"}
    metrics_calibrated_json = {k: v for k, v in metrics_calibrated.items() if k != "y_pred"}

    results_payload = {
        "metadata": {
            "model_type": "RandomForestClassifier",
            "ablation_status": "Packet_Count excluded",
            "features_used": FEATURE_COLUMNS,
            "train_domain": os.path.basename(friday_file),
            "test_domain": os.path.basename(wednesday_file),
            "real_time_batch_size": 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "friday_calibration_statistics": friday_stats,
        "wednesday_test_statistics": wednesday_stats,
        "threshold_calibration": {
            "method": "Temporal Friday validation split (last 20%)",
            "optimization_metric": "F2-score",
            "default_threshold": float(default_threshold),
            "optimal_threshold": float(optimal_threshold),
            "friday_validation_f2_score": float(friday_validation_f2),
        },
        "inference_performance": {
            **latency_stats,
            "prediction_time_runs_sec": run_times_sec.tolist(),
        },
        "discrimination_metrics": {
            "roc_auc": float(roc_auc_score_val),
            "average_precision_pr_auc": float(average_precision),
        },
        "evaluation_default_threshold": metrics_default_json,
        "evaluation_calibrated_threshold": metrics_calibrated_json,
        "generated_figures": paths,
    }

    json_output_path = os.path.join(RESULTS_DIR, "rf_wednesday_crossday_results_no_packet_count.json")
    save_json(results_payload, json_output_path)

    print(f"[INFO] JSON ablation results saved to: {json_output_path}")
    for k, v in paths.items():
        print(f"[INFO] Figure saved [{k}]: {v}")

    print_final_results(
        metrics_default=metrics_default,
        metrics_calibrated=metrics_calibrated,
        latency_stats=latency_stats,
        roc_auc_score_val=roc_auc_score_val,
        average_precision=average_precision,
    )

    del model, y_test, probabilities_test
    gc.collect()

    print("[INFO] Cross-day ablation real-time evaluation completed successfully.")


if __name__ == "__main__":
    main()
