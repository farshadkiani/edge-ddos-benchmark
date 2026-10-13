#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAM-Optimized Gold-Standard Cross-Day Evaluation for Logistic Regression
Ablation: WITHOUT Flow_Duration (features kept: Packet_Count, Packet_Size, Flow_Rate)

Dual-Phase Protocol (Gold Standard):
  - Phase 1 (Qualitative): Vectorized evaluation on 100% of Wednesday dataset
      -> Accuracy, Precision, Recall, F1, F2, MCC, ROC-AUC, Confusion Matrix
  - Phase 2 (Hardware/Latency): Pure single-sample inference (batch_size=1)
      on 5,000 samples via time.perf_counter()
      -> Mean, P50, P90, P95, P99 computed directly from raw latency array

Visualization: Times New Roman, 10pt, 300 DPI, Legend, Units, Captions
"""

import os
import gc
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. Global Figure Configuration (Thesis Standard: TNR, 10pt, 300 DPI)
# ============================================================

FIG_DPI = 300
FIG_FONT_FAMILY = "Times New Roman"
FIG_FONT_SIZE = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [FIG_FONT_FAMILY, "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": FIG_FONT_SIZE,
    "axes.titlesize": FIG_FONT_SIZE,
    "axes.labelsize": FIG_FONT_SIZE,
    "xtick.labelsize": FIG_FONT_SIZE,
    "ytick.labelsize": FIG_FONT_SIZE,
    "legend.fontsize": FIG_FONT_SIZE,
    "figure.titlesize": FIG_FONT_SIZE,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "savefig.dpi": FIG_DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

# ============================================================
# 1. Global Pipeline Configuration
# ============================================================

BETA = 2.0
LATENCY_BENCHMARK_SAMPLES = 5000
CHUNK_SIZE = 15000
TRAIN_RATIO = 0.8

FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

ABLATION_NAME = "without_flow_duration"
REMOVED_FEATURE = "Flow_Duration"

# 3 Features kept in this ablation (Flow_Duration excluded)
FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Rate"]
ALL_FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]

RAW_NUMERIC_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
]

CANONICAL_RAW_COLUMNS = RAW_NUMERIC_COLUMNS + ["Label"]

RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        "results/ablation_crossday/logistic_regression/without_flow_duration",
    )
)

MODEL_FILENAME = "lr_model_without_flow_duration.joblib"
SCALER_FILENAME = "scaler_lr_without_flow_duration.joblib"
METADATA_FILENAME = "lr_metadata_without_flow_duration.json"


# ============================================================
# 2. Utility & Helper Functions
# ============================================================

def out(filename: str) -> Path:
    """Helper to return path inside RESULTS_DIR."""
    return RESULTS_DIR / filename


def find_file(filename: str, required: bool = True) -> str:
    """
    Find file in common candidate paths or environment directories.
    Raises FileNotFoundError with clear candidate trace if required is True.
    """
    candidates = [
        str(RESULTS_DIR / filename),
        os.path.join(".", filename),
        os.path.join("data", filename),
        os.path.join("models", filename),
        os.path.join("results", filename),
        os.path.join("/mnt/data", filename),
        os.path.join("/app", filename),
        os.path.join("..", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
            
    if required:
        searched_paths = "\n".join([f"  - {os.path.abspath(p)}" for p in candidates])
        raise FileNotFoundError(
            f"\n[ERROR] Required file '{filename}' was not found in any of these candidate paths:\n"
            f"{searched_paths}\n"
            f"Please ensure the model training script has run or place the file in one of the searched directories."
        )
    return filename


def resolve_csv_columns(columns):
    """Strip whitespace and match canonical columns."""
    mapping = {}
    clean_cols = {c.strip(): c for c in columns}
    for canon in CANONICAL_RAW_COLUMNS:
        if canon in clean_cols:
            mapping[canon] = clean_cols[canon]
        else:
            for c in clean_cols:
                if c.lower() == canon.lower():
                    mapping[canon] = clean_cols[c]
                    break
    return mapping


def iter_prepared_chunks(csv_path, chunk_size=CHUNK_SIZE):
    """
    Streaming feature engineering.
    Fairness rule: rows are kept only if ALL raw features are
    valid (identical filter across every ablation -> fair comparison).
    """
    for raw_chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        resolved = resolve_csv_columns(raw_chunk.columns)
        chunk = raw_chunk.rename(columns={v: k for k, v in resolved.items()})

        # Cast raw numeric features safely
        for col in RAW_NUMERIC_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=RAW_NUMERIC_COLUMNS).copy()

        # Compute engineered features
        packet_count = (chunk["Total Fwd Packets"] + chunk["Total Backward Packets"]).astype(np.float32)
        packet_size = chunk["Average Packet Size"].astype(np.float32)
        flow_rate = chunk["Flow Packets/s"].astype(np.float32)

        # Ablation: Flow_Duration is EXCLUDED from the model input
        X = np.column_stack([packet_count, packet_size, flow_rate])
        y = (chunk["Label"].astype(str).str.strip().str.upper().ne("BENIGN")).astype(np.uint8).values

        if len(y) == 0:
            continue
        yield X, y


def load_calibrated_threshold():
    """Leak-free threshold: reuse only the threshold calibrated on the Friday validation partition."""
    candidate_files = [METADATA_FILENAME, "model_metadata.json", "metadata.json"]
    for fname in candidate_files:
        try:
            meta_path = find_file(fname, required=False)
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for key in ["best_threshold", "optimal_threshold", "calibrated_threshold", "threshold"]:
                if key in meta:
                    th = float(meta[key])
                    print(f"[INFO] Calibrated threshold T = {th:.4f} loaded from {meta_path}")
                    return th

            if "threshold_calibration" in meta and "optimal_threshold" in meta["threshold_calibration"]:
                th = float(meta["threshold_calibration"]["optimal_threshold"])
                print(f"[INFO] Calibrated threshold T = {th:.4f} loaded from threshold_calibration in {meta_path}")
                return th
        except Exception:
            continue

    print("[WARNING] Calibrated threshold metadata not found. Defaulting to 0.50")
    return 0.50


def validate_metadata(model, scaler):
    expected_features = len(FEATURE_COLUMNS)
    if hasattr(model, "n_features_in_") and model.n_features_in_ != expected_features:
        raise ValueError(
            f"[ERROR] Model expects {model.n_features_in_} features, "
            f"but ablation '{ABLATION_NAME}' provides {expected_features}."
        )
    try:
        meta_path = find_file(METADATA_FILENAME, required=False)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if "features_used" in meta and meta["features_used"] != FEATURE_COLUMNS:
                print(f"[WARNING] Feature mismatch in metadata: {meta['features_used']} vs {FEATURE_COLUMNS}")
    except Exception:
        pass


# ============================================================
# 3. Dual-Phase Evaluation Core
# ============================================================

def evaluate_full_stream(model, scaler, test_file):
    """
    Phase 1: Full-dataset evaluation in vectorized chunks.
    Monitors process memory and CPU profile per chunk.
    """
    print(f"\n[PHASE 1] Full-dataset streaming evaluation on {test_file}...")
    process = psutil.Process(os.getpid())

    y_true_list = []
    prob_list = []
    mem_trace = []
    cpu_trace = []

    # Prime CPU percent measurement
    process.cpu_percent(interval=None)

    for chunk_idx, (X_chunk, y_chunk) in enumerate(iter_prepared_chunks(test_file), 1):
        X_scaled = scaler.transform(X_chunk).astype(np.float32)
        probs = model.predict_proba(X_scaled)[:, 1]

        y_true_list.append(y_chunk)
        prob_list.append(probs)

        # Profile resource usage
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_pct = process.cpu_percent(interval=None)
        mem_trace.append(mem_mb)
        cpu_trace.append(cpu_pct)

        if chunk_idx % 10 == 0:
            print(f"   Processed chunk {chunk_idx:3d} | Memory: {mem_mb:6.2f} MB | CPU: {cpu_pct:5.1f}%")

    y_test = np.concatenate(y_true_list)
    probabilities = np.concatenate(prob_list)

    profile_data = {
        "memory_trace_mb": np.array(mem_trace, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace, dtype=np.float32),
        "peak_ram_mb": float(np.max(mem_trace)),
        "avg_ram_mb": float(np.mean(mem_trace)),
        "peak_cpu_pct": float(np.max(cpu_trace)),
        "avg_cpu_pct": float(np.mean(cpu_trace)),
    }

    return y_test, probabilities, profile_data


def benchmark_realtime_latency(model, scaler, test_file, n_samples=LATENCY_BENCHMARK_SAMPLES):
    """
    Phase 2: Pure single-sample inference latency (batch_size=1)
    measured with time.perf_counter().
    """
    print(f"\n[PHASE 2] Benchmarking single-sample inference latency (N = {n_samples})...")
    samples_collected = []
    for X_chunk, _ in iter_prepared_chunks(test_file, chunk_size=5000):
        samples_collected.append(X_chunk)
        if sum(len(s) for s in samples_collected) >= n_samples:
            break

    if len(samples_collected) == 0:
        raise ValueError("[ERROR] No samples could be read for latency benchmarking.")

    X_pool = np.vstack(samples_collected)[:n_samples]
    X_pool_scaled = scaler.transform(X_pool).astype(np.float32)

    actual_samples = len(X_pool_scaled)
    latencies_ms = np.zeros(actual_samples, dtype=np.float64)

    # Warmup
    for _ in range(50):
        _ = model.predict_proba(X_pool_scaled[0:1])

    # Benchmarking single inferences
    for i in range(actual_samples):
        x_single = X_pool_scaled[i:i+1]
        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()
        latencies_ms[i] = (t1 - t0) * 1000.0

    mean_lat = float(np.mean(latencies_ms))

    return {
        "n_samples": int(actual_samples),
        "batch_size": 1,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_fps": float(1000.0 / mean_lat) if mean_lat > 0 else 0.0,
        "latencies_array": latencies_ms,
    }


def evaluate_predictions(y_true, probabilities, threshold, roc_auc_val=None):
    y_pred = (probabilities >= threshold).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=BETA, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_val) if roc_auc_val is not None else float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


# ============================================================
# 4. Standardized Figures (Times New Roman, 10pt, 300 DPI)
# ============================================================

def plot_comprehensive_metrics_bar(metrics, threshold_tag, output_path, is_calibrated=False):
    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    values = [metrics[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=FIG_DPI)
    color = "#55A868" if is_calibrated else "#4C72B0"
    legend_label = f"Threshold T = {threshold_tag}"

    bars = ax.bar(metric_names, values, color=color, edgecolor="black", linewidth=0.8, width=0.52, label=legend_label)

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=FIG_FONT_SIZE)
    ax.set_xlabel("Evaluation Metric", fontsize=FIG_FONT_SIZE)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=FIG_FONT_SIZE)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, min(val + 0.02, 1.06), f"{val:.4f}",
                ha="center", va="bottom", fontsize=FIG_FONT_SIZE)

    caption = f"Figure: Classification metrics of LR ({ABLATION_NAME}) on Wednesday dataset (T = {threshold_tag})."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def plot_metrics_comparison_grouped(metrics_default, metrics_calibrated, threshold_calibrated, output_path):
    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "F2", "MCC", "ROC-AUC"]

    vals_def = [metrics_default[k] for k in metric_keys]
    vals_cal = [metrics_calibrated[k] for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=FIG_DPI)
    bars1 = ax.bar(x - width/2, vals_def, width, label="Default (T = 0.50)", color="#4C72B0", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, vals_cal, width, label=f"Calibrated (T = {threshold_calibrated:.4f})", color="#55A868", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=FIG_FONT_SIZE)
    ax.set_xlabel("Classification Metric", fontsize=FIG_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=FIG_FONT_SIZE)
    ax.set_ylim(-0.05, 1.18)
    ax.legend(loc="lower right", frameon=True, fontsize=FIG_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    caption = "Figure: Comparative evaluation of classification metrics across default and calibrated decision thresholds."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def plot_latency_percentiles(latency_stats, output_path):
    keys = ["mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    labels = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    vals = [latency_stats[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=FIG_DPI)
    bars = ax.bar(labels, vals, color="#8172B2", edgecolor="black", linewidth=0.8, width=0.5,
                  label="Pure Inference (Batch Size = 1)")

    offset = max(vals) * 0.03 if max(vals) > 0 else 0.001
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + offset, f"{v:.4f} ms",
                ha="center", va="bottom", fontsize=FIG_FONT_SIZE)

    ax.set_ylabel("Inference Latency (ms/sample)", fontsize=FIG_FONT_SIZE)
    ax.set_xlabel("Latency Percentiles & Mean", fontsize=FIG_FONT_SIZE)
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)
    ax.legend(loc="upper left", frameon=True, fontsize=FIG_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption = (f"Figure: Pure single-sample inference latency percentiles for LR ({ABLATION_NAME}) "
               f"over {latency_stats['n_samples']:,} samples.")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, caption_label, output_path, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=FIG_DPI)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False,
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        ax=ax, annot_kws={"size": FIG_FONT_SIZE}
    )
    ax.set_title(title, fontsize=FIG_FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=FIG_FONT_SIZE)
    ax.set_ylabel("True Class", fontsize=FIG_FONT_SIZE)

    caption = f"Figure: Confusion matrix of LR ({ABLATION_NAME}) on Wednesday dataset under {caption_label}."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()


def plot_roc_pr_curves(y_test, probabilities, output_path):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_val = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    avg_precision = average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=FIG_DPI)

    axes[0].plot(fpr, tpr, color="#4C72B0", lw=1.5, label=f"LR (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Chance")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=FIG_FONT_SIZE)
    axes[0].set_ylabel("True Positive Rate", fontsize=FIG_FONT_SIZE)
    axes[0].set_title("(a) Receiver Operating Characteristic", fontsize=FIG_FONT_SIZE, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(recall_curve, precision_curve, color="#C44E52", lw=1.5, label=f"LR (AP = {avg_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=FIG_FONT_SIZE)
    axes[1].set_ylabel("Precision", fontsize=FIG_FONT_SIZE)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=FIG_FONT_SIZE, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=9)

    caption = f"Figure: Discrimination performance (ROC and PR curves) of LR ({ABLATION_NAME}) under Cross-Day evaluation."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()

    return float(roc_auc_val), float(avg_precision)


def plot_resource_profiles(profile_data, mem_path, cpu_path):
    mem_trace = profile_data["memory_trace_mb"]
    cpu_trace = profile_data["cpu_trace_pct"]
    chunks = np.arange(1, len(mem_trace) + 1)

    fig_mem, ax_mem = plt.subplots(figsize=(6.8, 3.8), dpi=FIG_DPI)
    ax_mem.plot(chunks, mem_trace, color="#4C72B0", marker="o", markersize=3, linewidth=1.5, label="Measured RSS Memory")
    ax_mem.axhline(1024, color="#C44E52", linestyle="--", linewidth=1.0, alpha=0.7, label="Raspberry Pi 1 GB Budget")
    ax_mem.set_ylabel("Resident Set Size (MB)", fontsize=FIG_FONT_SIZE)
    ax_mem.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=FIG_FONT_SIZE)
    ax_mem.set_ylim(0, max(float(np.max(mem_trace)) * 1.3, 1100.0))
    ax_mem.legend(loc="upper right", frameon=True, fontsize=FIG_FONT_SIZE)
    ax_mem.grid(True, linestyle="--", alpha=0.5)

    caption_mem = "Figure: Memory consumption profile (RSS in MB) during continuous streaming inference."
    fig_mem.text(0.5, 0.01, caption_mem, ha="center", fontsize=FIG_FONT_SIZE, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(mem_path, dpi=FIG_DPI)
    plt.close(fig_mem)

    fig_cpu, ax_cpu = plt.subplots(figsize=(6.8, 3.8), dpi=FIG_DPI)
    ax_cpu.plot(chunks, cpu_trace, color="#E1974C", marker="s", markersize=3, linewidth=1.5, label="Process CPU Load (%)")
    avg_cpu = profile_data["avg_cpu_pct"]
    ax_cpu.axhline(avg_cpu, color="#55A868", linestyle="--", linewidth=1.2, label=f"Mean CPU Load ({avg_cpu:.1f}%)")
    ax_cpu.set_ylabel("CPU Utilization (%)", fontsize=FIG_FONT_SIZE)
    ax_cpu.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=FIG_FONT_SIZE)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.legend(loc="upper right", frameon=True, fontsize=FIG_FONT_SIZE)
    ax_cpu.grid(True, linestyle="--", alpha=0.5)

    caption_cpu = "Figure: Processor load profile over streaming inference chunks across the Wednesday dataset."
    fig_cpu.text(0.5, 0.01, caption_cpu, ha="center", fontsize=FIG_FONT_SIZE, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(cpu_path, dpi=FIG_DPI)
    plt.close(fig_cpu)


# ============================================================
# 5. Main Execution Pipeline
# ============================================================

def main():
    print("=" * 84)
    print(f" LOGISTIC REGRESSION ABLATION: {ABLATION_NAME.upper()} (CROSS-DAY GOLD-STANDARD PIPELINE)")
    print("=" * 84)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wednesday_file = find_file(WEDNESDAY_FILENAME, required=True)
    model_path = find_file(MODEL_FILENAME, required=True)
    scaler_path = find_file(SCALER_FILENAME, required=True)

    print(f"[INFO] Loaded Model : {model_path}")
    print(f"[INFO] Loaded Scaler: {scaler_path}")
    print(f"[INFO] Test Dataset : {wednesday_file}")
    print(f"[INFO] Features Used: {FEATURE_COLUMNS} | Removed: {REMOVED_FEATURE}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    validate_metadata(model, scaler)

    calibrated_th = load_calibrated_threshold()

    # ---------- PHASE 2: benchmark latency (batch_size=1) ----------
    latency_stats = benchmark_realtime_latency(model, scaler, wednesday_file, n_samples=LATENCY_BENCHMARK_SAMPLES)

    # ---------- PHASE 1: full 100% Wednesday dataset ----------
    y_test, probabilities_test, profile_data = evaluate_full_stream(model, scaler, wednesday_file)

    roc_auc_val, avg_precision = plot_roc_pr_curves(y_test, probabilities_test, out("lr_without_flow_duration_roc_pr_curves.png"))

    metrics_default = evaluate_predictions(y_test, probabilities_test, threshold=0.50, roc_auc_val=roc_auc_val)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, threshold=calibrated_th, roc_auc_val=roc_auc_val)

    print("\n[INFO] Saving standardized figures (Times New Roman, 10pt, 300 DPI)...")
    plot_comprehensive_metrics_bar(metrics_default, "0.50", out("lr_without_flow_duration_metrics_default.png"), is_calibrated=False)
    plot_comprehensive_metrics_bar(metrics_calibrated, f"{calibrated_th:.4f}", out("lr_without_flow_duration_metrics_calibrated.png"), is_calibrated=True)
    plot_metrics_comparison_grouped(metrics_default, metrics_calibrated, calibrated_th, out("lr_without_flow_duration_metrics_comparison.png"))

    plot_confusion_matrix(y_test, metrics_default["y_pred"], "Confusion Matrix (T = 0.50)", "default threshold", out("lr_without_flow_duration_cm_default.png"), cmap="Blues")
    plot_confusion_matrix(y_test, metrics_calibrated["y_pred"], f"Confusion Matrix (T = {calibrated_th:.4f})", "calibrated threshold", out("lr_without_flow_duration_cm_calibrated.png"), cmap="Greens")

    plot_latency_percentiles(latency_stats, out("lr_without_flow_duration_latency_percentiles.png"))
    plot_resource_profiles(profile_data, out("lr_without_flow_duration_memory_usage.png"), out("lr_without_flow_duration_cpu_usage.png"))

    # ---------- JSON Export ----------
    results_json = {
        "experiment": {
            "model": "Logistic Regression",
            "study": "Cross-Day Ablation (without Flow_Duration)",
            "removed_feature": REMOVED_FEATURE,
            "ablation_name": ABLATION_NAME,
            "features_used": FEATURE_COLUMNS,
            "test_dataset": WEDNESDAY_FILENAME,
            "threshold_calibration_source": "Friday last-20% validation partition (leak-free)",
            "protocol": ("Dual-Phase Gold Standard: Phase 1 = qualitative metrics on 100% of the "
                         "test dataset (vectorized); Phase 2 = pure single-sample inference latency "
                         "(batch_size=1) on 5,000 samples with time.perf_counter(), "
                         "percentiles computed directly from the raw latency array."),
        },
        "default_metrics_t_0_50": {k: v for k, v in metrics_default.items() if k != "y_pred"},
        "calibrated_metrics": {k: v for k, v in metrics_calibrated.items() if k != "y_pred"},
        "latency_benchmark_ms": {k: v for k, v in latency_stats.items() if k != "latencies_array"},
        "hardware_profile": {
            "peak_ram_mb": profile_data["peak_ram_mb"],
            "avg_ram_mb": profile_data["avg_ram_mb"],
            "peak_cpu_pct": profile_data["peak_cpu_pct"],
            "avg_cpu_pct": profile_data["avg_cpu_pct"],
        },
    }

    json_path = out("lr_without_flow_duration_crossday_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)

    print(f"[INFO] JSON results successfully saved to: {json_path}")

    print("\n" + "=" * 84)
    print(f" EVALUATION SUMMARY (LOGISTIC REGRESSION - {ABLATION_NAME.upper()})")
    print("=" * 84)
    print(f"{'Metric':<22}{'Default (T=0.50)':>24}{f'Calibrated (T={calibrated_th:.4f})':>28}")
    print("-" * 84)
    for m in ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]:
        print(f"{m.upper():<22}{metrics_default[m]:>24.4f}{metrics_calibrated[m]:>28.4f}")
    print("-" * 84)
    print(" PURE REAL-TIME LATENCY (Batch Size = 1, N = 5,000):")
    print(f"   Mean: {latency_stats['mean_latency_ms']:.4f} ms | P50: {latency_stats['p50_latency_ms']:.4f} ms | "
          f"P90: {latency_stats['p90_latency_ms']:.4f} ms | P95: {latency_stats['p95_latency_ms']:.4f} ms | "
          f"P99: {latency_stats['p99_latency_ms']:.4f} ms")
    print(f"   Throughput: {latency_stats['throughput_fps']:.2f} Samples/Sec")
    print(" HARDWARE PROFILING (Phase 1, Full Dataset):")
    print(f"   Peak RAM: {profile_data['peak_ram_mb']:.2f} MB | Avg RAM: {profile_data['avg_ram_mb']:.2f} MB | "
          f"Peak CPU: {profile_data['peak_cpu_pct']:.2f}% | Avg CPU: {profile_data['avg_cpu_pct']:.2f}%")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
