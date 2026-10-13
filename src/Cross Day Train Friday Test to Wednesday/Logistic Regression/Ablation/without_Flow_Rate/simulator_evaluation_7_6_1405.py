#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Day Real-Time Evaluation Pipeline for Logistic Regression (Ablation: without_flow_rate)
Dual-Phase Protocol:
  - Phase 1: 100% Data Vectorized Evaluation -> Classification Metrics (Acc, Prec, Rec, F1, F2, MCC, ROC-AUC)
  - Phase 2: 5,000 Samples with Batch Size = 1 -> Pure Single-Sample Inference Latency (Mean, P50, P90, P95, P99)
Visualization:
  - Typography: Times New Roman, 10pt
  - Quality: 300 DPI, Uniform Legends, Labels, and Captions
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
CHUNK_SIZE = 25000

FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

ABLATION_NAME = "without_flow_rate"
REMOVED_FEATURE = "Flow_Rate"

# 3 Features for this ablation study (Flow_Rate excluded)
FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Duration"]
ALL_FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Duration", "Flow_Rate"]

CANONICAL_RAW_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        "results/ablation_crossday/logistic_regression/without_flow_rate",
    )
)

MODEL_FILENAME = "lr_model_without_flow_rate.joblib"
SCALER_FILENAME = "scaler_lr_without_flow_rate.joblib"
METADATA_FILENAME = "lr_metadata_without_flow_rate.json"


# ============================================================
# 2. Utility & IO Helpers
# ============================================================

def out(filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / filename


def find_file(filename: str) -> Path:
    candidates = [
        Path(filename),
        RESULTS_DIR / filename,
        Path("/mnt/data") / filename,
        Path("/app") / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"[ERROR] Required file not found in search paths: {filename}")


def resolve_csv_columns(df_columns):
    stripped = {str(col).strip(): col for col in df_columns}
    resolved = {}
    missing = []

    for canonical in CANONICAL_RAW_COLUMNS:
        if canonical in stripped:
            resolved[canonical] = stripped[canonical]
        else:
            missing.append(canonical)

    if missing:
        raise ValueError(f"[ERROR] CSV missing canonical columns: {missing}")

    return resolved


def iter_prepared_chunks(csv_path, chunk_size=CHUNK_SIZE):
    for raw_chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        resolved = resolve_csv_columns(raw_chunk.columns)
        chunk = raw_chunk.rename(columns={v: k for k, v in resolved.items()})

        for col in CANONICAL_RAW_COLUMNS[:-1]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=CANONICAL_RAW_COLUMNS[:-1]).copy()

        packet_count = (chunk["Total Fwd Packets"] + chunk["Total Backward Packets"]).astype(np.float32)
        packet_size = chunk["Average Packet Size"].astype(np.float32)
        flow_duration = chunk["Flow Duration"].astype(np.float32)

        X = np.column_stack([packet_count, packet_size, flow_duration])
        y = (chunk["Label"].astype(str).str.strip().str.upper().ne("BENIGN")).astype(np.uint8).values

        if len(y) == 0:
            continue
        yield X, y


def load_calibrated_threshold():
    candidate_files = [METADATA_FILENAME, "model_metadata.json", "metadata.json"]
    for fname in candidate_files:
        try:
            meta_path = find_file(fname)
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

    print("[WARNING] Metadata not found. Defaulting calibrated threshold to 0.50")
    return 0.50


# ============================================================
# 3. Dual-Phase Evaluation Logic
# ============================================================

def evaluate_full_stream(model, scaler, test_file):
    """فاز ۱: اجرای دسته‌ای و سریع روی ۱۰۰٪ داده‌ها برای معیارهای آماری و کارایی پردازنده"""
    print("[PHASE 1] Vectorized evaluation on 100% of Wednesday dataset...")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    prob_parts = []
    mem_trace = []
    cpu_trace = []

    for X_chunk, y_chunk in iter_prepared_chunks(test_file):
        mem_before = process.memory_info().rss / (1024.0 * 1024.0)

        X_scaled = scaler.transform(X_chunk).astype(np.float32)
        probs = model.predict_proba(X_scaled)[:, 1].astype(np.float32)

        mem_after = process.memory_info().rss / (1024.0 * 1024.0)

        y_parts.append(y_chunk)
        prob_parts.append(probs)
        mem_trace.append(max(mem_before, mem_after))
        cpu_trace.append(process.cpu_percent(interval=None))

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities = np.concatenate(prob_parts).astype(np.float32)

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
    """فاز ۲: سنجش تأخیر بلادرنگ تک‌نمونه‌ای (Batch Size = 1) روی ۵۰۰۰ نمونه با زمان‌گیری خالص"""
    print(f"[PHASE 2] Pure inference latency benchmark (Batch Size = 1, N = {n_samples:,})...")

    samples_list = []
    for X_chunk, _ in iter_prepared_chunks(test_file):
        samples_list.append(X_chunk)
        if sum(len(c) for c in samples_list) >= n_samples:
            break

    X_bench = np.vstack(samples_list)[:n_samples]
    X_bench_scaled = scaler.transform(X_bench).astype(np.float32)

    # Warm-up phase (100 samples)
    for i in range(min(100, len(X_bench_scaled))):
        _ = model.predict_proba(X_bench_scaled[i : i + 1])

    latencies_ms = np.empty(len(X_bench_scaled), dtype=np.float64)

    # Pure Single-Sample Inference Timer
    for i in range(len(X_bench_scaled)):
        x_single = X_bench_scaled[i : i + 1]

        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()

        latencies_ms[i] = (t1 - t0) * 1000.0

    return {
        "n_samples": int(len(latencies_ms)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_fps": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0,
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
# 4. Standard Thesis Figures (Times New Roman, 10pt, 300 DPI)
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

    caption = f"Figure: Classification metrics of LR (Without Flow Rate) on Wednesday dataset (T = {threshold_tag})."
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
    bars = ax.bar(labels, vals, color="#8172B2", edgecolor="black", linewidth=0.8, width=0.5, label="Pure Inference (Batch Size = 1)")

    offset = max(vals) * 0.03 if max(vals) > 0 else 0.001
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + offset, f"{v:.4f} ms", ha="center", va="bottom", fontsize=FIG_FONT_SIZE)

    ax.set_ylabel("Inference Latency (ms/sample)", fontsize=FIG_FONT_SIZE)
    ax.set_xlabel("Latency Percentiles & Mean", fontsize=FIG_FONT_SIZE)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.legend(loc="upper left", frameon=True, fontsize=FIG_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption = "Figure: Pure single-sample inference latency percentiles for LR (Without Flow Rate) over 5,000 samples."
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

    caption = f"Figure: Confusion matrix of LR (Without Flow Rate) on Wednesday dataset under {caption_label}."
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

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#4C72B0", lw=1.5, label=f"LR (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Chance")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=FIG_FONT_SIZE)
    axes[0].set_ylabel("True Positive Rate", fontsize=FIG_FONT_SIZE)
    axes[0].set_title("(a) Receiver Operating Characteristic", fontsize=FIG_FONT_SIZE, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    # PR Curve
    axes[1].plot(recall_curve, precision_curve, color="#C44E52", lw=1.5, label=f"LR (AP = {avg_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=FIG_FONT_SIZE)
    axes[1].set_ylabel("Precision", fontsize=FIG_FONT_SIZE)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=FIG_FONT_SIZE, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=9)

    caption = "Figure: Discrimination performance (ROC and PR curves) of LR (Without Flow Rate) under Cross-Day evaluation."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=FIG_FONT_SIZE, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=FIG_DPI)
    plt.close()

    return float(roc_auc_val), float(avg_precision)


def plot_resource_profiles(profile_data, mem_path, cpu_path):
    mem_trace = profile_data["memory_trace_mb"]
    cpu_trace = profile_data["cpu_trace_pct"]
    chunks = np.arange(1, len(mem_trace) + 1)

    # Memory Profile
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

    # CPU Profile
    fig_cpu, ax_cpu = plt.subplots(figsize=(6.8, 3.8), dpi=FIG_DPI)
    ax_cpu.plot(chunks, cpu_trace, color="#E1974C", marker="s", markersize=3, linewidth=1.5, label="Process CPU Load (%)")
    avg_cpu = profile_data["avg_cpu_pct"]
    ax_cpu.axhline(avg_cpu, color="#55A868", linestyle="--", linewidth=1.2, label=f"Mean CPU Load ({avg_cpu:.1f}%)")
    ax_cpu.set_ylabel("Processor Utilization (%)", fontsize=FIG_FONT_SIZE)
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
    print(" LOGISTIC REGRESSION ABLATION: WITHOUT FLOW RATE (CROSS-DAY REAL-TIME PIPELINE)")
    print("=" * 84)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wednesday_file = find_file(WEDNESDAY_FILENAME)
    model_path = find_file(MODEL_FILENAME)
    scaler_path = find_file(SCALER_FILENAME)

    print(f"[INFO] Loaded Model : {model_path}")
    print(f"[INFO] Loaded Scaler: {scaler_path}")
    print(f"[INFO] Test Dataset : {wednesday_file}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(f"Model expects {model.n_features_in_} features, but {len(FEATURE_COLUMNS)} are given.")

    calibrated_th = load_calibrated_threshold()

    # ۱. بنچمارک دقیق تاخیر بلادرنگ (Batch Size = 1) روی نمونه‌های تکی
    latency_stats = benchmark_realtime_latency(model, scaler, wednesday_file, n_samples=LATENCY_BENCHMARK_SAMPLES)

    # ۲. استریم کامل داده‌های روز چهارشنبه (Full 100% Test Dataset) برای معیارهای کیفی و سخت‌افزاری
    y_test, probabilities_test, profile_data = evaluate_full_stream(model, scaler, wednesday_file)

    # ۳. ارزیابی مشخصه‌ها و معیارهای ۷‌گانه
    roc_auc_val, avg_precision = plot_roc_pr_curves(y_test, probabilities_test, out("lr_without_flow_rate_roc_pr_curves.png"))

    metrics_default = evaluate_predictions(y_test, probabilities_test, threshold=0.50, roc_auc_val=roc_auc_val)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, threshold=calibrated_th, roc_auc_val=roc_auc_val)

    # ۴. ترسیم و ذخیره‌سازی نمودارهای پایان‌نامه
    print("\n[INFO] Saving standardized thesis figures (Times New Roman, 10pt, 300 DPI)...")
    plot_comprehensive_metrics_bar(metrics_default, "0.50", out("lr_without_flow_rate_metrics_default.png"), is_calibrated=False)
    plot_comprehensive_metrics_bar(metrics_calibrated, f"{calibrated_th:.4f}", out("lr_without_flow_rate_metrics_calibrated.png"), is_calibrated=True)
    plot_metrics_comparison_grouped(metrics_default, metrics_calibrated, calibrated_th, out("lr_without_flow_rate_metrics_comparison.png"))

    plot_confusion_matrix(y_test, metrics_default["y_pred"], "Confusion Matrix (T = 0.50)", "default threshold", out("lr_without_flow_rate_cm_default.png"), cmap="Blues")
    plot_confusion_matrix(y_test, metrics_calibrated["y_pred"], f"Confusion Matrix (T = {calibrated_th:.4f})", "calibrated threshold", out("lr_without_flow_rate_cm_calibrated.png"), cmap="Greens")

    plot_latency_percentiles(latency_stats, out("lr_without_flow_rate_latency_percentiles.png"))
    plot_resource_profiles(profile_data, out("lr_without_flow_rate_memory_usage.png"), out("lr_without_flow_rate_cpu_usage.png"))

    # ۵. تولید ساختار گزارش JSON
    results_json = {
        "experiment": {
            "model": "Logistic Regression",
            "study": "Cross-Day Ablation (without Flow_Rate)",
            "removed_feature": REMOVED_FEATURE,
            "ablation_name": ABLATION_NAME,
            "features_used": FEATURE_COLUMNS,
            "test_dataset": WEDNESDAY_FILENAME,
            "protocol": "Dual-Phase: Full 100% test evaluation for metrics + 5,000 samples (batch_size=1) for latency",
        },
        "default_metrics_t_0_50": {k: v for k, v in metrics_default.items() if k != "y_pred"},
        "calibrated_metrics": {k: v for k, v in metrics_calibrated.items() if k != "y_pred"},
        "latency_benchmark_ms": {k: v for k, v in latency_stats.items() if k != "latencies_array"},
        "hardware_profile": {
            "peak_ram_mb": profile_data["peak_ram_mb"],
            "avg_ram_mb": profile_data["avg_ram_mb"],
            "peak_cpu_pct": profile_data["peak_cpu_pct"],
            "avg_cpu_pct": profile_data["avg_cpu_pct"],
        }
    }

    json_path = out("lr_without_flow_rate_crossday_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)

    print(f"[INFO] JSON results successfully saved to: {json_path}")

    print("\n" + "=" * 84)
    print(" EVALUATION SUMMARY (LOGISTIC REGRESSION - WITHOUT FLOW RATE)")
    print("=" * 84)
    print(f"{'Metric':<22}{'Default (T=0.50)':>24}{f'Calibrated (T={calibrated_th:.4f})':>28}")
    print("-" * 84)
    for m in ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]:
        print(f"{m.upper():<22}{metrics_default[m]:>24.4f}{metrics_calibrated[m]:>28.4f}")
    print("-" * 84)
    print(" PURE REAL-TIME LATENCY (Batch Size = 1, N = 5,000):")
    print(f"   Mean: {latency_stats['mean_latency_ms']:.4f} ms | P50: {latency_stats['p50_latency_ms']:.4f} ms | P90: {latency_stats['p90_latency_ms']:.4f} ms | P95: {latency_stats['p95_latency_ms']:.4f} ms | P99: {latency_stats['p99_latency_ms']:.4f} ms")
    print(f"   Throughput: {latency_stats['throughput_fps']:.2f} Samples/Sec")
    print(" HARDWARE PROFILING:")
    print(f"   Peak RAM: {profile_data['peak_ram_mb']:.2f} MB | Avg RAM: {profile_data['avg_ram_mb']:.2f} MB")
    print(f"   Peak CPU: {profile_data['peak_cpu_pct']:.2f}% | Avg CPU: {profile_data['avg_cpu_pct']:.2f}%")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
