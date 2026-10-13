#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Day Real-Time Evaluation Pipeline for Logistic Regression (Ablation: Without Packet Count)
Evaluation Protocol:
  1. Classification Metrics (Accuracy, Precision, Recall, F1, F2, MCC, ROC-AUC) -> Full Test Dataset (100%)
  2. Single-Sample Inference Latency (Mean, P50, P90, P95, P99) -> 5,000 Samples with Batch Size = 1
  3. Hardware Profiling -> RSS Memory Footprint (MB) & CPU Load (%)
Visualization:
  - Font: Times New Roman, Size: 10pt
  - DPI: 300, Standard Units, Legends & Captions
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
from pathlib import Path

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
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. Thesis Plotting Standard Configuration (10pt, Times New Roman, 300 DPI)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

ABLATION_NAME = "without_packet_count"
REMOVED_FEATURE = "Packet_Count"

# Paths & File Definitions
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        os.path.join("results", "ablation_crossday", "logistic_regression", ABLATION_NAME),
    )
)

FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

MODEL_FILENAME = "lr_model_without_packet_count.joblib"
SCALER_FILENAME = "scaler_lr_without_packet_count.joblib"
METADATA_FILENAME = "lr_metadata_without_packet_count.json"
FALLBACK_METADATA_FILENAME = "model_metadata.json"

CHUNK_SIZE = 25000
BENCHMARK_LATENCY_SAMPLES = 5000

# Canonical CSV Columns to extract
CANONICAL_RAW_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

# 3 Features for Ablation (Without Packet_Count)
FEATURE_COLUMNS = [
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

ALL_FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]


# ============================================================
# 2. File Resolution & Stream Helpers
# ============================================================
def out(filename: str) -> str:
    """تضمین وجود دایرکتوری خروجی و بازگرداندن مسیر کامل فایل"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(RESULTS_DIR / filename)


def find_file(filename: str) -> str:
    """جستجوی دقیق فایل در مسیرهای تعریف‌شده"""
    candidate_paths = [
        str(RESULTS_DIR / filename),
        os.path.join("/mnt/data", filename),
        os.path.join("/app", filename),
        os.path.join(".", filename),
        filename,
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] Required file not found: {filename}")


def resolve_csv_columns(file_path):
    header_df = pd.read_csv(file_path, nrows=0)
    actual_columns = list(header_df.columns)
    normalized_to_actual = {str(col).strip(): col for col in actual_columns}

    missing_cols = [col for col in CANONICAL_RAW_COLUMNS if col not in normalized_to_actual]
    if missing_cols:
        raise ValueError(f"Missing columns in {file_path}: {missing_cols}")

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

        prepared_all = pd.DataFrame({
            "Packet_Count": (fwd_packets + bwd_packets).astype(np.float32),
            "Packet_Size": avg_packet_size.astype(np.float32),
            "Flow_Rate": flow_rate.astype(np.float32),
            "Flow_Duration": flow_duration.astype(np.float32),
            "Label": labels,
        })

        prepared_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        prepared_all.dropna(subset=ALL_FEATURE_COLUMNS + ["Label"], inplace=True)

        if len(prepared_all) == 0:
            del raw_chunk, prepared_all
            gc.collect()
            continue

        X_chunk = prepared_all[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        y_chunk = (prepared_all["Label"].to_numpy() != "BENIGN").astype(np.uint8)

        del raw_chunk, prepared_all
        gc.collect()

        yield X_chunk, y_chunk


def load_calibrated_threshold():
    """بارگذاری مطمئن آستانه بهینه از متادیتا"""
    candidate_files = [METADATA_FILENAME, FALLBACK_METADATA_FILENAME, "metadata.json"]
    for fname in candidate_files:
        try:
            meta_path = find_file(fname)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            for key in ["best_threshold", "optimal_threshold", "calibrated_threshold", "threshold"]:
                if key in meta:
                    th = float(meta[key])
                    print(f">> [METADATA] Successfully loaded T = {th:.4f} (Key: '{key}') from {meta_path}")
                    return th

            if "threshold_calibration" in meta and "optimal_threshold" in meta["threshold_calibration"]:
                th = float(meta["threshold_calibration"]["optimal_threshold"])
                print(f">> [METADATA] Successfully loaded T = {th:.4f} from threshold_calibration in {meta_path}")
                return th
        except Exception:
            continue

    print(">> [WARNING] Metadata not found. Defaulting calibrated threshold to 0.50")
    return 0.50


# ============================================================
# 3. Evaluation Functions
# ============================================================
def evaluate_predictions(y_true, y_prob, threshold, roc_auc_val=None):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_val) if roc_auc_val is not None else float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


def benchmark_realtime_latency(model, scaler, test_file, n_samples=BENCHMARK_LATENCY_SAMPLES):
    """
    سنجش دقیق تأخیر استنتاج خالص به‌صورت تک‌نمونه‌ای (Batch Size = 1)
    """
    print(f">> [BENCHMARK] Measuring Pure Single-Sample Latency (Batch Size = 1, N = {n_samples:,})...")

    samples_list = []
    for X_chunk, _ in iter_prepared_chunks(test_file):
        samples_list.append(X_chunk)
        if sum(len(c) for c in samples_list) >= n_samples:
            break

    X_bench = np.vstack(samples_list)[:n_samples]
    X_bench_scaled = scaler.transform(X_bench)

    # فاز گرم کردن مدل (Warm-up Phase)
    for i in range(min(100, len(X_bench_scaled))):
        _ = model.predict_proba(X_bench_scaled[i:i+1])

    latencies_ms = np.empty(len(X_bench_scaled), dtype=np.float64)

    # تایمر خالص استنتاج
    for i in range(len(X_bench_scaled)):
        x_single = X_bench_scaled[i:i+1]

        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()

        latencies_ms[i] = (t1 - t0) * 1000.0

    return {
        "n_samples": int(len(latencies_ms)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_fps": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0,
        "latencies_array": latencies_ms,
    }


def evaluate_full_stream(model, scaler, test_file):
    """
    پردازش ۱۰۰٪ داده‌های تست برای ارزیابی معیارهای کلی و سنجش مصرف حافظه و CPU
    """
    print(">> [STREAMING EVALUATION] Processing 100% of Wednesday Test Dataset...")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    prob_parts = []
    mem_trace = []
    cpu_trace = []

    for X_chunk, y_chunk in iter_prepared_chunks(test_file):
        mem_before = process.memory_info().rss / (1024 * 1024)

        X_chunk_scaled = scaler.transform(X_chunk)
        probs = model.predict_proba(X_chunk_scaled)[:, 1].astype(np.float32)

        mem_after = process.memory_info().rss / (1024 * 1024)

        y_parts.append(y_chunk)
        prob_parts.append(probs)
        mem_trace.append(max(mem_before, mem_after))
        cpu_trace.append(process.cpu_percent(interval=None))

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities_test = np.concatenate(prob_parts).astype(np.float32)

    profile_data = {
        "memory_trace_mb": np.array(mem_trace, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace, dtype=np.float32),
        "peak_ram_mb": float(np.max(mem_trace)),
        "avg_ram_mb": float(np.mean(mem_trace)),
        "peak_cpu_pct": float(np.max(cpu_trace)),
        "avg_cpu_pct": float(np.mean(cpu_trace)),
    }

    return y_test, probabilities_test, profile_data


# ============================================================
# 4. Standard Thesis Visualizations (Times New Roman, 10pt, 300 DPI)
# ============================================================
def plot_comprehensive_metrics_bar(metrics, threshold_tag, output_path, is_calibrated=False):
    """
    رسم نمودار ستونی ۷ معیار اصلی طبقه‌بندی
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    values = [metrics[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    color = "#59A14F" if is_calibrated else "#2b5c8f"
    legend_label = f"Threshold T = {threshold_tag}"

    bars = ax.bar(metric_names, values, color=color, edgecolor="black", linewidth=0.8, width=0.52, label=legend_label)

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=10)
    ax.set_xlabel("Evaluation Metric", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(val + 0.02, 1.06),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    caption = f"Figure: Classification performance metrics of LR (Without Packet Count) on Wednesday test data (T = {threshold_tag})."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics_comparison_grouped(metrics_default, metrics_calibrated, threshold_calibrated, output_path):
    """
    نمودار مقایسه‌ای هم‌زمان ۷ معیار (آستانه پیش‌فرض ۰.۵۰ در برابر آستانه کالیبره‌شده)
    """
    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "F2", "MCC", "ROC-AUC"]

    vals_def = [metrics_default[k] for k in metric_keys]
    vals_cal = [metrics_calibrated[k] for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    bars1 = ax.bar(x - width/2, vals_def, width, label="Default (T = 0.50)", color="#4C78A8", edgecolor="black", linewidth=0.7)
    bars2 = ax.bar(x + width/2, vals_cal, width, label=f"Calibrated (T = {threshold_calibrated:.4f})", color="#59A14F", edgecolor="black", linewidth=0.7)

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=10)
    ax.set_xlabel("Classification Metric", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(-0.05, 1.18)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    caption = "Figure: Comparative evaluation of classification metrics before and after decision threshold calibration."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latency_percentiles(latency_stats, output_path):
    """
    نمودار صدک‌های تأخیر بلادرنگ تک‌نمونه‌ای
    """
    keys = ["mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    labels = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    vals = [latency_stats[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    bars = ax.bar(labels, vals, color="#7A68A6", edgecolor="black", linewidth=0.8, width=0.5, label="Pure Inference (Batch Size = 1)")

    offset = max(vals) * 0.03 if max(vals) > 0 else 0.001
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + offset, f"{v:.4f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Inference Latency (ms/sample)", fontsize=10)
    ax.set_xlabel("Latency Percentiles & Mean", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption = "Figure: Pure single-sample inference latency percentiles for LR (Without Packet Count) over 5,000 samples."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, caption_label, output_path, cmap="Blues"):
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
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)

    caption = f"Figure: Confusion matrix of LR (Without Packet Count) on Wednesday dataset under {caption_label}."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_pr_curves(y_test, probabilities, output_path):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    avg_precision = average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300)

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#D55E00", lw=1.5, label=f"LR (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Chance")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].set_title("(a) Receiver Operating Characteristic", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=9)

    # PR Curve
    axes[1].plot(recall, precision, color="#0072B2", lw=1.5, label=f"LR (AP = {avg_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=10, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower left", fontsize=9)

    caption = "Figure: Discrimination characteristics (ROC and PR curves) of LR (Without Packet Count) under Cross-Day evaluation."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(roc_auc_val), float(avg_precision)


def plot_resource_profiles(profile_data, mem_path, cpu_path):
    mem_trace = profile_data["memory_trace_mb"]
    cpu_trace = profile_data["cpu_trace_pct"]
    chunks = np.arange(1, len(mem_trace) + 1)

    # 1. RAM Plot
    fig_mem, ax_mem = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax_mem.plot(chunks, mem_trace, color="#1f77b4", marker="o", markersize=3, linewidth=1.5, label="Measured RSS Memory")
    ax_mem.axhline(1024, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7, label="1 GB RAM Budget")
    ax_mem.set_ylabel("Resident Set Size (MB)", fontsize=10)
    ax_mem.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=10)
    ax_mem.set_ylim(0, max(float(np.max(mem_trace)) * 1.3, 1100.0))
    ax_mem.legend(loc="upper right", frameon=True, fontsize=10)
    ax_mem.grid(True, linestyle="--", alpha=0.5)

    caption_mem = "Figure: Memory consumption profile (RSS in MB) during continuous streaming inference."
    fig_mem.text(0.5, 0.01, caption_mem, ha="center", fontsize=10, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(mem_path, dpi=300)
    plt.close(fig_mem)

    # 2. CPU Plot
    fig_cpu, ax_cpu = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    ax_cpu.plot(chunks, cpu_trace, color="#ff7f0e", marker="s", markersize=3, linewidth=1.5, label="Process CPU Load (%)")
    avg_cpu = profile_data["avg_cpu_pct"]
    ax_cpu.axhline(avg_cpu, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"Mean CPU Load ({avg_cpu:.1f}%)")
    ax_cpu.set_ylabel("Processor Utilization (%)", fontsize=10)
    ax_cpu.set_xlabel("Evaluation Stream Progress (Chunk Sequence)", fontsize=10)
    ax_cpu.set_ylim(0, 105)
    ax_cpu.legend(loc="upper right", frameon=True, fontsize=10)
    ax_cpu.grid(True, linestyle="--", alpha=0.5)

    caption_cpu = "Figure: Processor load profile over streaming inference chunks across the Wednesday dataset."
    fig_cpu.text(0.5, 0.01, caption_cpu, ha="center", fontsize=10, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(cpu_path, dpi=300)
    plt.close(fig_cpu)


# ============================================================
# 5. Main Execution Pipeline
# ============================================================
def main():
    print("=" * 80)
    print(" LOGISTIC REGRESSION ABLATION: WITHOUT PACKET COUNT (CROSS-DAY REAL-TIME)")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wednesday_file = find_file(WEDNESDAY_FILENAME)
    model_path = find_file(MODEL_FILENAME)
    scaler_path = find_file(SCALER_FILENAME)

    print(f">> Loaded Model : {model_path}")
    print(f">> Loaded Scaler: {scaler_path}")
    print(f">> Test Dataset : {wednesday_file}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # اعتبارسنجی ابعاد ویژگی‌ها
    if hasattr(model, "n_features_in_") and model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(f"Model expects {model.n_features_in_} features, but {len(FEATURE_COLUMNS)} are given.")

    calibrated_th = load_calibrated_threshold()

    # ۱. بنچمارک دقیق تاخیر بلادرنگ (Batch Size = 1)
    latency_stats = benchmark_realtime_latency(model, scaler, wednesday_file, n_samples=BENCHMARK_LATENCY_SAMPLES)

    # ۲. استریم کامل داده‌های روز چهارشنبه (Full Test Dataset)
    y_test, probabilities_test, profile_data = evaluate_full_stream(model, scaler, wednesday_file)

    # ۳. محاسبه تفکیک‌پذیری و متریک‌ها
    roc_auc_val, avg_precision = plot_roc_pr_curves(y_test, probabilities_test, out("lr_without_packet_count_wednesday_roc_pr_curves.png"))

    metrics_default = evaluate_predictions(y_test, probabilities_test, threshold=0.50, roc_auc_val=roc_auc_val)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, threshold=calibrated_th, roc_auc_val=roc_auc_val)

    # ۴. ذخیره نمودارهای استاندارد پایان‌نامه
    print("\n>> Generating standardized thesis figures (Times New Roman, 10pt, 300 DPI)...")
    plot_comprehensive_metrics_bar(metrics_default, "0.50", out("lr_without_packet_count_metrics_default.png"), is_calibrated=False)
    plot_comprehensive_metrics_bar(metrics_calibrated, f"{calibrated_th:.4f}", out("lr_without_packet_count_metrics_calibrated.png"), is_calibrated=True)
    plot_metrics_comparison_grouped(metrics_default, metrics_calibrated, calibrated_th, out("lr_without_packet_count_metrics_comparison.png"))

    plot_confusion_matrix(y_test, metrics_default["y_pred"], "Confusion Matrix (T = 0.50)", "default threshold", out("lr_without_packet_count_cm_default.png"), cmap="Blues")
    plot_confusion_matrix(y_test, metrics_calibrated["y_pred"], f"Confusion Matrix (T = {calibrated_th:.4f})", "calibrated threshold", out("lr_without_packet_count_cm_calibrated.png"), cmap="Greens")

    plot_latency_percentiles(latency_stats, out("lr_without_packet_count_latency_percentiles.png"))
    plot_resource_profiles(profile_data, out("lr_without_packet_count_memory_usage.png"), out("lr_without_packet_count_cpu_usage.png"))

    # ۵. ساختار خروجی JSON
    results_json = {
        "experiment": {
            "model": "Logistic Regression",
            "ablation": ABLATION_NAME,
            "removed_feature": REMOVED_FEATURE,
            "features_used": FEATURE_COLUMNS,
            "test_dataset": WEDNESDAY_FILENAME,
            "protocol": "Full 100% test evaluation for metrics + 5,000 samples (batch_size=1) for latency",
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

    json_path = out("lr_without_packet_count_wednesday_crossday_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)

    print(f"\n>> JSON results successfully saved to: {json_path}")

    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY (LOGISTIC REGRESSION - WITHOUT PACKET COUNT)")
    print("=" * 80)
    print(f"{'Metric':<20}{'Default (T=0.50)':>25}{f'Calibrated (T={calibrated_th:.4f})':>28}")
    print("-" * 80)
    for m in ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]:
        print(f"{m.upper():<20}{metrics_default[m]:>25.4f}{metrics_calibrated[m]:>28.4f}")
    print("-" * 80)
    print(" PURE REAL-TIME LATENCY (Batch Size = 1, N = 5,000):")
    print(f"   Mean: {latency_stats['mean_latency_ms']:.4f} ms | P50: {latency_stats['p50_latency_ms']:.4f} ms | P90: {latency_stats['p90_latency_ms']:.4f} ms | P95: {latency_stats['p95_latency_ms']:.4f} ms | P99: {latency_stats['p99_latency_ms']:.4f} ms")
    print(" HARDWARE PROFILING:")
    print(f"   Peak RAM: {profile_data['peak_ram_mb']:.2f} MB | Avg RAM: {profile_data['avg_ram_mb']:.2f} MB")
    print(f"   Peak CPU: {profile_data['peak_cpu_pct']:.2f}% | Avg CPU: {profile_data['avg_cpu_pct']:.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
