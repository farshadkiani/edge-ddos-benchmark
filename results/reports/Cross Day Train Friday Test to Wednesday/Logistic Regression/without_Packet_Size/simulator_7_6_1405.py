#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Day Real-Time Evaluation Pipeline for Logistic Regression (Ablation: Without Packet Size)
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
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. Thesis Plotting Standard Configuration (10pt, Times New Roman, 300 DPI)
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ABLATION_NAME = "without_packet_size"

# Paths & File Definitions
RESULTS_DIR = Path(
    os.environ.get(
        "RESULTS_DIR",
        os.path.join(
            "results",
            "ablation_crossday",
            "logistic_regression",
            ABLATION_NAME,
        ),
    )
)

WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

# فایل‌های آزمایش بدون Packet_Size
MODEL_FILENAME = "lr_model_without_packet_size.joblib"
SCALER_FILENAME = "scaler_lr_without_packet_size.joblib"
METADATA_FILENAME = "lr_metadata_without_packet_size.json"
FALLBACK_METADATA_FILENAME = "model_metadata.json"

CHUNK_SIZE = 25000
BENCHMARK_LATENCY_SAMPLES = 5000

# ستون‌های خام مورد نیاز از مجموعه داده CICIDS2017
CANONICAL_RAW_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

# ۳ ویژگی مهندسی‌شده ورودی مدل رگرسیون لجستیک (حذف بعد اندازه بسته)
FEATURE_COLUMNS = [
    "Packet_Count",
    "Flow_Rate",
    "Flow_Duration",
]


# ============================================================
# 2. File Resolution & Stream Helpers
# ============================================================
def out(filename):
    """تضمین وجود دایرکتوری خروجی و بازگرداندن مسیر کامل فایل"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return str(Path(RESULTS_DIR) / filename)

def find_file(filename):
    """جستجوی دقیق و بازگشتی فایل‌ها در مسیرهای مختلف داکر و هاست"""
    base_dirs = [
        str(RESULTS_DIR),
        "/app/results",
        "/app",
        ".",
        "/mnt/data"
    ]
    
    for b in base_dirs:
        p = os.path.join(b, filename)
        if os.path.exists(p):
            return p
        nested = os.path.join(b, "ablation_crossday", "logistic_regression", "without_packet_size", filename)
        if os.path.exists(nested):
            return nested

    for b in base_dirs:
        if os.path.exists(b):
            for root, _, files in os.walk(b):
                if filename in files:
                    return os.path.join(root, filename)

    raise FileNotFoundError(f"[ERROR] Required file not found: {filename}")

def load_calibrated_threshold():
    """بارگذاری مطمئن و انعطاف‌پذیر آستانه کالیبره شده"""
    candidate_files = [METADATA_FILENAME, FALLBACK_METADATA_FILENAME, "metadata.json"]
    for fname in candidate_files:
        try:
            meta_path = find_file(fname)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # بررسی کلیدهای مختلف
            for key in ["best_threshold", "optimal_threshold", "calibrated_threshold", "threshold"]:
                if key in meta:
                    th = float(meta[key])
                    print(f">> [METADATA] Successfully loaded T = {th:.4f} (Key: '{key}') from {meta_path}")
                    return th
            
            # در صورتی که ساختار تو در تو باشد
            if "metrics" in meta and isinstance(meta["metrics"], dict):
                for key in ["best_threshold", "optimal_threshold", "calibrated_threshold", "threshold"]:
                    if key in meta["metrics"]:
                        th = float(meta["metrics"][key])
                        print(f">> [METADATA] Successfully loaded T = {th:.4f} from nested metrics in {meta_path}")
                        return th
        except Exception:
            continue
            
    print(">> [WARNING] Metadata not found or failed to parse. Defaulting calibrated threshold to 0.50")
    return 0.50

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
        flow_rate = pd.to_numeric(raw_chunk["Flow Packets/s"], errors="coerce")
        flow_duration = pd.to_numeric(raw_chunk["Flow Duration"], errors="coerce")
        labels = raw_chunk["Label"].astype(str).str.strip().str.upper()

        prepared = pd.DataFrame({
            "Packet_Count": (fwd_packets + bwd_packets).astype(np.float32),
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
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_val) if roc_auc_val is not None else float(roc_auc_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


def benchmark_realtime_latency(model, scaler, test_file, n_samples=BENCHMARK_LATENCY_SAMPLES):
    print(f">> [BENCHMARK] Measuring Single-Sample Latency (Batch Size = 1, N = {n_samples:,})...")

    samples_list = []
    for X_chunk, _ in iter_prepared_chunks(test_file):
        samples_list.append(X_chunk)
        if sum(len(c) for c in samples_list) >= n_samples:
            break

    X_bench = np.vstack(samples_list)[:n_samples]
    X_bench_scaled = scaler.transform(X_bench)

    # فاز گرم کردن مدل (Warm-up phase)
    for i in range(min(100, len(X_bench_scaled))):
        _ = model.predict_proba(X_bench_scaled[i : i + 1])

    latencies_ms = np.empty(len(X_bench_scaled), dtype=np.float64)

    for i in range(len(X_bench_scaled)):
        x_single = X_bench_scaled[i : i + 1]

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
    }


def evaluate_full_stream(model, scaler, test_file):
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
def plot_metrics_bar(metrics, threshold_tag, output_path, is_calibrated=False):
    metric_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    values = [metrics[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=300)
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

    caption = f"Figure: Classification performance metrics on Wednesday dataset (T = {threshold_tag})."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_latency_percentiles(latency_stats, output_path):
    keys = ["p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    labels = ["P50 (Median)", "P90", "P95", "P99 (Tail)"]
    vals = [latency_stats[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=300)
    bars = ax.bar(labels, vals, color="#7A68A6", edgecolor="black", linewidth=0.8, width=0.5, label="Single-Sample Latency (Batch Size = 1)")

    offset = max(vals) * 0.03 if max(vals) > 0 else 0.001
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2.0, v + offset, f"{v:.4f} ms", ha="center", fontsize=10)

    ax.set_ylabel("Pure Inference Latency (ms/sample)", fontsize=10)
    ax.set_xlabel("Tail Latency Percentiles", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption = "Figure: Pure inference latency percentiles of LR (Without Packet Size) over 5,000 samples."
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

    caption = f"Figure: Confusion matrix of LR (Without Packet Size) on Wednesday dataset under {caption_label}."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


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
    print("=" * 78)
    print(" LR ABLATION EVALUATION (WITHOUT PACKET SIZE: 100% TEST DATA + REAL-TIME BENCH)")
    print("=" * 78)

    # ساخت دایرکتوری نتایج
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wednesday_file = find_file(WEDNESDAY_FILENAME)
    model_path = find_file(MODEL_FILENAME)
    scaler_path = find_file(SCALER_FILENAME)

    print(f">> Loaded Model : {model_path}")
    print(f">> Loaded Scaler: {scaler_path}")
    print(f">> Test Dataset : {wednesday_file}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # خواندن آستانه کالیبره شده با تابع ارتقایافته
    calibrated_th = load_calibrated_threshold()

    # ۱. بنچمارک تاخیر بلادرنگ تک‌نمونه‌ای
    latency_stats = benchmark_realtime_latency(model, scaler, wednesday_file, n_samples=BENCHMARK_LATENCY_SAMPLES)

    # ۲. استریم کامل داده‌های روز چهارشنبه
    y_test, probabilities_test, profile_data = evaluate_full_stream(model, scaler, wednesday_file)

    roc_auc_val = float(roc_auc_score(y_test, probabilities_test))

    # ۳. محاسبه متریک‌ها
    metrics_default = evaluate_predictions(y_test, probabilities_test, threshold=0.50, roc_auc_val=roc_auc_val)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, threshold=calibrated_th, roc_auc_val=roc_auc_val)

    # ۴. ذخیره نمودارها
    print("\n>> Generating standardized thesis figures (Times New Roman, 10pt, 300 DPI)...")
    plot_metrics_bar(metrics_default, "0.50", out("lr_ablation_metrics_default_without_packet_size.png"), is_calibrated=False)
    plot_metrics_bar(metrics_calibrated, f"{calibrated_th:.4f}", out("lr_ablation_metrics_calibrated_without_packet_size.png"), is_calibrated=True)
    plot_confusion_matrix(y_test, metrics_default["y_pred"], "Confusion Matrix (T = 0.50)", "default threshold", out("lr_ablation_cm_default_without_packet_size.png"), cmap="Blues")
    plot_confusion_matrix(y_test, metrics_calibrated["y_pred"], f"Confusion Matrix (T = {calibrated_th:.4f})", "calibrated threshold", out("lr_ablation_cm_calibrated_without_packet_size.png"), cmap="Greens")
    plot_latency_percentiles(latency_stats, out("lr_ablation_latency_percentiles_without_packet_size.png"))
    plot_resource_profiles(profile_data, out("lr_ablation_memory_usage_without_packet_size.png"), out("lr_ablation_cpu_usage_without_packet_size.png"))

    # ۵. ذخیره فایل نتایج ساختاریافته (JSON)
    results_json = {
        "model": "Logistic Regression",
        "ablation": ABLATION_NAME,
        "default_metrics_t_0_50": {k: v for k, v in metrics_default.items() if k != "y_pred"},
        "calibrated_metrics": {k: v for k, v in metrics_calibrated.items() if k != "y_pred"},
        "latency_benchmark": latency_stats,
        "hardware_profile": {
            "peak_ram_mb": profile_data["peak_ram_mb"],
            "avg_ram_mb": profile_data["avg_ram_mb"],
            "peak_cpu_pct": profile_data["peak_cpu_pct"],
            "avg_cpu_pct": profile_data["avg_cpu_pct"],
        }
    }
    with open(out("lr_without_packet_size_wednesday_crossday_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)

    print("\n" + "=" * 78)
    print(" EVALUATION SUMMARY (LOGISTIC REGRESSION - WITHOUT PACKET SIZE)")
    print("=" * 78)
    print(f"{'Metric':<20}{'Default (T=0.50)':>25}{f'Calibrated (T={calibrated_th:.4f})':>28}")
    print("-" * 78)
    for m in ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]:
        print(f"{m.upper():<20}{metrics_default[m]:>25.4f}{metrics_calibrated[m]:>28.4f}")
    print("-" * 78)
    print(" PURE REAL-TIME LATENCY (Batch Size = 1, N = 5,000):")
    print(f"   Mean: {latency_stats['mean_latency_ms']:.4f} ms | P50: {latency_stats['p50_latency_ms']:.4f} ms | P95: {latency_stats['p95_latency_ms']:.4f} ms | P99: {latency_stats['p99_latency_ms']:.4f} ms")
    print(" HARDWARE PROFILING:")
    print(f"   Peak RAM: {profile_data['peak_ram_mb']:.2f} MB | Avg RAM: {profile_data['avg_ram_mb']:.2f} MB")
    print(f"   Peak CPU: {profile_data['peak_cpu_pct']:.2f}% | Avg CPU: {profile_data['avg_cpu_pct']:.2f}%")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
