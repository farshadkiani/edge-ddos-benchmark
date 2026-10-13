#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAM-Optimized Fair Cross-Day Evaluation (Without Flow_Duration)
Train domain: Friday DDoS
Test domain : Wednesday

Ablation Setting:
- Excluded: Flow_Duration
- Active Features: Packet_Count, Packet_Size, Flow_Rate

Compliant with JNCA Thesis / Publication Standards:
- Typography: Times New Roman, 10 pt
- Standardized Units, Legends, and Captions
- Resolution: 300 DPI
- Pure Real-Time Latency (Batch Size = 1) with Tail Percentiles (P50, P90, P95, P99)
- Comprehensive Metric Bar Plots (Accuracy, Precision, Recall, F1, F2, MCC, ROC-AUC)
- System Resource Profiles (CPU & Memory Traces with 1 GB Threshold)
- Calibrated Predicted Probability Distributions
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

# ============================================================
# Matplotlib Standardized Publication Styling (10pt Times New Roman)
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 10,
    "figure.autolayout": False,
})

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"

# Pre-trained model without Flow_Duration
MODEL_FILENAME = "random_forest_friday_model_without_flow_duration.pkl"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)
CHUNK_SIZE = 15000

# Active features: Flow_Duration removed, Flow_Rate active
FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
]

CANONICAL_RAW_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Label",
]


# ============================================================
# Helpers
# ============================================================

def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def find_file(filename):
    candidate_paths = [
        os.path.join(RESULTS_DIR, filename),
        os.path.join("/mnt/data", filename),
        os.path.join("/app", filename),
        os.path.join(".", filename),
        filename,
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"[ERROR] Required dataset/model file not found: {filename}")


def resolve_csv_columns(file_path):
    preview = pd.read_csv(file_path, nrows=0)
    actual_columns = list(preview.columns)
    normalized_to_actual = {str(col).strip(): col for col in actual_columns}

    missing_columns = [col for col in CANONICAL_RAW_COLUMNS if col not in normalized_to_actual]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")

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
        packet_size = pd.to_numeric(raw_chunk["Average Packet Size"], errors="coerce")
        flow_rate = pd.to_numeric(raw_chunk["Flow Packets/s"], errors="coerce")
        labels = raw_chunk["Label"].astype(str).str.strip().str.upper()

        prepared = pd.DataFrame({
            "Packet_Count": (fwd_packets + bwd_packets).astype(np.float32),
            "Packet_Size": packet_size.astype(np.float32),
            "Flow_Rate": flow_rate.astype(np.float32),
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
# Evaluation & Calibration
# ============================================================

def evaluate_predictions(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "y_pred": y_pred,
    }


def calibrate_threshold_on_friday_validation(model, friday_file):
    friday_stats = get_dataset_statistics(friday_file)
    total_samples = friday_stats["n_rows_clean"]
    validation_start_index = int(total_samples * 0.80)

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

        del X_chunk, y_chunk, X_validation, y_validation
        gc.collect()

    y_validation_all = np.concatenate(validation_y_parts).astype(np.uint8)
    prob_validation_all = np.concatenate(validation_prob_parts).astype(np.float32)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.50
    best_f2 = -1.0

    for threshold in thresholds:
        y_pred = (prob_validation_all >= threshold).astype(np.uint8)
        current_f2 = fbeta_score(y_validation_all, y_pred, beta=2, zero_division=0)
        if current_f2 > best_f2:
            best_f2 = current_f2
            best_threshold = float(threshold)

    del y_validation_all, prob_validation_all, validation_y_parts, validation_prob_parts
    gc.collect()

    return best_threshold, float(best_f2), friday_stats


# ============================================================
# Latency & Streaming Evaluation
# ============================================================

def benchmark_single_sample_latency(model, wednesday_file, n_benchmark_samples=5000):
    collected_samples = []
    for X_chunk, _ in iter_prepared_chunks(wednesday_file):
        collected_samples.append(X_chunk)
        if sum(len(c) for c in collected_samples) >= n_benchmark_samples:
            break

    X_benchmark = np.vstack(collected_samples)[:n_benchmark_samples]
    total_eval = len(X_benchmark)

    # Warm-up
    for i in range(min(100, total_eval)):
        _ = model.predict_proba(X_benchmark[i : i + 1])

    latencies_ms = np.empty(total_eval, dtype=np.float64)

    for i in range(total_eval):
        x_single = X_benchmark[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()
        latencies_ms[i] = (t1 - t0) * 1000.0

    latency_stats = {
        "n_samples_evaluated": int(total_eval),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0,
    }

    del collected_samples, X_benchmark
    gc.collect()
    return latency_stats, latencies_ms


def evaluate_wednesday_streaming(model, wednesday_file):
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    probability_parts = []
    memory_trace_mb = []
    cpu_trace_pct = []

    total_samples = 0
    total_inference_time_sec = 0.0

    for X_chunk, y_chunk in iter_prepared_chunks(wednesday_file):
        mem_before = get_memory_usage_mb()
        start_time = time.perf_counter()

        chunk_probabilities = model.predict_proba(X_chunk)[:, 1].astype(np.float32)

        elapsed_sec = time.perf_counter() - start_time
        mem_after = get_memory_usage_mb()
        cpu_pct = process.cpu_percent(interval=None)

        y_parts.append(y_chunk.copy())
        probability_parts.append(chunk_probabilities)
        memory_trace_mb.append(max(mem_before, mem_after))
        cpu_trace_pct.append(cpu_pct)

        total_samples += len(y_chunk)
        total_inference_time_sec += elapsed_sec

        del X_chunk, y_chunk, chunk_probabilities
        gc.collect()

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities_test = np.concatenate(probability_parts).astype(np.float32)

    del y_parts, probability_parts
    gc.collect()

    stream_stats = {
        "total_inference_time_sec": float(total_inference_time_sec),
        "n_inference_samples": int(total_samples),
        "peak_memory_mb": float(np.max(memory_trace_mb)) if memory_trace_mb else 0.0,
        "avg_cpu_percent": float(np.mean(cpu_trace_pct)) if cpu_trace_pct else 0.0,
    }

    profile_data = {
        "memory_trace_mb": np.array(memory_trace_mb, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace_pct, dtype=np.float32),
    }

    return y_test, probabilities_test, stream_stats, profile_data


# ============================================================
# Standardized Thesis Plotting Functions (Times New Roman, 10pt)
# ============================================================

def save_comprehensive_metrics_bar(metrics, roc_auc_val, threshold_label, output_path):
    ensure_parent_dir(output_path)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    vals = [
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("f2", 0.0),
        metrics.get("mcc", 0.0),
        roc_auc_val,
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    bars = ax.bar(metric_names, vals, color="#2b5c8f", edgecolor="black", linewidth=0.8, width=0.55, label="Test Metric Score")

    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.02 if val >= 0 else val - 0.05,
            f"{val:.4f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    caption_text = f"Figure: Comprehensive evaluation metrics on Wednesday test dataset ({threshold_label})."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_comparison_figure(metrics_default, metrics_calibrated, threshold_calibrated, roc_auc_val, output_path):
    ensure_parent_dir(output_path)
    metrics_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1", "F2", "MCC", "ROC-AUC"]

    x = np.arange(len(metrics_keys))
    w = 0.35

    vals_default = [metrics_default.get(k, roc_auc_val) if k != "roc_auc" else roc_auc_val for k in metrics_keys]
    vals_cal = [metrics_calibrated.get(k, roc_auc_val) if k != "roc_auc" else roc_auc_val for k in metrics_keys]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=300)
    bars1 = ax.bar(x - w/2, vals_default, width=w, color="#4C78A8", edgecolor="black", linewidth=0.6, label="Threshold = 0.50")
    bars2 = ax.bar(x + w/2, vals_cal, width=w, color="#59A14F", edgecolor="black", linewidth=0.6, label=f"Calibrated T = {threshold_calibrated:.2f}")

    for bar in bars1:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.015, 1.08), f"{v:.3f}", ha="center", fontsize=8)
    for bar in bars2:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.015, 1.08), f"{v:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=9)
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("Score (Unitless [0.0 - 1.0])", fontsize=10)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    caption_text = "Figure: Comparison of classification metrics across default and calibrated decision thresholds."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_realtime_latency_percentiles_plot(latency_stats, output_path):
    ensure_parent_dir(output_path)
    percentiles_keys = ["mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    percentile_labels = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    percentile_vals = [latency_stats[k] for k in percentiles_keys]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    bars = ax.bar(percentile_labels, percentile_vals, color="#7A68A6", edgecolor="black", linewidth=0.8, width=0.5, label="Pure Inference Latency")

    offset = max(percentile_vals) * 0.03 if max(percentile_vals) > 0 else 0.001
    for b, v in zip(bars, percentile_vals):
        ax.text(b.get_x() + b.get_width()/2, v + offset, f"{v:.4f} ms", ha="center", fontsize=9)

    ax.set_ylabel("Latency per Sample (ms/sample)", fontsize=10)
    ax.set_ylim(0, max(percentile_vals) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    caption_text = "Figure: Single-sample pure inference latency percentiles (batch_size=1) for real-time edge evaluation."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix_comparison(y_test, probabilities, threshold_default, threshold_calibrated, output_path):
    ensure_parent_dir(output_path)
    pred_default = (probabilities >= threshold_default).astype(np.uint8)
    pred_calibrated = (probabilities >= threshold_calibrated).astype(np.uint8)

    cm_default = confusion_matrix(y_test, pred_default, labels=[0, 1])
    cm_calibrated = confusion_matrix(y_test, pred_calibrated, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300)

    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0], annot_kws={"size": 10})
    axes[0].set_title(f"(a) Standard Threshold (T = {threshold_default:.2f})", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontsize=10)
    axes[0].set_ylabel("True Class", fontsize=10)
    axes[0].set_xticklabels(["Benign", "Attack"], fontsize=9)
    axes[0].set_yticklabels(["Benign", "Attack"], rotation=0, fontsize=9)

    sns.heatmap(cm_calibrated, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1], annot_kws={"size": 10})
    axes[1].set_title(f"(b) Calibrated Threshold (T = {threshold_calibrated:.2f})", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontsize=10)
    axes[1].set_ylabel("True Class", fontsize=10)
    axes[1].set_xticklabels(["Benign", "Attack"], fontsize=9)
    axes[1].set_yticklabels(["Benign", "Attack"], rotation=0, fontsize=9)

    caption_text = "Figure: Confusion matrix comparison on Wednesday test dataset under standard and calibrated thresholds."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_pr_figure(y_test, probabilities, output_path):
    ensure_parent_dir(output_path)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    avg_precision = average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300)

    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=1.8, label=f"RF (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Guess")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
    axes[0].set_title("(a) Receiver Operating Characteristic", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True, fontsize=9)

    axes[1].plot(recall, precision, color="#0072B2", lw=1.8, label=f"RF (PR-AUC = {avg_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (TPR)", fontsize=10)
    axes[1].set_ylabel("Precision (PPV)", fontsize=10)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=10)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", frameon=True, fontsize=9)

    caption_text = "Figure: Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves for Wednesday dataset."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(roc_auc_val), float(avg_precision)


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
    ax.legend(loc="upper right", frameon=True, fontsize=9)

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
    ax.legend(loc="upper right", frameon=True, fontsize=9)

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

    ax.set_title(f"Attack Probability Distribution (Calibrated T={threshold:.4f})", fontsize=10)
    ax.set_xlabel("Predicted Attack Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels_ = ax.get_legend_handles_labels()
    remapped = {
        "0": "Benign Traffic (0)",
        "1": "DDoS Traffic (1)",
    }
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels_):
        new_handles.append(h)
        new_labels.append(remapped.get(l, l))
    ax.legend(new_handles, new_labels, loc="upper right", frameon=True, fontsize=9)

    caption_text = "Figure: Predicted attack probability distribution by class with calibrated decision threshold."
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# Main Entry Point
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    friday_file = find_file(FRIDAY_FILENAME)
    wednesday_file = find_file(WEDNESDAY_FILENAME)
    
    # Load fair 200-trees model trained without Flow_Duration
    model = joblib.load(find_file(MODEL_FILENAME))

    # Single-thread execution for pure latency edge timing benchmark
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    # 1. Friday Threshold Calibration (F2-score on validation slice)
    optimal_threshold, friday_val_f2, friday_stats = calibrate_threshold_on_friday_validation(model, friday_file)

    # 2. Wednesday Dataset Stats
    wednesday_stats = get_dataset_statistics(wednesday_file)

    # 3. Real-Time Single-Sample Latency (Batch Size = 1, N = 5000)
    latency_stats, _ = benchmark_single_sample_latency(model, wednesday_file, n_benchmark_samples=5000)

    # 4. Streaming Inference
    y_test, probabilities_test, stream_stats, profile_data = evaluate_wednesday_streaming(model, wednesday_file)

    # 5. Metrics Evaluation
    default_threshold = 0.50
    metrics_default = evaluate_predictions(y_test, probabilities_test, default_threshold)
    metrics_calibrated = evaluate_predictions(y_test, probabilities_test, optimal_threshold)

    # 6. Plot Generations
    paths = {
        "metrics_bar_default": os.path.join(RESULTS_DIR, "rf_wednesday_comprehensive_metrics_default_no_flow_duration.png"),
        "metrics_bar_calibrated": os.path.join(RESULTS_DIR, "rf_wednesday_comprehensive_metrics_calibrated_no_flow_duration.png"),
        "confusion_matrix_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrices_comparison_no_flow_duration.png"),
        "roc_pr_curves": os.path.join(RESULTS_DIR, "rf_wednesday_roc_pr_curves_no_flow_duration.png"),
        "metrics_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_metrics_comparison_no_flow_duration.png"),
        "realtime_latency_percentiles": os.path.join(RESULTS_DIR, "rf_realtime_latency_percentiles_no_flow_duration.png"),
        "memory_usage": os.path.join(RESULTS_DIR, "rf_memory_usage_over_time_no_flow_duration.png"),
        "cpu_usage": os.path.join(RESULTS_DIR, "rf_cpu_usage_over_time_no_flow_duration.png"),
        "probability_distribution_calibrated": os.path.join(RESULTS_DIR, "rf_probability_distribution_calibrated_no_flow_duration.png"),
    }

    roc_auc_val, avg_precision = save_roc_pr_figure(y_test, probabilities_test, paths["roc_pr_curves"])

    save_comprehensive_metrics_bar(
        metrics_default, roc_auc_val, f"Standard T = {default_threshold:.2f}", paths["metrics_bar_default"]
    )
    save_comprehensive_metrics_bar(
        metrics_calibrated, roc_auc_val, f"Calibrated T = {optimal_threshold:.2f}", paths["metrics_bar_calibrated"]
    )
    save_confusion_matrix_comparison(
        y_test, probabilities_test, default_threshold, optimal_threshold, paths["confusion_matrix_comparison"]
    )
    save_metrics_comparison_figure(
        metrics_default, metrics_calibrated, optimal_threshold, roc_auc_val, paths["metrics_comparison"]
    )
    save_realtime_latency_percentiles_plot(latency_stats, paths["realtime_latency_percentiles"])

    # System Resource Profiles
    if len(profile_data["memory_trace_mb"]) > 0:
        save_memory_usage_plot(profile_data["memory_trace_mb"], paths["memory_usage"])

    if len(profile_data["cpu_trace_pct"]) > 0:
        save_cpu_usage_plot(profile_data["cpu_trace_pct"], paths["cpu_usage"])

    # Predicted Probability Distribution (Calibrated Threshold)
    save_probability_distribution(
        y_test,
        probabilities_test,
        optimal_threshold,
        paths["probability_distribution_calibrated"],
    )

    # Export JSON
    payload = {
        "ablation_scenario": "without_flow_duration",
        "active_features": FEATURE_COLUMNS,
        "threshold_calibration": {
            "default_threshold": float(default_threshold),
            "optimal_threshold": float(optimal_threshold),
            "friday_val_f2": float(friday_val_f2),
        },
        "realtime_latency_batch_size_1": latency_stats,
        "metrics_standard_threshold": {k: v for k, v in metrics_default.items() if k != "y_pred"},
        "metrics_calibrated_threshold": {k: v for k, v in metrics_calibrated.items() if k != "y_pred"},
        "roc_auc": float(roc_auc_val),
        "pr_auc": float(avg_precision),
        "system_profile": {
            "peak_memory_mb": float(stream_stats["peak_memory_mb"]),
            "avg_cpu_percent": float(stream_stats["avg_cpu_percent"]),
            "total_inference_time_sec": float(stream_stats["total_inference_time_sec"]),
        },
    }

    json_path = os.path.join(RESULTS_DIR, "rf_cross_day_evaluation_results_no_flow_duration.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"[SUCCESS] All figures (Times New Roman, 10pt, 300 DPI) and JSON exported to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
