#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAM-Optimized Fair Cross-Day Evaluation Script
Models: Random Forest (Lightweight Flow-Level DDoS Detection)
Benchmark: CICIDS2017 (Train: Friday DDoS | Test: Wednesday)
Target Architecture: Edge IoT / Raspberry Pi 4
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
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. Global Academic Typography & Visual Theme (300 DPI, 10pt)
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.0)


# ============================================================
# 2. Paths and Global Configuration
# ============================================================
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

FRIDAY_FILENAME = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
WEDNESDAY_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"
MODEL_FILENAME = "random_forest_friday_model.pkl"
MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_FILENAME)

CHUNK_SIZE = 25000
N_LATENCY_STREAMING_SAMPLES = 5000  # Strict single-sample latency testing

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
# 3. Helpers & Preprocessing
# ============================================================
def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(payload, output_path):
    ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)


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

    print(f"[*] Profiling class distribution: {os.path.basename(file_path)}")
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


# ============================================================
# 4. Independent Validation Threshold Calibration (F2 + Tie-Break)
# ============================================================
def calibrate_threshold_on_friday_validation(model, friday_file):
    """
    شرایط ۵ گانه کالیبراسیون:
    ۱. استخراج صرفاً از Validation جمعه (بدون دخالت Wednesday)
    ۲. جستجوی گرید روی ۹۹ آستانه (0.01 تا 0.99)
    ۳. بیشینه‌سازی F2-Score
    ۴. قاعده شکستن تساوی (Tie-Breaking): بالاتر بودن Precision و سپس نزدیکی به 0.50
    """
    print("\n[*] Calibrating decision threshold on Friday temporal validation split...")

    friday_stats = get_dataset_statistics(friday_file)
    total_samples = friday_stats["n_rows_clean"]

    if total_samples < 2:
        raise ValueError("[ERROR] Friday dataset has insufficient valid samples.")

    # 20% پایانی داده‌های جمعه بر اساس توالی زمانی
    validation_start_index = int(total_samples * 0.80)

    print(
        f"    -> Friday clean flows: {total_samples:,} | "
        f"Validation slice (last 20%): {total_samples - validation_start_index:,}"
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

    y_val = np.concatenate(validation_y_parts).astype(np.uint8)
    prob_val = np.concatenate(validation_prob_parts).astype(np.float32)

    del validation_y_parts, validation_prob_parts
    gc.collect()

    thresholds = np.linspace(0.01, 0.99, 99)
    best_candidate = {
        "threshold": 0.50,
        "f2": -1.0,
        "precision": -1.0,
        "dist_to_half": 1.0
    }

    for t in thresholds:
        y_pred = (prob_val >= t).astype(np.uint8)
        current_f2 = fbeta_score(y_val, y_pred, beta=2.0, zero_division=0)
        current_prec = precision_score(y_val, y_pred, zero_division=0)
        dist_to_half = abs(t - 0.50)

        if current_f2 > best_candidate["f2"] + 1e-5:
            best_candidate = {
                "threshold": float(t),
                "f2": float(current_f2),
                "precision": float(current_prec),
                "dist_to_half": float(dist_to_half)
            }
        elif abs(current_f2 - best_candidate["f2"]) <= 1e-5:
            if current_prec > best_candidate["precision"] + 1e-5:
                best_candidate = {
                    "threshold": float(t),
                    "f2": float(current_f2),
                    "precision": float(current_prec),
                    "dist_to_half": float(dist_to_half)
                }
            elif abs(current_prec - best_candidate["precision"]) <= 1e-5:
                if dist_to_half < best_candidate["dist_to_half"]:
                    best_candidate = {
                        "threshold": float(t),
                        "f2": float(current_f2),
                        "precision": float(current_prec),
                        "dist_to_half": float(dist_to_half)
                    }

    best_threshold = best_candidate["threshold"]
    best_f2 = best_candidate["f2"]

    print(f"    [+] Calibrated Optimal Threshold: T = {best_threshold:.4f} (Validation F2: {best_f2:.6f})\n")

    del y_val, prob_val
    gc.collect()

    return best_threshold, float(best_f2), friday_stats


# ============================================================
# 5. Full Evaluation Core & Latency Profiling (Batch Size = 1)
# ============================================================
def compute_binary_metrics(y_true, y_pred, y_prob, threshold):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f2": float(f2),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true))
    }


def evaluate_wednesday_full_and_stream(model, wednesday_file, n_latency_samples=N_LATENCY_STREAMING_SAMPLES):
    print("[*] Evaluating 100% of Wednesday Cross-Day Test Dataset...")

    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    y_parts = []
    probability_parts = []
    latency_stream_subset = []
    
    mem_trace = []
    cpu_trace = []

    for chunk_id, (X_chunk, y_chunk) in enumerate(iter_prepared_chunks(wednesday_file), start=1):
        if len(y_chunk) == 0:
            continue

        # ذخیره نمونه‌ها برای اندازه‌گیری Batch Size = 1
        if len(latency_stream_subset) < n_latency_samples:
            needed = n_latency_samples - len(latency_stream_subset)
            latency_stream_subset.append(X_chunk[:needed])

        # ارزیابی سریع و کامل دسته‌ای برای کل دیتاست
        probs = model.predict_proba(X_chunk)[:, 1].astype(np.float32)

        y_parts.append(y_chunk.copy())
        probability_parts.append(probs)

        mem_trace.append(process.memory_info().rss / (1024 * 1024))
        cpu_trace.append(process.cpu_percent(interval=None))

        del X_chunk, y_chunk, probs
        gc.collect()

    y_test = np.concatenate(y_parts).astype(np.uint8)
    probabilities_test = np.concatenate(probability_parts).astype(np.float32)
    del y_parts, probability_parts
    gc.collect()

    # ------------------------------------------------------------
    # اندازه‌گیری استاندارد تأخیر با Batch Size = 1
    # ------------------------------------------------------------
    X_latency = np.vstack(latency_stream_subset)
    n_stream = len(X_latency)
    print(f"[*] Measuring Pure Inference Latency (Batch Size = 1 on {n_stream:,} samples)...")

    # فاز گرم‌کردن (Warm-up)
    for i in range(min(100, n_stream)):
        _ = model.predict_proba(X_latency[i:i+1])

    latencies_ms = np.zeros(n_stream, dtype=np.float64)

    for i in range(n_stream):
        x_single = X_latency[i:i+1]

        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()

        latencies_ms[i] = (t1 - t0) * 1000.0

    latency_stats = {
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "n_samples_profiled": int(n_stream),
        "peak_memory_mb": float(np.max(mem_trace)) if mem_trace else 0.0,
        "avg_cpu_percent": float(np.mean(cpu_trace)) if cpu_trace else 0.0,
    }

    profile_data = {
        "memory_trace_mb": np.array(mem_trace, dtype=np.float32),
        "cpu_trace_pct": np.array(cpu_trace, dtype=np.float32),
        "latencies_ms": latencies_ms
    }

    return y_test, probabilities_test, latency_stats, profile_data


# ============================================================
# 6. Standardized Publication Figures (300 DPI, 10pt Font)
# ============================================================
def save_single_confusion_matrix(y_true, y_pred, title, output_path, cmap="Blues"):
    ensure_parent_dir(output_path)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False,
        xticklabels=["Benign (0)", "Attack (1)"],
        yticklabels=["Benign (0)", "Attack (1)"],
        annot_kws={"size": 10}, ax=ax
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix_comparison(y_test, probabilities, threshold_default, threshold_calibrated, output_path):
    ensure_parent_dir(output_path)
    pred_default = (probabilities >= threshold_default).astype(np.uint8)
    pred_calibrated = (probabilities >= threshold_calibrated).astype(np.uint8)

    cm_default = confusion_matrix(y_test, pred_default, labels=[0, 1])
    cm_calibrated = confusion_matrix(y_test, pred_calibrated, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=300)

    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"],
                annot_kws={"size": 10}, ax=axes[0])
    axes[0].set_title(f"(a) Standard Threshold (T = {threshold_default:.2f})", fontsize=10)
    axes[0].set_xlabel("Predicted Label", fontsize=10)
    axes[0].set_ylabel("True Label", fontsize=10)

    sns.heatmap(cm_calibrated, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"],
                annot_kws={"size": 10}, ax=axes[1])
    axes[1].set_title(f"(b) Calibrated Threshold (T = {threshold_calibrated:.4f})", fontsize=10)
    axes[1].set_xlabel("Predicted Label", fontsize=10)
    axes[1].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_roc_pr_figure(y_test, probabilities, output_path):
    ensure_parent_dir(output_path)
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)

    axes[0].plot(fpr, tpr, color="#2b5c8f", lw=1.5, label=f"Random Forest (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7f8c8d", lw=1.0, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
    axes[0].set_title("(a) Receiver Operating Characteristic", fontsize=10)
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].plot(recall, precision, color="#36827f", lw=1.5, label=f"Random Forest (AP = {average_precision:.4f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Detection Rate)", fontsize=10)
    axes[1].set_ylabel("Precision (PPV)", fontsize=10)
    axes[1].set_title("(b) Precision-Recall Curve", fontsize=10)
    axes[1].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return float(roc_auc_val), float(average_precision), fpr, tpr, precision, recall


def save_comprehensive_metrics_bar(metrics, threshold_label, output_path):
    ensure_parent_dir(output_path)
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    vals = [
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["f2"], metrics["mcc"], metrics["roc_auc"]
    ]
    colors = ['#2b5c8f', '#36827f', '#44a187', '#73b879', '#a6c86a', '#dfd55b', '#e8a145']

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=300)
    bars = ax.bar(metric_names, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=10)
    ax.set_title(f"Classification Performance Metrics ({threshold_label})", fontsize=10)
    ax.set_ylim(-0.05, 1.15)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_metrics_comparison_figure(metrics_default, metrics_calibrated, threshold_calibrated, output_path):
    ensure_parent_dir(output_path)
    metrics_keys = ["accuracy", "precision", "recall", "f1", "f2", "mcc", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]

    vals_default = [metrics_default[k] for k in metrics_keys]
    vals_cal = [metrics_calibrated[k] for k in metrics_keys]

    x = np.arange(len(metric_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=300)
    rects1 = ax.bar(x - w/2, vals_default, width=w, label="Default (T = 0.50)", color="#34495e", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + w/2, vals_cal, width=w, label=f"Calibrated (T = {threshold_calibrated:.4f})", color="#e67e22", edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=10)
    ax.set_title("Classification Metrics Comparison: Default vs Calibrated Threshold", fontsize=10)
    ax.legend(loc="lower right", frameon=True)

    for r in list(rects1) + list(rects2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width() / 2.0, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_realtime_latency_percentiles_plot(latency_stats, output_path):
    ensure_parent_dir(output_path)
    keys = ["mean_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms"]
    labels = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    vals = [latency_stats[k] for k in keys]
    colors = ['#34495e', '#2980b9', '#16a085', '#f39c12', '#c0392b']

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.set_ylabel("Inference Latency (ms / sample)", fontsize=10)
    ax.set_title("Real-Time Single-Sample (Batch Size = 1) Latency Profile", fontsize=10)

    max_val = max(vals) if max(vals) > 0 else 1.0
    ax.set_ylim(0, max_val * 1.25)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + (max_val * 0.02), f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_memory_usage_plot(memory_trace_mb, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    ax.plot(np.linspace(0, 1, len(memory_trace_mb)), memory_trace_mb, color="#ba5555", lw=1.2, label="RSS Memory Footprint")
    ax.axhline(y=1024.0, color="r", linestyle="--", lw=1.0, label="1 GB Edge RAM Budget")
    ax.set_xlabel("Normalized Timeline", fontsize=10)
    ax.set_ylabel("Memory Consumption (MB)", fontsize=10)
    ax.set_title("Edge Memory Footprint Profile Over Evaluation", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_cpu_usage_plot(cpu_trace_pct, output_path):
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    ax.plot(np.linspace(0, 1, len(cpu_trace_pct)), cpu_trace_pct, color="#4e9a68", lw=1.2, label="Process CPU Utilization")
    ax.set_xlabel("Normalized Timeline", fontsize=10)
    ax.set_ylabel("CPU Utilization (%)", fontsize=10)
    ax.set_title("Processor Load Profile Over Evaluation", fontsize=10)
    max_cpu = max(cpu_trace_pct) if len(cpu_trace_pct) > 0 else 0
    ax.set_ylim(0, max(100.0, max_cpu * 1.1))
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_probability_distribution(y_true, probabilities, threshold, threshold_label, output_path):
    ensure_parent_dir(output_path)
    
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)
    
    # توزیع احتمال برای ترافیک عادی (Benign)
    sns.histplot(
        probabilities[y_true == 0],
        color='#2b5c8f',
        label='Benign Traffic (0)',
        stat="density",
        bins=40,
        kde=True,
        ax=ax
    )
    
    # توزیع احتمال برای ترافیک حمله (DDoS)
    sns.histplot(
        probabilities[y_true == 1],
        color='#e74c3c',
        label='DDoS Traffic (1)',
        stat="density",
        bins=40,
        kde=True,
        ax=ax
    )
    
    # خط آستانه تصمیم‌گیری
    ax.axvline(
        x=threshold,
        color="black",
        linestyle="--",
        lw=1.5,
        label=f"Threshold (T = {threshold:.4f})"
    )
    
    ax.set_xlabel("Predicted Attack Probability", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Attack Probability Distribution ({threshold_label})", fontsize=10)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper center", frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def print_final_results(metrics_default, metrics_calibrated, latency_stats, roc_auc_val, average_precision):
    calib_title = f"Calibrated (T={metrics_calibrated['threshold']:.4f})"
    print("\n" + "=" * 85)
    print(f"{'Performance Metric':<30} | {'Default (T=0.50)':<22} | {calib_title:<22}")
    print("=" * 85)

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall (Detection Rate)", "recall"),
        ("F1-Score", "f1"),
        ("F2-Score (Recall Priority)", "f2"),
        ("Matthews Corr. Coef. (MCC)", "mcc"),
        ("ROC-AUC Score", "roc_auc"),
        ("Specificity (TNR)", "specificity"),
        ("False Positive Rate (FPR)", "false_positive_rate"),
    ]

    for display_name, key in metric_names:
        print(
            f"{display_name:<30} | "
            f"{metrics_default.get(key, 0.0):<22.4f} | "
            f"{metrics_calibrated.get(key, 0.0):<22.4f}"
        )

    print("-" * 85)
    print("Real-Time Single-Sample Latency Profile (Batch Size = 1):")
    print(f"  • Mean Latency : {latency_stats['mean_latency_ms']:.4f} ms")
    print(f"  • P50 (Median) : {latency_stats['p50_latency_ms']:.4f} ms")
    print(f"  • P90 Latency  : {latency_stats['p90_latency_ms']:.4f} ms")
    print(f"  • P95 Latency  : {latency_stats['p95_latency_ms']:.4f} ms")
    print(f"  • P99 (Tail)   : {latency_stats['p99_latency_ms']:.4f} ms")
    print("=" * 85 + "\n")


# ============================================================
# 7. Main Workflow Execution
# ============================================================
def main():
    print("=" * 85)
    print(" RAM-OPTIMIZED FAIR CROSS-DAY REAL-TIME EVALUATION (RANDOM FOREST)")
    print("=" * 85)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    friday_file = find_file(FRIDAY_FILENAME)
    wednesday_file = find_file(WEDNESDAY_FILENAME)

    if not os.path.exists(MODEL_PATH):
        model_candidates = [
            MODEL_PATH,
            os.path.join("/mnt/data", MODEL_FILENAME),
            os.path.join("/app", MODEL_FILENAME),
            MODEL_FILENAME,
        ]
        found_model = next((c for c in model_candidates if os.path.exists(c)), None)
        if found_model is None:
            raise FileNotFoundError(f"[ERROR] Model file not found: {MODEL_FILENAME}")
        model_path = found_model
    else:
        model_path = MODEL_PATH

    print(f"[*] Workspace Directory  : {RESULTS_DIR}")
    print(f"[*] Friday Train/Val File: {friday_file}")
    print(f"[*] Wednesday Test File  : {wednesday_file}")
    print(f"[*] Pre-trained Model    : {model_path}")

    print("\n[*] Loading Random Forest Classifier...")
    model = joblib.load(model_path)

    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    # ۱. کالیبراسیون آستانه مستقل روی جمعه (Validation)
    optimal_threshold, friday_validation_f2, friday_stats = calibrate_threshold_on_friday_validation(model, friday_file)

    # ۲. شمارش و ارزیابی ۱۰۰٪ داده‌های چهارشنبه
    wednesday_stats = get_dataset_statistics(wednesday_file)
    print(
        f"[*] Wednesday Total Flows: {wednesday_stats['n_rows_clean']:,} | "
        f"Benign: {wednesday_stats['n_benign']:,} | "
        f"Attack: {wednesday_stats['n_attack']:,}"
    )

    y_test, probabilities_test, latency_stats, profile_data = evaluate_wednesday_full_and_stream(model, wednesday_file)

    # ۳. استخراج متریک‌ها
    default_threshold = 0.50
    metrics_default = compute_binary_metrics(y_test, (probabilities_test >= default_threshold).astype(np.uint8), probabilities_test, default_threshold)
    metrics_calibrated = compute_binary_metrics(y_test, (probabilities_test >= optimal_threshold).astype(np.uint8), probabilities_test, optimal_threshold)

    # ۴. ترسیم نمودارهای ۳۰۰ DPI و Times New Roman
    print("\n[*] Generating Standardized Academic Figures (300 DPI, 10pt Font)...")

    paths = {
        "confusion_matrix_default": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrix_default.png"),
        "confusion_matrix_calibrated": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrix_calibrated.png"),
        "confusion_matrix_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_confusion_matrices_comparison.png"),
        "metrics_calibrated_bar": os.path.join(RESULTS_DIR, "rf_wednesday_metrics_bar_calibrated.png"),
        "metrics_comparison": os.path.join(RESULTS_DIR, "rf_wednesday_metrics_comparison.png"),
        "roc_pr_curves": os.path.join(RESULTS_DIR, "rf_wednesday_roc_pr_curves.png"),
        "latency_percentiles": os.path.join(RESULTS_DIR, "rf_latency_percentiles.png"),
        "probability_distribution_calibrated": os.path.join(RESULTS_DIR, "rf_probability_distribution_calibrated.png"),
        "memory_usage": os.path.join(RESULTS_DIR, "rf_memory_usage_over_time.png"),
        "cpu_usage": os.path.join(RESULTS_DIR, "rf_cpu_usage_over_time.png"),
    }

    roc_auc_val, average_precision, fpr, tpr, precision_curve, recall_curve = save_roc_pr_figure(
        y_test, probabilities_test, paths["roc_pr_curves"]
    )

    save_single_confusion_matrix(
        y_test, (probabilities_test >= default_threshold).astype(np.uint8),
        "Confusion Matrix (Default Threshold T=0.50)", paths["confusion_matrix_default"], cmap="Blues"
    )

    save_single_confusion_matrix(
        y_test, (probabilities_test >= optimal_threshold).astype(np.uint8),
        f"Confusion Matrix (Calibrated Threshold T={optimal_threshold:.4f})", paths["confusion_matrix_calibrated"], cmap="Oranges"
    )

    save_confusion_matrix_comparison(
        y_test, probabilities_test, default_threshold, optimal_threshold, paths["confusion_matrix_comparison"]
    )

    save_comprehensive_metrics_bar(
        metrics_calibrated, f"Calibrated T={optimal_threshold:.4f}", paths["metrics_calibrated_bar"]
    )

    save_metrics_comparison_figure(
        metrics_default, metrics_calibrated, optimal_threshold, paths["metrics_comparison"]
    )

    save_realtime_latency_percentiles_plot(
        latency_stats, paths["latency_percentiles"]
    )

    save_probability_distribution(
        y_test, probabilities_test, optimal_threshold, f"Calibrated T={optimal_threshold:.4f}", paths["probability_distribution_calibrated"]
    )

    if len(profile_data["memory_trace_mb"]) > 0:
        save_memory_usage_plot(profile_data["memory_trace_mb"], paths["memory_usage"])

    if len(profile_data["cpu_trace_pct"]) > 0:
        save_cpu_usage_plot(profile_data["cpu_trace_pct"], paths["cpu_usage"])

    # ۵. خروجی JSON جامع
    results_payload = {
        "experiment": {
            "model": "Random Forest",
            "train_validation_domain": FRIDAY_FILENAME,
            "test_domain": WEDNESDAY_FILENAME,
            "features": FEATURE_COLUMNS,
            "target_hardware": "Raspberry Pi 4 / Edge IoT"
        },
        "friday_validation_statistics": friday_stats,
        "wednesday_test_statistics": wednesday_stats,
        "calibration": {
            "metric": "F2-Score with Precision Tie-Break",
            "default_threshold": default_threshold,
            "calibrated_threshold": optimal_threshold,
            "validation_f2_score": friday_validation_f2
        },
        "latency_profile_batch_size_1_ms": latency_stats,
        "metrics_default_t050": metrics_default,
        "metrics_calibrated": metrics_calibrated,
        "discrimination": {
            "roc_auc": roc_auc_val,
            "average_precision": average_precision
        },
        "figures": paths
    }

    json_path = os.path.join(RESULTS_DIR, "rf_crossday_evaluation_summary.json")
    save_json(results_payload, json_path)

    print_final_results(
        metrics_default, metrics_calibrated, latency_stats, roc_auc_val, average_precision
    )

    del model, y_test, probabilities_test
    gc.collect()
    print("[+] Cross-day evaluation completed successfully.")


if __name__ == "__main__":
    main()
