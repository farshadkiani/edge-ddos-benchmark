"""
Cross-Dataset Real-Time Evaluation Pipeline for Lightweight DDoS Detection
Model: Random Forest
Training Dataset: Friday (CICIDS2017)
Evaluation Dataset: Wednesday (Cross-Dataset Scenario)
Feature Set: Packet_Count, Packet_Size, Flow_Duration (Flow_Rate excluded)
Standard: 100% Data for Evaluation | N=5000 (Batch Size = 1) for Latency & Hardware Profiling
"""

import os
import gc
import json
import time
import psutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef
)

# ============================================================
# 1. Matplotlib Configuration (Academic Standard)
# ============================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
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
    "mathtext.fontset": "stix"
})

RESULTS_DIR = "results"
LATENCY_SAMPLES = 5000  # نمونه‌های بنچ‌مارک تاخیر بلادرنگ تک‌به‌تک
FEATURES = ["Packet_Count", "Packet_Size", "Flow_Duration"]
LABEL_COL = "Label"

def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def out(filename):
    return os.path.join(RESULTS_DIR, filename)

# ============================================================
# 2. Robust Data Loading & Feature Extraction
# ============================================================
def load_and_preprocess_dataset(csv_path):
    """
    Loads dataset dynamically matching CICIDS2017 raw headers
    Constructs features: Packet_Count, Packet_Size, Flow_Duration
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    headers = pd.read_csv(csv_path, nrows=0).columns.tolist()
    stripped_headers = [h.strip() for h in headers]

    col_map = {
        "fwd_pkts": "Total Fwd Packets",
        "bwd_pkts": "Total Backward Packets",
        "pkt_size": "Average Packet Size",
        "duration": "Flow Duration",
        "label": "Label"
    }

    usecols = []
    for key, name in col_map.items():
        if name in stripped_headers:
            usecols.append(headers[stripped_headers.index(name)])
        else:
            matched = [h for h in headers if name.lower() in h.strip().lower()]
            if matched:
                usecols.append(matched[0])
            else:
                raise KeyError(f"Required column '{name}' not found in {csv_path}")

    df = pd.read_csv(csv_path, usecols=usecols, dtype=object)
    df.columns = df.columns.str.strip()

    num_cols = ["Total Fwd Packets", "Total Backward Packets", "Average Packet Size", "Flow Duration"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    X = np.empty((len(df), 3), dtype=np.float32)
    X[:, 0] = df["Total Fwd Packets"].values + df["Total Backward Packets"].values  # Packet_Count
    X[:, 1] = df["Average Packet Size"].values                                     # Packet_Size
    X[:, 2] = df["Flow Duration"].values                                          # Flow_Duration

    y = (df["Label"].str.strip().str.upper() != "BENIGN").astype(np.int32).values

    del df
    gc.collect()
    return X, y

# ============================================================
# 3. Comprehensive Metrics Calculation
# ============================================================
def calculate_metrics(y_true, y_pred, y_prob, threshold_val=0.5):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "threshold": float(threshold_val),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f2_score": f2,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": fpr_val,
        "false_negative_rate": fnr_val,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

def find_optimal_threshold_f2(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_th = 0.5
    best_f2 = -1.0
    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        score = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
        if score > best_f2:
            best_f2 = score
            best_th = th
    return float(best_th), float(best_f2)

# ============================================================
# 4. Latency & Hardware Profiling (Batch Size = 1)
# ============================================================
def profile_hardware_and_latency(model, X_test, n_samples):
    process = psutil.Process(os.getpid())
    n_bench = min(n_samples, len(X_test))
    
    # 50 Warm-up iterations
    for i in range(50):
        _ = model.predict_proba(X_test[i:i+1])

    _ = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    latencies_ms = np.empty(n_bench, dtype=np.float64)
    peak_ram_mb = mem_before

    for i in range(n_bench):
        sample = X_test[i:i+1]
        
        t0 = time.perf_counter()
        _ = model.predict_proba(sample)
        t1 = time.perf_counter()
        
        latencies_ms[i] = (t1 - t0) * 1000.0
        
        if i % 1000 == 0:
            current_mem = process.memory_info().rss / (1024 * 1024)
            if current_mem > peak_ram_mb:
                peak_ram_mb = current_mem

    cpu_usage_pct = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = max(peak_ram_mb, mem_after)

    p50, p90, p95, p99 = np.percentile(latencies_ms, [50, 90, 95, 99])
    mean_lat = float(np.mean(latencies_ms))
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    latency_stats = {
        "mean_latency_ms": mean_lat,
        "p50_ms": float(p50),
        "p90_ms": float(p90),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "throughput_samples_per_sec": float(throughput)
    }

    hardware_stats = {
        "cpu_usage_percent": float(cpu_usage_pct),
        "peak_ram_mb": float(peak_ram_mb),
        "ram_before_mb": float(mem_before),
        "ram_after_mb": float(mem_after)
    }

    return latencies_ms, latency_stats, hardware_stats

# ============================================================
# 5. Thesis Figures Generation (Uniform, 10pt, 300 DPI)
# ============================================================
def generate_thesis_plots(y_test, y_prob_all, res_05, res_calib, latencies_ms, cpu_pct, peak_ram, model_size):
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "f2_score", "mcc", "roc_auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]

    # 1. Confusion Matrix - Default Threshold (0.50)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(res_05["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 10, "family": "serif"})
    ax.set_title("Cross-Dataset CM (Threshold = 0.50)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xticklabels(["BENIGN", "DDoS"], fontsize=10)
    ax.set_yticklabels(["BENIGN", "DDoS"], rotation=90, fontsize=10)
    plt.tight_layout()
    plt.savefig(out("fig_rf_confusion_matrix_default_050.png"))
    plt.close()

    # 2. Confusion Matrix - Calibrated Threshold
    calib_th = res_calib["threshold"]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(res_calib["confusion_matrix"], annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
                annot_kws={"size": 10, "family": "serif"})
    ax.set_title(f"Cross-Dataset CM (Calibrated T = {calib_th:.2f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xticklabels(["BENIGN", "DDoS"], fontsize=10)
    ax.set_yticklabels(["BENIGN", "DDoS"], rotation=90, fontsize=10)
    plt.tight_layout()
    plt.savefig(out("fig_rf_confusion_matrix_calibrated.png"))
    plt.close()

    # 3. ROC & PR Curves
    fpr, tpr, _ = roc_curve(y_test, y_prob_all)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_all)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=1.5, label=f"ROC (AUC = {res_05['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.0)
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
    axes[0].set_title("Cross-Dataset ROC Curve", fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(rec_arr, prec_arr, color="#2ca02c", lw=1.5, label=f"PR (AUC = {res_05['pr_auc']:.4f})")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("Cross-Dataset PR Curve", fontsize=10, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out("fig_rf_roc_pr_curves.png"))
    plt.close()

    # 4. Metrics Bar Plot - Standard Threshold 0.50
    vals_05 = [res_05[k] for k in metric_keys]
    plt.figure(figsize=(7.5, 3.8))
    bars_05 = plt.bar(metric_names, vals_05, color="#4C78A8", width=0.45)
    for b, v in zip(bars_05, vals_05):
        plt.text(b.get_x() + b.get_width() / 2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.title("Cross-Dataset Classification Metrics (Default T = 0.50)", fontsize=10, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out("fig_rf_metrics_threshold_050.png"))
    plt.close()

    # 5. Metrics Bar Plot - Calibrated Threshold
    vals_cal = [res_calib[k] for k in metric_keys]
    plt.figure(figsize=(7.5, 3.8))
    bars_cal = plt.bar(metric_names, vals_cal, color="#59A14F", width=0.45)
    for b, v in zip(bars_cal, vals_cal):
        plt.text(b.get_x() + b.get_width() / 2, min(v + 0.02, 1.05), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Score", fontsize=10)
    plt.title(f"Cross-Dataset Classification Metrics (Calibrated T = {calib_th:.2f})", fontsize=10, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out("fig_rf_metrics_calibrated_threshold.png"))
    plt.close()

    # 6. Separate CPU Utilization Plot
    plt.figure(figsize=(4.5, 3.8))
    bar_cpu = plt.bar(["CPU Usage"], [cpu_pct], color="#E45756", width=0.35)
    plt.ylabel("Utilization Percentage (%)", fontsize=10)
    plt.title("Cross-Dataset Real-Time CPU Utilization", fontsize=10, fontweight="bold")
    plt.ylim(0, max(100.0, cpu_pct * 1.25))
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bar_cpu:
        plt.text(b.get_x() + b.get_width() / 2, cpu_pct + 2.0, f"{cpu_pct:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out("fig_rf_cpu_usage.png"))
    plt.close()

    # 7. Separate Memory & Storage Footprint Plot
    mem_labels = ["Peak RAM", "Model Size"]
    mem_values = [peak_ram, model_size]
    plt.figure(figsize=(5.5, 3.8))
    bars_mem = plt.bar(mem_labels, mem_values, color=["#72B7B2", "#F58518"], width=0.45)
    plt.ylabel("Memory Footprint (MB)", fontsize=10)
    plt.title("Cross-Dataset Memory & Storage Footprint", fontsize=10, fontweight="bold")
    plt.ylim(0, max(mem_values) * 1.25)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for b, v in zip(bars_mem, mem_values):
        plt.text(b.get_x() + b.get_width() / 2, v + (max(mem_values) * 0.02), f"{v:.2f} MB", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out("fig_rf_memory_usage.png"))
    plt.close()

    # 8. Real-Time Latency Percentiles
    p50, p90, p95, p99 = np.percentile(latencies_ms, [50, 90, 95, 99])
    mean_lat = float(np.mean(latencies_ms))
    p_names = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    p_vals = [mean_lat, p50, p90, p95, p99]

    plt.figure(figsize=(7, 3.8))
    bars_p = plt.bar(p_names, p_vals, color="#8172B2", width=0.45)
    max_p = max(p_vals)
    for b, v in zip(bars_p, p_vals):
        plt.text(b.get_x() + b.get_width() / 2, v + (max_p * 0.02), f"{v:.3f} ms", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Latency (ms)", fontsize=10)
    plt.title(f"Inference Latency Percentiles (Batch Size = 1, N = {len(latencies_ms):,})", fontsize=10, fontweight="bold")
    plt.ylim(0, max_p * 1.20)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out("fig_rf_latency_percentiles.png"))
    plt.close()

# ============================================================
# 6. Main Execution Pipeline
# ============================================================
def main():
    friday_path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    wednesday_path = "Wednesday-workingHours.pcap_ISCX.csv"

    # پشتیبانی از مسیر داخل پوشه data در صورت وجود
    if not os.path.exists(friday_path) and os.path.exists(os.path.join("data", friday_path)):
        friday_path = os.path.join("data", friday_path)
    if not os.path.exists(wednesday_path) and os.path.exists(os.path.join("data", wednesday_path)):
        wednesday_path = os.path.join("data", wednesday_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_save_path = out("rf_model_without_flow_rate.joblib")

    print("[INFO] Step 1: Loading Friday training dataset...", flush=True)
    X_train, y_train = load_and_preprocess_dataset(friday_path)

    print("[INFO] Step 2: Training Random Forest model (100 Trees, Depth 12)...", flush=True)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    print("[INFO] Step 3: Calibrating optimal threshold on Friday dataset...", flush=True)
    train_proba = rf.predict_proba(X_train)[:, 1]
    optimal_th, best_train_f2 = find_optimal_threshold_f2(y_train, train_proba)
    print(f"       Optimal Threshold (F2-Max): {optimal_th:.2f} (Train F2: {best_train_f2:.4f})", flush=True)

    joblib.dump(rf, model_save_path)
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)

    del X_train, y_train, train_proba
    gc.collect()

    print("[INFO] Step 4: Loading Wednesday Cross-Dataset...", flush=True)
    X_test, y_test = load_and_preprocess_dataset(wednesday_path)

    print(f"[INFO] Step 5: Fast full-dataset probability prediction ({len(y_test):,} samples)...", flush=True)
    y_proba_test = rf.predict_proba(X_test)[:, 1]

    print(f"[INFO] Step 6: Profiling Latency & Resources (Batch Size = 1 on {LATENCY_SAMPLES:,} samples)...", flush=True)
    latencies_ms, latency_stats, hardware_stats = profile_hardware_and_latency(rf, X_test, LATENCY_SAMPLES)
    hardware_stats["model_file_size_mb"] = model_size_mb

    # Evaluating metrics on 100% data
    preds_05 = (y_proba_test >= 0.50).astype(int)
    preds_calib = (y_proba_test >= optimal_th).astype(int)

    res_05 = calculate_metrics(y_test, preds_05, y_proba_test, 0.50)
    res_calib = calculate_metrics(y_test, preds_calib, y_proba_test, optimal_th)

    print("[INFO] Step 7: Generating publication-grade figures (Times New Roman, 10pt, 300 DPI)...", flush=True)
    generate_thesis_plots(
        y_test, y_proba_test, res_05, res_calib, latencies_ms,
        hardware_stats["cpu_usage_percent"],
        hardware_stats["peak_ram_mb"],
        model_size_mb
    )

    print("[INFO] Step 8: Exporting JSON report...", flush=True)
    results_payload = {
        "model": "Random Forest",
        "features_utilized": FEATURES,
        "evaluation_protocol": "Gold Standard (100% Data for Classification, N=5000 Batch Size = 1 for Latency)",
        "total_test_samples": int(len(y_test)),
        "latency_benchmark_samples": int(len(latencies_ms)),
        "metrics_default_050": res_05,
        "metrics_calibrated_optimal": res_calib,
        "edge_hardware_resources": hardware_stats,
        "real_time_latency_ms": latency_stats
    }

    report_path = out("rf_cross_dataset_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results_payload, f, indent=4)

    # Printing Summary Table
    print("\n" + "=" * 80)
    print("           RANDOM FOREST CROSS-DATASET EVALUATION RESULTS (FRIDAY -> WEDNESDAY)")
    print("=" * 80)
    calib_header = f"Calibrated (T = {optimal_th:.2f})"
    print(f"{'Metric':<25}{'Default (T = 0.50)':>25}{calib_header:>28}")
    print("-" * 80)
    for k, name in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                    ("f1_score", "F1-Score"), ("f2_score", "F2-Score"), ("mcc", "MCC"),
                    ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        print(f"{name:<25}{res_05[k]:>25.4f}{res_calib[k]:>28.4f}")
    print("-" * 80)
    print(f"Hardware Footprint  -> CPU: {hardware_stats['cpu_usage_percent']:.1f}% | Peak RAM: {hardware_stats['peak_ram_mb']:.2f} MB | Model Size: {model_size_mb:.2f} MB")
    print(f"Latency (Batch=1)   -> Mean: {latency_stats['mean_latency_ms']:.4f} ms | P50: {latency_stats['p50_ms']:.4f} ms | P95: {latency_stats['p95_ms']:.4f} ms | P99: {latency_stats['p99_ms']:.4f} ms")
    print(f"Throughput          -> {latency_stats['throughput_samples_per_sec']:.2f} samples/sec")
    print("=" * 80)
    print(f"[SUCCESS] All figures and JSON report saved in directory: {RESULTS_DIR}/\n")

if __name__ == "__main__":
    main()
