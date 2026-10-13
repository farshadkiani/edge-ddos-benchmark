# -*- coding: utf-8 -*-


import os
import sys
import time
import json
import threading

import numpy as np
import pandas as pd
import joblib
import psutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

# =========================================================
# Matplotlib Publication Styling (Times New Roman, 10pt, 300 DPI)
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# =========================================================
# Config & Paths
# =========================================================
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "xgboost_attack_detector.pkl")

os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

TARGET_COL = "Label"
TEST_RATIO = 0.2
SIM_SAMPLE_SIZE = 50000
WARMUP_SAMPLES = 100
RANDOM_STATE = 42
RESOURCE_SAMPLE_INTERVAL = 0.1

# =========================================================
# File Verification & Data Loading
# =========================================================
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset file not found at: {DATA_PATH}")
    sys.exit(1)

print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

df = df[RAW_REQUIRED_COLUMNS].copy()

print("[INFO] Engineering features...")
df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
)
df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).reset_index(drop=True)

df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()
df["binary_label"] = (df[TARGET_COL] != "BENIGN").astype(np.int32)

X = df[FEATURE_COLUMNS].values.astype(np.float32)
y = df["binary_label"].values.astype(np.int32)

# Split train/test (Chronological split without random shuffling to preserve temporal behavior)
split_idx = int(len(df) * (1 - TEST_RATIO))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

print(f"[INFO] Train samples: {len(X_train)}, Test pool samples: {len(X_test)}")

# =========================================================
# Load or Train Model
# =========================================================
if os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading existing model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
else:
    print("[INFO] Training new XGBoost model...")
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train_end = time.perf_counter()
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model trained in {t_train_end - t_train_start:.2f}s and saved to: {MODEL_PATH}")

# Subsample test set for realistic simulation if larger than SIM_SAMPLE_SIZE
if len(X_test) > SIM_SAMPLE_SIZE:
    indices = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=SIM_SAMPLE_SIZE, replace=False)
    X_eval = X_test[indices]
    y_true = y_test[indices]
else:
    X_eval = X_test
    y_true = y_test

print(f"[INFO] Evaluated benchmark samples count: {len(X_eval)}")

# =========================================================
# Resource Monitoring Thread
# =========================================================
resource_cpu = []
resource_mem = []
resource_timestamps = []
monitor_stop = threading.Event()

def monitor_resources(interval=0.1):
    process = psutil.Process(os.getpid())
    start = time.perf_counter()

    while not monitor_stop.is_set():
        try:
            resource_timestamps.append(time.perf_counter() - start)
            resource_cpu.append(psutil.cpu_percent(interval=None))
            resource_mem.append(process.memory_info().rss / (1024 ** 2))
        except Exception:
            pass
        time.sleep(interval)

# =========================================================
# Warm-Up Stage
# =========================================================
print("[INFO] Running warm-up...")
warmup_count = min(WARMUP_SAMPLES, len(X_eval))
for i in range(warmup_count):
    x_single = X_eval[i : i + 1]
    _ = model.predict(x_single)

# =========================================================
# Single-Sample Pure Model Inference Benchmark (Batch Size = 1)
# =========================================================
print("[INFO] Starting real-time single-sample inference benchmark...")
monitor_thread = threading.Thread(
    target=monitor_resources,
    kwargs={"interval": RESOURCE_SAMPLE_INTERVAL},
    daemon=True,
)
monitor_thread.start()

all_latencies_ms = []
all_preds = []
all_probs = []
has_proba = hasattr(model, "predict_proba")

benchmark_start = time.perf_counter()

for idx in range(len(X_eval)):
    x_single = X_eval[idx : idx + 1]

    # Pure Model Inference Scope
    t0 = time.perf_counter()
    pred = model.predict(x_single)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    all_latencies_ms.append(latency_ms)
    all_preds.append(int(pred[0]))

    if has_proba:
        prob = model.predict_proba(x_single)[0, 1]
        all_probs.append(float(prob))

benchmark_end = time.perf_counter()

monitor_stop.set()
monitor_thread.join(timeout=2.0)

# =========================================================
# Metrics Calculation
# =========================================================
latencies = np.asarray(all_latencies_ms, dtype=np.float64)
y_pred = np.asarray(all_preds, dtype=np.int32)
y_prob = np.asarray(all_probs, dtype=np.float32) if has_proba else None

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

try:
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
except ValueError:
    roc_auc = None

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true, y_pred, labels=[0, 1], target_names=["BENIGN", "DDoS"], zero_division=0
)

mean_latency_ms = float(np.mean(latencies))
median_latency_ms = float(np.median(latencies))
p50_latency_ms = float(np.percentile(latencies, 50))
p90_latency_ms = float(np.percentile(latencies, 90))
p95_latency_ms = float(np.percentile(latencies, 95))
p99_latency_ms = float(np.percentile(latencies, 99))

throughput = (
    float(len(latencies) / np.sum(latencies / 1000.0))
    if np.sum(latencies) > 0
    else None
)

process = psutil.Process(os.getpid())
current_rss_mb = process.memory_info().rss / (1024 ** 2)
peak_rss_mb = float(max(resource_mem)) if resource_mem else float(current_rss_mb)
avg_cpu_percent = float(np.mean(resource_cpu)) if resource_cpu else None
peak_cpu_percent = float(max(resource_cpu)) if resource_cpu else None

# =========================================================
# Standardized Publication Plots (Times New Roman, 10pt, 300 DPI)
# =========================================================
print("[INFO] Generating publication-ready plots...")

# 1. Comprehensive Model Metrics Bar Chart
plt.figure(figsize=(6.2, 3.8))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
metric_values = [accuracy, precision, recall, f1, mcc]

if roc_auc is not None:
    metric_names.append("ROC-AUC")
    metric_values.append(roc_auc)

bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"][: len(metric_names)]
bars = plt.bar(metric_names, metric_values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.7)
plt.axhline(0, color="gray", linewidth=0.8, linestyle="-")
plt.ylim(-1.05 if mcc < 0 else 0, 1.08)
plt.title("Classification Performance Metrics (XGBoost)")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score [-]")

for bar, val in zip(bars, metric_values):
    va_align = "bottom" if val >= 0 else "top"
    offset = 0.02 if val >= 0 else -0.05
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + offset,
        f"{val:.4f}",
        ha="center",
        va=va_align,
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_metrics_bar_chart_xgb.png"), bbox_inches="tight")
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(4.5, 3.8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["BENIGN", "DDoS"],
    yticklabels=["BENIGN", "DDoS"],
    cbar=True,
    annot_kws={"size": 10, "family": "serif"},
)
plt.title("Confusion Matrix (XGBoost)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_confusion_matrix_xgb.png"), bbox_inches="tight")
plt.close()

# 3. ROC Curve
if roc_auc is not None and y_prob is not None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(4.5, 3.8))
    plt.plot(fpr, tpr, color="#1f77b4", lw=1.5, label=f"XGBoost (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_roc_curve_xgb.png"), bbox_inches="tight")
    plt.close()

# 4. Latency Percentiles Bar Chart
plt.figure(figsize=(5.0, 3.8))
p_labels = ["P50", "P90", "P95", "P99"]
p_values = [p50_latency_ms, p90_latency_ms, p95_latency_ms, p99_latency_ms]
bars = plt.bar(p_labels, p_values, color="#4C72B0", width=0.55, edgecolor="black", linewidth=0.7)
plt.title("Inference Latency Percentiles (Batch Size = 1)")
plt.xlabel("Latency Metric")
plt.ylabel("Latency [ms]")
for bar, val in zip(bars, p_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.005 * max(p_values),
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_latency_percentiles_xgb.png"), bbox_inches="tight")
plt.close()

# 5. Latency Distribution Histogram & KDE
plt.figure(figsize=(5.0, 3.8))
sns.histplot(latencies, bins=35, kde=True, color="#4C72B0", edgecolor="black", linewidth=0.5)
plt.axvline(mean_latency_ms, color="red", linestyle="--", lw=1.2, label=f"Mean = {mean_latency_ms:.4f} ms")
plt.axvline(p95_latency_ms, color="green", linestyle=":", lw=1.2, label=f"P95 = {p95_latency_ms:.4f} ms")
plt.title("Per-Sample Latency Distribution")
plt.xlabel("Inference Latency [ms]")
plt.ylabel("Sample Count")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_latency_distribution_xgb.png"), bbox_inches="tight")
plt.close()

# 6. Feature Importance Bar Chart
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": importances}).sort_values(
        by="Importance", ascending=True
    )
    plt.figure(figsize=(5.5, 3.8))
    plt.barh(fi_df["Feature"], fi_df["Importance"], color="#55A868", edgecolor="black", linewidth=0.7)
    plt.title("Feature Importance Analysis (XGBoost)")
    plt.xlabel("Gain Importance [-]")
    plt.ylabel("Extracted Feature")
    for idx, val in enumerate(fi_df["Importance"]):
        plt.text(val + 0.01, idx, f"{val:.4f}", va="center", fontsize=10)
    plt.xlim(0, max(fi_df["Importance"]) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_feature_importance_xgb.png"), bbox_inches="tight")
    plt.close()

# 7. Resource Usage Over Time (Dual Axis)
if resource_timestamps and resource_cpu and resource_mem:
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))

    line1 = ax1.plot(resource_timestamps, resource_cpu, color="#1f77b4", lw=1.2, label="CPU Utilization [%]")
    ax1.set_xlabel("Elapsed Time [s]")
    ax1.set_ylabel("CPU Utilization [%]", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    line2 = ax2.plot(resource_timestamps, resource_mem, color="#d62728", lw=1.2, label="Memory (RSS) [MB]")
    ax2.set_ylabel("Memory Usage [MB]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("System Resource Utilization Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_resource_usage_xgb.png"), bbox_inches="tight")
    plt.close()

# =========================================================
# Export Captions for Journal / Thesis
# =========================================================
caption_file = os.path.join(RESULTS_DIR, "figure_captions_xgb.txt")
with open(caption_file, "w", encoding="utf-8") as f:
    f.write("Standardized Figure Captions (Times New Roman, 10pt, 300 DPI)\n")
    f.write("=" * 65 + "\n\n")
    f.write("Figure 1: Comprehensive classification performance metrics (Accuracy, Precision, Recall, F1-Score, MCC, and ROC-AUC) for XGBoost.\n")
    f.write("Figure 2: Confusion matrix of the XGBoost model evaluated on single-sample test network flows.\n")
    f.write("Figure 3: Receiver Operating Characteristic (ROC) curve and area under the curve (AUC) metric for XGBoost.\n")
    f.write("Figure 4: Single-sample pure inference latency percentiles (P50, P90, P95, and P99) in milliseconds [ms].\n")
    f.write("Figure 5: Per-sample pure inference latency distribution with Kernel Density Estimation (KDE).\n")
    f.write("Figure 6: Relative feature importance scores based on Gain metric for XGBoost.\n")
    f.write("Figure 7: Time-series profiling of CPU utilization [%] and Resident Set Size (RSS) memory consumption [MB].\n")

# =========================================================
# Save JSON & Summary Report
# =========================================================
results = {
    "model": "XGBoost",
    "dataset": DATA_PATH,
    "model_path": MODEL_PATH,
    "evaluation_protocol": {
        "batch_size": 1,
        "latency_scope": "pure_model_inference",
        "feature_extraction_in_latency": False,
        "preprocessing_in_latency": False,
        "inference_in_latency": True,
        "warmup_samples": WARMUP_SAMPLES,
        "number_of_evaluated_samples": int(len(X_eval)),
    },
    "classification_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    },
    "latency_metrics_ms": {
        "mean": mean_latency_ms,
        "median": median_latency_ms,
        "p50": p50_latency_ms,
        "p90": p90_latency_ms,
        "p95": p95_latency_ms,
        "p99": p99_latency_ms,
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "std": float(np.std(latencies)),
    },
    "resource_metrics": {
        "peak_rss_mb": peak_rss_mb,
        "average_cpu_percent": avg_cpu_percent,
        "peak_cpu_percent": peak_cpu_percent,
    },
    "throughput_samples_per_second": throughput,
    "confusion_matrix": {
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    },
}

with open(os.path.join(RESULTS_DIR, "benchmark_xgb_indomain2017.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(os.path.join(RESULTS_DIR, "benchmark_report_xgb_indomain2017.txt"), "w", encoding="utf-8") as f:
    f.write("=== XGBOOST REAL-TIME BENCHMARK REPORT ===\n")
    f.write(report)
    f.write(f"\nMCC: {mcc:.6f}\n")
    if roc_auc is not None:
        f.write(f"ROC-AUC: {roc_auc:.6f}\n")
    f.write("\n=== LATENCY (Batch Size = 1, Pure Inference) ===\n")
    f.write(f"Mean Latency: {mean_latency_ms:.4f} ms\n")
    f.write(f"P50 (Median): {p50_latency_ms:.4f} ms\n")
    f.write(f"P95: {p95_latency_ms:.4f} ms\n")
    f.write(f"P99: {p99_latency_ms:.4f} ms\n")
    f.write(f"Throughput: {throughput:.2f} samples/sec\n")

print("\n[INFO] XGBoost benchmark and plot generation completed successfully.")
print(f"[INFO] Results saved to: {RESULTS_DIR}")
