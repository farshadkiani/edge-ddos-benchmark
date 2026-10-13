import os
import sys
import time
import json
import threading

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

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

# --------------------------------------------------
# TFLite Import
# --------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except ImportError:
        print("[ERROR] Neither tflite_runtime nor tensorflow found.")
        sys.exit(1)

# ==================================================
# Config
# ==================================================
MODEL_PATH = "results/lstm_attack_detector_final.tflite"
SCALER_PATH = "results/ddos_scaler.pkl"
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"

SEQUENCE_LENGTH = 10
TEST_RATIO = 0.2
THRESHOLD = 0.5
WARMUP_SAMPLES = 100
MAX_EVAL_SAMPLES = 50000
RESOURCE_SAMPLE_INTERVAL = 0.1

FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Protocol_Type",
    "Flow_Rate",
]

RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Label",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================================================
# File Verification
# ==================================================
for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (DATA_PATH, "Dataset")]:
    if not os.path.exists(path):
        print(f"[ERROR] {name} file not found at: {path}")
        sys.exit(1)

# ==================================================
# Load Model & Scaler
# ==================================================
print("[INFO] Loading TFLite model and RobustScaler...")
scaler = joblib.load(SCALER_PATH)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

# ==================================================
# Quantization / Dequantization Helpers
# ==================================================
def prepare_input_tensor(x, detail):
    dtype = detail["dtype"]
    if dtype == np.float32:
        return x.astype(np.float32)

    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale is None or scale == 0.0:
        return x.astype(dtype)

    q = np.round(x / scale + zero_point)
    if dtype == np.int8:
        q = np.clip(q, -128, 127)
    elif dtype == np.uint8:
        q = np.clip(q, 0, 255)

    return q.astype(dtype)

def extract_output_score(output, detail):
    out = output[0] if output.ndim > 1 else output
    score = float(out[0])

    dtype = detail["dtype"]
    if dtype == np.float32:
        return score

    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale is None or scale == 0.0:
        return score

    return (score - zero_point) * scale

def create_sequences(features, labels, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i + seq_len])
        y_seq.append(labels[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)

# ==================================================
# Data Loading & Feature Engineering
# ==================================================
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

missing_cols = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

df = df[RAW_REQUIRED_COLUMNS].copy()

for col in ["Total Fwd Packets", "Total Backward Packets", "Average Packet Size", "Flow Packets/s"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Label"] = df["Label"].astype(str).str.strip().str.upper()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=RAW_REQUIRED_COLUMNS).reset_index(drop=True)

df["Packet_Count"] = df["Total Fwd Packets"].fillna(0) + df["Total Backward Packets"].fillna(0)
df["Packet_Size"] = df["Average Packet Size"].fillna(0)
df["Protocol_Type"] = 6.0
df["Flow_Rate"] = df["Flow Packets/s"].fillna(0)
df["target"] = (df["Label"] != "BENIGN").astype(np.int32)

# Test Split (Last 20%)
split_idx = int(len(df) * (1 - TEST_RATIO))
df_test = df.iloc[split_idx:].reset_index(drop=True).copy()

X_test_raw = df_test[FEATURE_COLUMNS].values.astype(np.float32)
X_scaled = scaler.transform(X_test_raw).astype(np.float32)
y_all = df_test["target"].values.astype(np.int32)

print("[INFO] Creating sliding window sequences...")
X_eval, y_true_sim = create_sequences(X_scaled, y_all, SEQUENCE_LENGTH)

if len(X_eval) > MAX_EVAL_SAMPLES:
    X_eval = X_eval[:MAX_EVAL_SAMPLES]
    y_true_sim = y_true_sim[:MAX_EVAL_SAMPLES]

print(f"[INFO] Evaluated sequence count: {len(X_eval)}")

# ==================================================
# Resource Monitoring
# ==================================================
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

# ==================================================
# Warm-Up
# ==================================================
print("[INFO] Running warm-up...")
warmup_count = min(WARMUP_SAMPLES, len(X_eval))
for i in range(warmup_count):
    x_single = prepare_input_tensor(X_eval[i:i + 1], input_details[0])
    interpreter.set_tensor(input_index, x_single)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_index)

# ==================================================
# Single-Sample Inference Benchmark (Batch Size = 1)
# ==================================================
print("[INFO] Starting single-sample inference benchmark...")
monitor_thread = threading.Thread(
    target=monitor_resources,
    kwargs={"interval": RESOURCE_SAMPLE_INTERVAL},
    daemon=True,
)
monitor_thread.start()

all_scores = []
all_preds = []
all_latencies_ms = []

benchmark_start = time.perf_counter()

for idx in range(len(X_eval)):
    # آماده‌سازی تانسور خارج از تایمر
    input_tensor = prepare_input_tensor(X_eval[idx:idx + 1], input_details[0])

    # صرفاً Model Inference درون تایمر
    t0 = time.perf_counter()
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    score = extract_output_score(output, output_details[0])
    pred_label = 1 if score >= THRESHOLD else 0

    all_latencies_ms.append(latency_ms)
    all_scores.append(score)
    all_preds.append(pred_label)

benchmark_end = time.perf_counter()

monitor_stop.set()
monitor_thread.join(timeout=2.0)

# ==================================================
# Metrics Calculation
# ==================================================
latencies = np.asarray(all_latencies_ms, dtype=np.float64)
y_pred = np.asarray(all_preds, dtype=np.int32)
y_scores = np.asarray(all_scores, dtype=np.float32)
y_true = np.asarray(y_true_sim, dtype=np.int32)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)

try:
    roc_auc = roc_auc_score(y_true, y_scores)
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

# ==================================================
# Standardized Plots (Times New Roman, 10pt, 300 DPI)
# ==================================================
print("[INFO] Generating publication-ready plots...")

# 1. Comprehensive Model Metrics Bar Chart
plt.figure(figsize=(6.2, 3.8))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
metric_values = [accuracy, precision, recall, f1, mcc]

if roc_auc is not None:
    metric_names.append("ROC-AUC")
    metric_values.append(roc_auc)

bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"][:len(metric_names)]
bars = plt.bar(metric_names, metric_values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.7)
plt.axhline(0, color="gray", linewidth=0.8, linestyle="-")
plt.ylim(-1.05 if mcc < 0 else 0, 1.08)
plt.title("Classification Performance Metrics (LSTM - TFLite)")
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
        fontsize=10
    )
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_metrics_bar_chart_lstm.png"), bbox_inches="tight")
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
    annot_kws={"size": 10, "family": "serif"}
)
plt.title("Confusion Matrix (LSTM - TFLite)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_confusion_matrix_lstm.png"), bbox_inches="tight")
plt.close()

# 3. ROC Curve
if roc_auc is not None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(4.5, 3.8))
    plt.plot(fpr, tpr, color="#1f77b4", lw=1.5, label=f"LSTM (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.0, linestyle="--", label="Random Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_roc_curve_lstm.png"), bbox_inches="tight")
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
        fontsize=10
    )
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_latency_percentiles_lstm.png"), bbox_inches="tight")
plt.close()

# 5. Latency Distribution Histogram & KDE
plt.figure(figsize=(5.0, 3.8))
sns.histplot(latencies, bins=35, kde=True, color="#4C72B0", edgecolor="black", linewidth=0.5)
plt.axvline(mean_latency_ms, color="red", linestyle="--", lw=1.2, label=f"Mean = {mean_latency_ms:.3f} ms")
plt.axvline(p95_latency_ms, color="green", linestyle=":", lw=1.2, label=f"P95 = {p95_latency_ms:.3f} ms")
plt.title("Per-Sample Latency Distribution")
plt.xlabel("Inference Latency [ms]")
plt.ylabel("Sample Count")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_latency_distribution_lstm.png"), bbox_inches="tight")
plt.close()

# 6. Resource Usage Over Time (Dual Axis)
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
    plt.savefig(os.path.join(RESULTS_DIR, "fig_resource_usage_lstm.png"), bbox_inches="tight")
    plt.close()

# ==================================================
# Export Captions for Journal / Thesis
# ==================================================
caption_file = os.path.join(RESULTS_DIR, "figure_captions_lstm.txt")
with open(caption_file, "w", encoding="utf-8") as f:
    f.write("Standardized Figure Captions (Times New Roman, 10pt, 300 DPI)\n")
    f.write("=" * 65 + "\n\n")
    f.write("Figure 1: Classification performance metrics (Accuracy, Precision, Recall, F1-Score, MCC, and ROC-AUC) for LSTM (TFLite).\n")
    f.write("Figure 2: Confusion matrix of the quantized LSTM model evaluated on sequential test flow traffic.\n")
    f.write("Figure 3: Receiver Operating Characteristic (ROC) curve and corresponding AUC metric for LSTM.\n")
    f.write("Figure 4: Single-sample pure inference latency percentiles (P50, P90, P95, and P99) in milliseconds [ms].\n")
    f.write("Figure 5: Probability distribution and kernel density estimation (KDE) of per-sample inference latency [ms].\n")
    f.write("Figure 6: Time-series profiling of CPU utilization [%] and Resident Set Size (RSS) memory consumption [MB].\n")

# ==================================================
# Save JSON & Summary Report
# ==================================================
results = {
    "model": "LSTM (TFLite)",
    "dataset": DATA_PATH,
    "model_path": MODEL_PATH,
    "evaluation_protocol": {
        "batch_size": 1,
        "sequence_length": SEQUENCE_LENGTH,
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

with open(os.path.join(RESULTS_DIR, "benchmark_lstm_indomain2017.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n[INFO] LSTM benchmark and plot generation completed successfully.")
print(f"[INFO] Results saved to: {RESULTS_DIR}")
