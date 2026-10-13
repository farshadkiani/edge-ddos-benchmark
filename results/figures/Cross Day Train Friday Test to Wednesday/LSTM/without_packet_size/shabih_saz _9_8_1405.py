import os
import time
import json
import joblib
import psutil
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, matthews_corrcoef
)

# =========================================================
# ENV SETUP
# =========================================================
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================================
# PATHS & CONFIGURATION (Ablation: WITHOUT Packet_Size)
# =========================================================
WEDNESDAY_PATH = "Wednesday-workingHours.pcap_ISCX.csv"
RESULTS_DIR = "results"
SEQUENCE_LENGTH = 10
LATENCY_BENCHMARK_SAMPLES = 5000  # ۵۰۰۰ نمونه تک‌به‌تک برای تخمین دقیق صدک‌های تاخیر بلادرنگ

FEATURE_COLUMNS = ["Packet_Count", "Flow_Rate", "Flow_Duration"]

SCALER_PATH = os.path.join(RESULTS_DIR, "ddos_scaler_without_packet_size.pkl")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_lstm_attack_detector_without_packet_size.keras")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "best_threshold_without_packet_size.txt")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_2017_without_packet_size.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# UNIFIED THESIS MATPLOTLIB STYLING (Times New Roman, 10pt, 300 DPI)
# =========================================================
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

# =========================================================
# HELPERS
# =========================================================
def out(filename):
    return os.path.join(RESULTS_DIR, filename)

def compute_f2(precision, recall):
    denom = 4 * precision + recall
    return (5 * precision * recall) / denom if denom != 0 else 0.0

def evaluate_predictions(y_true, y_pred, y_prob, threshold):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(compute_f2(precision, recall))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)
    roc_auc_val = float(auc(fpr_curve, tpr_curve))
    pr_auc_val = float(average_precision_score(y_true, y_prob))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "f2_score": f2,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "mcc": mcc,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }

# =========================================================
# FAST DATA LOADER & PREPARATION
# =========================================================
process = psutil.Process(os.getpid())
print(f"[INFO] Loading dataset from: {WEDNESDAY_PATH}", flush=True)
headers = pd.read_csv(WEDNESDAY_PATH, nrows=0).columns.tolist()
stripped_headers = [h.strip() for h in headers]

logical_cols = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Flow Duration",
    "Label"
]

actual_usecols = []
for col in logical_cols:
    if col in stripped_headers:
        actual_usecols.append(headers[stripped_headers.index(col)])
    else:
        matched = [h for h in headers if col in h.strip()]
        if matched:
            actual_usecols.append(matched[0])
        else:
            raise KeyError(f"Column '{col}' not found in CSV headers.")

df_wed = pd.read_csv(WEDNESDAY_PATH, usecols=actual_usecols, dtype=object)
df_wed.columns = df_wed.columns.str.strip()

for c in ["Total Fwd Packets", "Total Backward Packets", "Flow Packets/s", "Flow Duration"]:
    df_wed[c] = pd.to_numeric(df_wed[c], errors="coerce").astype(np.float32)

df_wed = df_wed.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

print("[INFO] Computing composite edge features (Ablation: Without Packet Size)...", flush=True)
X_wed_raw = np.empty((len(df_wed), len(FEATURE_COLUMNS)), dtype=np.float32)
X_wed_raw[:, 0] = df_wed["Total Fwd Packets"].values + df_wed["Total Backward Packets"].values  # Packet_Count
X_wed_raw[:, 1] = df_wed["Flow Packets/s"].values                                             # Flow_Rate
X_wed_raw[:, 2] = df_wed["Flow Duration"].values                                              # Flow_Duration
y_wed_raw = (df_wed["Label"].str.strip().str.upper() != "BENIGN").astype(np.int32).values

del df_wed
gc.collect()

print("[INFO] Scaling features with pre-trained ablation scaler...", flush=True)
scaler = joblib.load(SCALER_PATH)
X_wed_scaled = scaler.transform(X_wed_raw).astype(np.float32)
del X_wed_raw
gc.collect()

y_wed = y_wed_raw[SEQUENCE_LENGTH:]

print("[INFO] Loading pre-trained ablation model and calibrated threshold...", flush=True)
model = tf.keras.models.load_model(BEST_MODEL_PATH)
with open(THRESHOLD_PATH, "r") as f:
    best_threshold = float(f.read().strip())

# =========================================================
# STEP 1: FAST FULL-DATASET PREDICTION FOR 100% SAMPLES
# =========================================================
print(f"[INFO] Predicting 100% dataset ({len(y_wed):,} samples) using fast chunking...", flush=True)
num_total = len(X_wed_scaled) - SEQUENCE_LENGTH
batch_chunk = 30000
y_wed_prob = np.zeros(num_total, dtype=np.float32)

for start in range(0, num_total, batch_chunk):
    end = min(start + batch_chunk, num_total)
    batch_seqs = np.empty((end - start, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)), dtype=np.float32)
    for idx, i in enumerate(range(start, end)):
        batch_seqs[idx] = X_wed_scaled[i:i + SEQUENCE_LENGTH]
    y_wed_prob[start:end] = model.predict(batch_seqs, batch_size=4096, verbose=0).ravel()
    del batch_seqs

gc.collect()

# =========================================================
# STEP 2: PURE REAL-TIME LATENCY & RESOURCE PROFILING (BATCH SIZE = 1)
# =========================================================
n_bench = min(LATENCY_BENCHMARK_SAMPLES, num_total)
print(f"[INFO] Benchmarking Single-Sample Latency & Resources on {n_bench:,} sequential samples...", flush=True)

bench_sequences = [np.expand_dims(X_wed_scaled[i:i + SEQUENCE_LENGTH], axis=0).astype(np.float32) for i in range(n_bench)]

@tf.function
def predict_single(x):
    return model(x, training=False)

# Warm-up (۵۰ بار اجرای اولیه برای تثبیت کش CPU و کامپایل گراف Keras)
for _ in range(50):
    _ = predict_single(bench_sequences[0])

# پایش منابع قبل از شروع بنچمارک بلادرنگ
_ = process.cpu_percent(interval=None)
mem_before = process.memory_info().rss / (1024 * 1024)

latencies_ms = np.empty(n_bench, dtype=np.float32)
peak_ram_mb = mem_before

t_bench_start = time.perf_counter()
for i in range(n_bench):
    x_single = bench_sequences[i]
    
    t0 = time.perf_counter()
    _ = predict_single(x_single)
    t1 = time.perf_counter()
    
    latencies_ms[i] = (t1 - t0) * 1000.0
    
    if i % 1000 == 0:
        current_ram = process.memory_info().rss / (1024 * 1024)
        if current_ram > peak_ram_mb:
            peak_ram_mb = current_ram

t_bench_end = time.perf_counter()
cpu_usage_pct = process.cpu_percent(interval=None)
mem_after = process.memory_info().rss / (1024 * 1024)
peak_ram_mb = max(peak_ram_mb, mem_after)
model_size_mb = os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024)

del bench_sequences
gc.collect()

# =========================================================
# EVALUATION & PERCENTILES
# =========================================================
mean_latency = float(np.mean(latencies_ms))
p50_latency, p90_latency, p95_latency, p99_latency = np.percentile(latencies_ms, [50, 90, 95, 99])

y_pred_05 = (y_wed_prob >= 0.50).astype(int)
y_pred_opt = (y_wed_prob >= best_threshold).astype(int)

res_05 = evaluate_predictions(y_wed, y_pred_05, y_wed_prob, 0.50)
res_opt = evaluate_predictions(y_wed, y_pred_opt, y_wed_prob, best_threshold)

print(f"[RESULT] Pure Latency (Batch Size = 1, N={n_bench}) -> Mean: {mean_latency:.3f} ms | P50: {p50_latency:.3f} ms | P95: {p95_latency:.3f} ms | P99: {p99_latency:.3f} ms")
print(f"[RESULT] Edge Resources -> CPU: {cpu_usage_pct:.1f}% | RAM Peak: {peak_ram_mb:.2f} MB | Model Size: {model_size_mb:.2f} MB")

# =========================================================
# UNIFIED THESIS PLOTTING (Times New Roman, 10pt, 300 DPI)
# =========================================================

# 1) Confusion Matrix (Threshold = 0.50)
cm_05 = res_05["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
sns.heatmap(cm_05, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 10, "family": "serif"})
ax.set_title("Confusion Matrix (Threshold = 0.50)", fontsize=10, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
ax.set_xticklabels(["BENIGN", "DDoS"], fontsize=10)
ax.set_yticklabels(["BENIGN", "DDoS"], rotation=90, fontsize=10)
plt.tight_layout()
plt.savefig(out("fig_confusion_matrix_keras_default_without_packet_size.png"))
plt.close()

# 2) Confusion Matrix (Calibrated Threshold)
cm_opt = res_opt["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4.2, 3.6))
sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
            annot_kws={"size": 10, "family": "serif"})
ax.set_title(f"Confusion Matrix (Calibrated Threshold = {best_threshold:.2f})", fontsize=10, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=10)
ax.set_ylabel("True Label", fontsize=10)
ax.set_xticklabels(["BENIGN", "DDoS"], fontsize=10)
ax.set_yticklabels(["BENIGN", "DDoS"], rotation=90, fontsize=10)
plt.tight_layout()
plt.savefig(out("fig_confusion_matrix_keras_balanced_without_packet_size.png"))
plt.close()

# 3) ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_wed, y_wed_prob)
prec_curve, rec_curve, _ = precision_recall_curve(y_wed, y_wed_prob)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
axes[0].plot(fpr, tpr, lw=1.5, color="#1f77b4", label=f"ROC (AUC = {res_opt['roc_auc']:.4f})")
axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=10, fontweight="bold")
axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=10)
axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=10)
axes[0].legend(loc="lower right", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(rec_curve, prec_curve, lw=1.5, color="#2ca02c", label=f"PR (AUC = {res_opt['pr_auc']:.4f})")
axes[1].set_title("Precision-Recall (PR) Curve", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Recall", fontsize=10)
axes[1].set_ylabel("Precision", fontsize=10)
axes[1].legend(loc="lower left", fontsize=10)
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(out("fig_roc_pr_curves_keras_without_packet_size.png"))
plt.close()

# 4) Standalone Metric Bar Chart for Default Threshold (0.50)
metric_keys = ["accuracy", "precision", "recall", "f1_score", "f2_score", "mcc", "roc_auc"]
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
vals_05 = [res_05[k] for k in metric_keys]

plt.figure(figsize=(7.5, 3.8))
bars_05 = plt.bar(metric_names, vals_05, color="#4C78A8", width=0.45)
for b, v in zip(bars_05, vals_05):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
plt.ylabel("Score", fontsize=10)
plt.title("Performance Metrics (Standard Threshold = 0.50)", fontsize=10, fontweight="bold")
plt.ylim(0, 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out("fig_metrics_threshold_050_without_packet_size.png"))
plt.close()

# 5) Standalone Metric Bar Chart for Calibrated Threshold
vals_opt = [res_opt[k] for k in metric_keys]

plt.figure(figsize=(7.5, 3.8))
bars_opt = plt.bar(metric_names, vals_opt, color="#59A14F", width=0.45)
for b, v in zip(bars_opt, vals_opt):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
plt.ylabel("Score", fontsize=10)
plt.title(f"Performance Metrics (Calibrated Threshold = {best_threshold:.2f})", fontsize=10, fontweight="bold")
plt.ylim(0, 1.15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out("fig_metrics_calibrated_threshold_without_packet_size.png"))
plt.close()


# 5.1) Unified Metrics Comparison Bar Chart (Default vs Calibrated)
x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8.5, 4.0))
bars1 = ax.bar(x - width/2, vals_05, width, label="Default Threshold (0.50)", color="#4C78A8")
bars2 = ax.bar(x + width/2, vals_opt, width, label=f"Calibrated Threshold ({best_threshold:.2f})", color="#59A14F")

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

ax.set_ylabel("Score", fontsize=10)
ax.set_title("Detection Performance Comparison (Without Packet Size)", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 1.20)
ax.legend(loc="lower right", frameon=True, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(out("fig_metrics_comparison_keras_without_packet_size.png"))
plt.close()



# 6) Edge Hardware Resource Utilization Chart
res_labels = ["CPU Usage (%)", "Peak RAM (MB)", "Model Size (MB)"]
res_values = [cpu_usage_pct, peak_ram_mb, model_size_mb]

plt.figure(figsize=(5.8, 3.8))
bars_res = plt.bar(res_labels, res_values, color=["#E45756", "#72B7B2", "#F58518"], width=0.45)
for b, v in zip(bars_res, res_values):
    plt.text(b.get_x() + b.get_width() / 2, v + (max(res_values) * 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=10)
plt.ylabel("Utilization / Size", fontsize=10)
plt.title("Edge Device Hardware & Memory Footprint", fontsize=10, fontweight="bold")
plt.ylim(0, max(res_values) * 1.18)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out("fig_hardware_resources_without_packet_size.png"))
plt.close()

# 7) Latency Percentiles (Mean, P50, P90, P95, P99)
percentile_names = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
percentile_values = [mean_latency, p50_latency, p90_latency, p95_latency, p99_latency]

plt.figure(figsize=(7, 3.8))
bars_p = plt.bar(percentile_names, percentile_values, color="#8172B2", width=0.45)
max_val = max(percentile_values)

for b, v in zip(bars_p, percentile_values):
    plt.text(b.get_x() + b.get_width() / 2, v + (max_val * 0.02), f"{v:.3f} ms", ha="center", va="bottom", fontsize=10)

plt.ylabel("Latency (ms)", fontsize=10)
plt.title(f"Pure Inference Latency Distribution (Batch Size = 1, N = {n_bench:,})", fontsize=10, fontweight="bold")
plt.ylim(0, max_val * 1.20)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(out("latency_percentiles_keras_without_packet_size.png"))
plt.close()

# 8) Sequential Latency Stability
plt.figure(figsize=(8.5, 3.5))
plt.plot(latencies_ms, color="#C44E52", linewidth=0.8, alpha=0.85, label="Single-Sample Latency")
plt.axhline(mean_latency, color="black", linestyle="--", linewidth=1.2, label=f"Mean = {mean_latency:.3f} ms")
plt.axhline(p99_latency, color="navy", linestyle=":", linewidth=1.2, label=f"P99 = {p99_latency:.3f} ms")
plt.xlabel("Sequential Test Sample Index", fontsize=10)
plt.ylabel("Latency (ms)", fontsize=10)
plt.title(f"Real-Time Inference Latency Stability (Batch Size = 1, N = {n_bench:,})", fontsize=10, fontweight="bold")
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(out("latency_stability_over_time_keras_without_packet_size.png"))
plt.close()

# =========================================================
# SAVE METRICS REPORT (JSON)
# =========================================================
report = {
    "evaluation_info": {
        "dataset_name": "CICIDS2017 Wednesday Evaluation Set",
        "ablation_experiment": "WITHOUT Packet_Size",
        "features_utilized": FEATURE_COLUMNS,
        "evaluation_mode": "Real-Time (Batch Size = 1 for Latency & Resources, Full Set for Metrics)",
        "total_test_samples": int(len(y_wed)),
        "latency_benchmark_samples": int(n_bench),
        "default_threshold": 0.50,
        "calibrated_threshold": best_threshold
    },
    "edge_hardware_resources": {
        "cpu_usage_percent": float(cpu_usage_pct),
        "ram_before_benchmark_mb": float(mem_before),
        "ram_after_benchmark_mb": float(mem_after),
        "peak_ram_mb": float(peak_ram_mb),
        "model_file_size_mb": float(model_size_mb)
    },
    "latency_analysis_ms": {
        "mean": mean_latency,
        "p50_median": p50_latency,
        "p90": p90_latency,
        "p95": p95_latency,
        "p99_tail": p99_latency,
        "min": float(np.min(latencies_ms)),
        "max": float(np.max(latencies_ms))
    },
    "metrics_standard_threshold_050": res_05,
    "metrics_calibrated_threshold_opt": res_opt
}

with open(METRICS_PATH, "w") as f:
    json.dump(report, f, indent=4)

print(f"[SUCCESS] Execution completed. Results and figures saved to: {RESULTS_DIR}", flush=True)
