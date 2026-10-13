#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import json
import time
import math
import threading

# تنظیمات بهینه‌سازی پردازنده و محیط برای ثبات خروجی
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import joblib
import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    matthews_corrcoef,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Paths and Constants (همگام‌سازی شده با Logistic Regression)
# =========================================================
FILE_PATH = "DrDoS_DNS.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_RATIO = 0.2
RANDOM_STATE = 42
MONITOR_INTERVAL = 0.1
NUM_INFERENCE_RUNS = 5
LATENCY_SAMPLE_LIMIT = 5000       # همگام با LR (5,000 نمونه)
LATENCY_WARMUP_SAMPLES = 500      # همگام با LR (500 نمونه Warm-up)
TRAIN_RATIO_MAJORITY_TO_MINORITY = 20
CHUNK_SIZE = 50000
THRESHOLD = 0.50

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

MODEL_BEST_PATH = os.path.join(RESULTS_DIR, "lstm_model_best.h5")
SCALER_PATH = os.path.join(RESULTS_DIR, "scaler_lstm.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_lstm.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report_lstm.txt")
RESOURCE_JSON_PATH = os.path.join(RESULTS_DIR, "resource_monitor_lstm.json")

# نمودارها
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_lstm.png")
CONFUSION_MATRIX_NORM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_normalized_lstm.png")
METRICS_BAR_PATH = os.path.join(RESULTS_DIR, "metrics_bar_chart_lstm.png")
PER_CLASS_METRICS_PATH = os.path.join(RESULTS_DIR, "per_class_metrics_lstm.png")

LATENCY_DIST_PATH = os.path.join(RESULTS_DIR, "latency_distribution_lstm.png")
LATENCY_BOXPLOT_PATH = os.path.join(RESULTS_DIR, "latency_boxplot_lstm.png")
LATENCY_CDF_PATH = os.path.join(RESULTS_DIR, "latency_cdf_lstm.png")
LATENCY_PERCENTILES_PATH = os.path.join(RESULTS_DIR, "latency_percentiles_lstm.png")
LATENCY_OVER_TIME_PATH = os.path.join(RESULTS_DIR, "latency_stability_over_time_lstm.png")
PRED_TIME_RUNS_PATH = os.path.join(RESULTS_DIR, "prediction_time_runs_lstm.png")

ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve_lstm.png")
PR_CURVE_PATH = os.path.join(RESULTS_DIR, "precision_recall_curve_lstm.png")

RESOURCE_CPU_PATH = os.path.join(RESULTS_DIR, "resource_cpu_usage_lstm.png")
RESOURCE_MEM_PATH = os.path.join(RESULTS_DIR, "resource_memory_usage_lstm.png")
RESOURCE_SUMMARY_PATH = os.path.join(RESULTS_DIR, "resource_summary_lstm.png")

# =========================================================
# Plot Styling (Thesis Standard: 10pt, Times New Roman, 300 DPI)
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
sns.set_theme(style="whitegrid", rc={"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})

# =========================================================
# Resource Monitor
# =========================================================
class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.running = False
        self.thread = None
        self.timestamps = []
        self.cpu_percents = []
        self.memory_mb = []

    def _collect(self):
        self.process.cpu_percent(interval=None)
        start = time.time()
        while self.running:
            try:
                now = time.time() - start
                cpu = self.process.cpu_percent(interval=None)
                mem = self.process.memory_info().rss / (1024 * 1024)
                self.timestamps.append(now)
                self.cpu_percents.append(cpu)
                self.memory_mb.append(mem)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.timestamps = []
        self.cpu_percents = []
        self.memory_mb = []
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.cpu_percents or not self.memory_mb:
            return {"avg_cpu": 0.0, "max_cpu": 0.0, "avg_mem": 0.0, "max_mem": 0.0}
        return {
            "avg_cpu": float(np.mean(self.cpu_percents)),
            "max_cpu": float(np.max(self.cpu_percents)),
            "avg_mem": float(np.mean(self.memory_mb)),
            "max_mem": float(np.max(self.memory_mb)),
        }

# =========================================================
# Visualization Functions
# =========================================================
def save_confusion_matrix(cm):
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["BENIGN", "DDoS"], yticklabels=["BENIGN", "DDoS"],
        annot_kws={"size": 10}, cbar=True
    )
    plt.title("Confusion Matrix - LSTM")
    plt.xlabel("Predicted Class [-]")
    plt.ylabel("Actual Class [-]")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

def save_normalized_confusion_matrix(cm):
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, cm_sum, where=cm_sum != 0)
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".4f", cmap="Blues",
        xticklabels=["BENIGN", "DDoS"], yticklabels=["BENIGN", "DDoS"],
        annot_kws={"size": 10}, cbar=True
    )
    plt.title("Normalized Confusion Matrix - LSTM")
    plt.xlabel("Predicted Class [-]")
    plt.ylabel("Actual Class [-]")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_NORM_PATH)
    plt.close()

def save_metrics_bar_chart(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    values = [
        metrics.get("accuracy", 0.0), metrics.get("precision", 0.0),
        metrics.get("recall", 0.0), metrics.get("f1_score", 0.0),
        metrics.get("f2_score", 0.0), metrics.get("mcc", 0.0),
        metrics.get("roc_auc", 0.0)
    ]
    plt.figure(figsize=(8.5, 4.5))
    palette = sns.color_palette("mako", len(names))
    ax = sns.barplot(x=names, y=values, palette=palette, edgecolor="black", linewidth=0.5)
    plt.ylim(0.0, 1.15)
    plt.title("Key Evaluation Metrics - LSTM")
    plt.xlabel("Evaluation Metric [-]")
    plt.ylabel("Score [-]")
    for i, v in enumerate(values):
        ax.text(i, min(v + 0.02, 1.08), f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(METRICS_BAR_PATH)
    plt.close()

def save_per_class_metrics(y_test, y_pred):
    rows = []
    for cls, name in [(0, "BENIGN"), (1, "DDoS")]:
        rows.append((name, "Precision", precision_score(y_test, y_pred, pos_label=cls, zero_division=0)))
        rows.append((name, "Recall", recall_score(y_test, y_pred, pos_label=cls, zero_division=0)))
        rows.append((name, "F1-Score", f1_score(y_test, y_pred, pos_label=cls, zero_division=0)))
        rows.append((name, "F2-Score", fbeta_score(y_test, y_pred, beta=2, pos_label=cls, zero_division=0)))
        
    plot_df = pd.DataFrame(rows, columns=["Class", "Metric", "Score"])
    plt.figure(figsize=(7.5, 4.5))
    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Class", palette="Set2", edgecolor="black", linewidth=0.5)
    plt.ylim(0, 1.15)
    plt.title("Per-Class Evaluation Metrics - LSTM")
    plt.xlabel("Metric [-]")
    plt.ylabel("Score [-]")
    plt.legend(loc="lower right")
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(PER_CLASS_METRICS_PATH)
    plt.close()

def save_latency_visualizations(latencies, run_times):
    mean_val = float(np.mean(latencies))
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    # 1. Distribution
    plt.figure(figsize=(7, 4.5))
    sns.histplot(latencies, bins=40, kde=True, color="#3470a3")
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.4f} ms")
    plt.axvline(p50, color="orange", linestyle="-", linewidth=1.2, label=f"Median: {p50:.4f} ms")
    plt.title("Single-Sample Inference Latency Distribution (LSTM, Batch Size = 1)")
    plt.xlabel("Inference Latency [ms]")
    plt.ylabel("Frequency [-]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(LATENCY_DIST_PATH)
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(5, 4.5))
    sns.boxplot(y=latencies, color="#69b3a2")
    plt.title("Inference Latency Boxplot (LSTM, Batch Size = 1)")
    plt.ylabel("Inference Latency [ms]")
    plt.tight_layout()
    plt.savefig(LATENCY_BOXPLOT_PATH)
    plt.close()

    # 3. CDF
    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    plt.figure(figsize=(7, 4.5))
    plt.plot(sorted_lat, cdf, color="#1b7837", linewidth=1.5, label="Empirical CDF")
    plt.title("Cumulative Distribution Function (CDF) - LSTM")
    plt.xlabel("Inference Latency [ms]")
    plt.ylabel("Cumulative Probability [-]")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(LATENCY_CDF_PATH)
    plt.close()

    # 4. Percentiles Bar Chart
    names = ['Mean', 'P50', 'P90', 'P95', 'P99']
    values = [mean_val, p50, p90, p95, p99]
    plt.figure(figsize=(7, 4.5))
    colors = sns.color_palette("Blues_d", len(names))
    ax = sns.barplot(x=names, y=values, palette=colors, edgecolor="black", linewidth=0.5)
    plt.title("Inference Latency Percentiles - LSTM (Batch Size = 1)")
    plt.xlabel("Percentile [-]")
    plt.ylabel("Latency [ms]")
    for i, v in enumerate(values):
        ax.text(i, v + (max(values) * 0.02), f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(LATENCY_PERCENTILES_PATH)
    plt.close()

    # 5. Stability over Time
    plt.figure(figsize=(8.5, 4.5))
    plt.plot(range(len(latencies)), latencies, color="#2b5c8f", alpha=0.6, linewidth=0.8, label="Sample Latency")
    plt.axhline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean ({mean_val:.4f} ms)")
    plt.title("Single-Sample Inference Latency Stability Across Evaluations (LSTM)")
    plt.xlabel("Sample Index [-]")
    plt.ylabel("Latency [ms]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(LATENCY_OVER_TIME_PATH)
    plt.close()

    # 6. Batch Runs Time
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(run_times) + 1), run_times, marker="o", color="#756bb1", linewidth=1.5)
    plt.title("Batch Inference Time Across Runs - LSTM")
    plt.xlabel("Simulation Run Index [-]")
    plt.ylabel("Execution Time [s]")
    plt.tight_layout()
    plt.savefig(PRED_TIME_RUNS_PATH)
    plt.close()

def save_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr, tpr, color="#d95f02", linewidth=1.5, label=f"ROC (AUC = {roc_auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="#7570b3", linestyle="--", linewidth=1.0)
    plt.xlabel("False Positive Rate [-]")
    plt.ylabel("True Positive Rate [-]")
    plt.title("ROC Curve - LSTM")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()

def save_pr_curve(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(6, 4.5))
    plt.plot(recall, precision, color="#b2182b", linewidth=1.5, label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall [-]")
    plt.ylabel("Precision [-]")
    plt.title("Precision-Recall Curve - LSTM")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PR_CURVE_PATH)
    plt.close()

def save_resource_usage_plots(resource_data):
    t, cpu, mem = resource_data["timestamps"], resource_data["cpu_percents"], resource_data["memory_mb"]
    if len(t) == 0:
        return
    plt.figure(figsize=(7, 4.5))
    plt.plot(t, cpu, color="#e41a1c", linewidth=1.2)
    plt.fill_between(t, cpu, color="#e41a1c", alpha=0.15)
    plt.title("CPU Utilization Profile During Inference (LSTM)")
    plt.xlabel("Time [s]")
    plt.ylabel("CPU Utilization [%]")
    plt.tight_layout()
    plt.savefig(RESOURCE_CPU_PATH)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(t, mem, color="#377eb8", linewidth=1.2)
    plt.fill_between(t, mem, color="#377eb8", alpha=0.15)
    plt.title("Memory Footprint Profile During Inference (LSTM)")
    plt.xlabel("Time [s]")
    plt.ylabel("Memory [MB]")
    plt.tight_layout()
    plt.savefig(RESOURCE_MEM_PATH)
    plt.close()

def save_resource_summary_plot(resource_summary):
    plot_df = pd.DataFrame({
        "Metric": ["Avg CPU", "Max CPU", "Avg RAM", "Max RAM"],
        "Unit": ["[%]", "[%]", "[MB]", "[MB]"],
        "Value": [resource_summary["avg_cpu"], resource_summary["max_cpu"], resource_summary["avg_mem"], resource_summary["max_mem"]]
    })
    plot_df["Label"] = plot_df["Metric"] + " " + plot_df["Unit"]
    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(data=plot_df, x="Label", y="Value", palette="vlag", edgecolor="black", linewidth=0.5)
    plt.title("Hardware Resource Consumption Summary - LSTM")
    plt.xlabel("Resource Metric [-]")
    plt.ylabel("Usage Magnitude [-]")
    for i, v in enumerate(plot_df["Value"]):
        ax.text(i, v + (max(plot_df["Value"]) * 0.02), f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESOURCE_SUMMARY_PATH)
    plt.close()

# =========================================================
# Main Pipeline
# =========================================================
def main():
    print("[INFO] Starting standard training and evaluation pipeline for LSTM...")
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # 1. Load Dataset
    print("[INFO] Loading and preprocessing dataset...")
    header_df = pd.read_csv(FILE_PATH, nrows=0)
    col_mapping = {c.strip(): c for c in header_df.columns}

    missing_cols = [c for c in RAW_REQUIRED_COLUMNS if c not in col_mapping]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    actual_cols_to_load = [col_mapping[c] for c in RAW_REQUIRED_COLUMNS]
    df = pd.read_csv(FILE_PATH, usecols=actual_cols_to_load, low_memory=True)
    df.columns = df.columns.str.strip()
    df = df[RAW_REQUIRED_COLUMNS].copy()

    # Feature Engineering
    df["Packet_Count"] = (
        pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
    )
    df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
    df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
    df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["Label"]).reset_index(drop=True)

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    df["binary_label"] = (df["Label"] != "BENIGN").astype(np.int8)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["binary_label"].to_numpy(dtype=np.int8)
    del df; gc.collect()

    # 2. Stratified Train / Test Split
    print("[INFO] Splitting dataset with StratifiedShuffleSplit (80% Train, 20% Test)...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train_raw, y_train_raw = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    del X, y; gc.collect()

    print(f"  - Raw Train Samples: {len(X_train_raw):,} (BENIGN: {np.sum(y_train_raw == 0):,}, DDoS: {np.sum(y_train_raw == 1):,})")
    print(f"  - Test Samples (Untouched): {len(X_test_raw):,} (BENIGN: {np.sum(y_test == 0):,}, DDoS: {np.sum(y_test == 1):,})")

    # 3. Hybrid Downsampling on Train Only (20:1 Ratio)
    print("[INFO] Applying Hybrid Downsampling strictly on the Training set (20:1)...")
    train_df = pd.DataFrame(X_train_raw, columns=FEATURE_COLUMNS)
    train_df["binary_label"] = y_train_raw

    minority_df = train_df[train_df["binary_label"] == 0]
    majority_df = train_df[train_df["binary_label"] == 1]

    minority_count = len(minority_df)
    majority_count = len(majority_df)

    majority_target_count = min(minority_count * TRAIN_RATIO_MAJORITY_TO_MINORITY, majority_count)
    majority_downsampled = majority_df.sample(n=majority_target_count, random_state=RANDOM_STATE)

    hybrid_train_df = pd.concat([minority_df, majority_downsampled], axis=0).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X_train = hybrid_train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = hybrid_train_df["binary_label"].to_numpy(dtype=np.int8)
    del train_df, minority_df, majority_df, majority_downsampled, hybrid_train_df; gc.collect()

    print(f"  - Downsampled Train Samples: {len(X_train):,} (BENIGN: {np.sum(y_train == 0):,}, DDoS: {np.sum(y_train == 1):,})")

    # 4. Feature Scaling (همگام با StandardScaler در اسکریپت LR)
    print("[INFO] Fitting StandardScaler on downsampled train set and transforming test set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_raw)
    joblib.dump(scaler, SCALER_PATH)

    # Reshape for LSTM: (samples, timesteps=4, features=1)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], len(FEATURE_COLUMNS), 1))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], len(FEATURE_COLUMNS), 1))

    # 5. Build and Train Lightweight LSTM
    print("[INFO] Building and training lightweight LSTM architecture...")
    model = Sequential([
        LSTM(32, input_shape=(len(FEATURE_COLUMNS), 1), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_BEST_PATH, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    train_start = time.time()
    model.fit(
        X_train_lstm,
        y_train,
        epochs=15,
        batch_size=512,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - train_start
    print(f"[INFO] Model trained successfully in {train_time:.2f} s.")

    # 6. Inference Resource Monitoring (Batch Predictions)
    print("[INFO] Profiling batch inference and hardware resource utilization...")
    i_monitor = ResourceMonitor(interval=MONITOR_INTERVAL)
    i_monitor.start()
    run_times = []
    
    # 5 تکرار تست برای سنجش پایداری
    for run in range(NUM_INFERENCE_RUNS):
        t0 = time.time()
        y_prob = model.predict(X_test_lstm, batch_size=CHUNK_SIZE, verbose=0).ravel()
        run_times.append(time.time() - t0)
        print(f"  - Inference Run {run+1}: {run_times[-1]:.3f} s")
        
    i_monitor.stop()
    i_res = i_monitor.summary()

    y_pred = (y_prob >= THRESHOLD).astype(np.int8)

    # 7. Pure Single-Sample Latency Profiling (Batch Size = 1 on 5,000 samples)
    print(f"[INFO] Measuring isolated Pure Single-Sample latency (Batch Size = 1, N={LATENCY_SAMPLE_LIMIT:,})...")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(LATENCY_SAMPLE_LIMIT, len(X_test_scaled))
    indices = rng.choice(len(X_test_scaled), size=sample_size, replace=False)
    test_samples_for_latency = X_test_scaled[indices]

    # بهینه‌سازی گراف برای ثبت دقیق‌ترین تاخیر خالص رابط
    @tf.function
    def infer_step(x_tensor):
        return model(x_tensor, training=False)

    # فاز Warm-up (۵۰۰ سمپل) جهت پایداری کش و گراف محاسباتی
    warmup_n = min(LATENCY_WARMUP_SAMPLES, len(test_samples_for_latency))
    for i in range(warmup_n):
        dummy_in = tf.convert_to_tensor(test_samples_for_latency[i].reshape((1, 4, 1)), dtype=tf.float32)
        _ = infer_step(dummy_in)

    # اندازه‌گیری تاخیر تک‌نمونه
    latencies = np.empty(sample_size, dtype=np.float64)
    for i in range(sample_size):
        x_single = tf.convert_to_tensor(test_samples_for_latency[i].reshape((1, 4, 1)), dtype=tf.float32)
        t0 = time.perf_counter_ns()
        _ = infer_step(x_single)
        t1 = time.perf_counter_ns()
        latencies[i] = (t1 - t0) / 1_000_000.0  # تبدیل به میلی‌ثانیه

    # 8. Calculate Comprehensive Performance Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
    mcc = float(matthews_corrcoef(y_test, y_pred))
    roc_auc_val = float(roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    final_metrics = {
        "model_name": "LSTM Lightweight",
        "training_time_seconds": float(train_time),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f2_score": f2,
        "mcc": mcc,
        "roc_auc": roc_auc_val,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "latency_stats_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99))
        },
        "resource_usage": {"inference": i_res}
    }

    # Save Metrics and Reports
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4, ensure_ascii=False)

    report_text = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], digits=4, zero_division=0)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(RESOURCE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "timestamps_sec": i_monitor.timestamps,
            "cpu_percent": i_monitor.cpu_percents,
            "memory_mb": i_monitor.memory_mb,
            "latencies_ms": latencies.tolist()
        }, f, indent=4, ensure_ascii=False)

    # 9. Generate Standardized Thesis Visualizations
    print("[INFO] Generating thesis-ready standardized plots...")
    save_confusion_matrix(cm)
    save_normalized_confusion_matrix(cm)
    save_metrics_bar_chart(final_metrics)
    save_per_class_metrics(y_test, y_pred)
    save_latency_visualizations(latencies, run_times)
    save_resource_usage_plots({"timestamps": i_monitor.timestamps, "cpu_percents": i_monitor.cpu_percents, "memory_mb": i_monitor.memory_mb})
    save_resource_summary_plot(i_res)
    save_roc_curve(y_test, y_prob)
    save_pr_curve(y_test, y_prob)

    print(f"\n[SUMMARY RESULTS - LSTM]")
    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"F1-Score:  {f1:.4f} | F2-Score:  {f2:.4f} | MCC:    {mcc:.4f} | ROC-AUC: {roc_auc_val:.4f}")
    print(f"Latency:   Mean = {np.mean(latencies):.4f} ms | P95 = {np.percentile(latencies, 95):.4f} ms | P99 = {np.percentile(latencies, 99):.4f} ms")
    print(f"[SUCCESS] LSTM Pipeline executed successfully. Artifacts saved in '{RESULTS_DIR}'.")

if __name__ == "__main__":
    main()
