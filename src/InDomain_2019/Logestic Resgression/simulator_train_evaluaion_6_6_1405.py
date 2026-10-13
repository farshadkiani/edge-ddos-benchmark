#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

# =========================================================
# Paths and constants
# =========================================================
FILE_PATH = "DrDoS_DNS.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_RATIO = 0.2
RANDOM_STATE = 42
MONITOR_INTERVAL = 0.1
NUM_INFERENCE_RUNS = 5
LATENCY_SAMPLE_LIMIT = 5000
WARMUP_SAMPLES = 500

RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

# Exactly four engineered features
FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

MODEL_PATH = os.path.join(RESULTS_DIR, "logistic_regression_attack_detector_sim.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_lr_sim.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report_lr_sim.txt")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix_lr_sim.png")
ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve_lr_sim.png")
METRICS_BAR_PATH = os.path.join(RESULTS_DIR, "metrics_bar_chart_lr_sim.png")
PRED_DIST_PATH = os.path.join(RESULTS_DIR, "prediction_distribution_lr_sim.png")
RESOURCE_USAGE_PATH = os.path.join(RESULTS_DIR, "resource_usage_lr_sim.png")
CPU_USAGE_PATH = os.path.join(RESULTS_DIR, "cpu_usage_over_time_lr_sim.png")
MEMORY_USAGE_PATH = os.path.join(RESULTS_DIR, "memory_usage_over_time_lr_sim.png")
LATENCY_DIST_PATH = os.path.join(RESULTS_DIR, "latency_distribution_lr_sim.png")
LATENCY_BOXPLOT_PATH = os.path.join(RESULTS_DIR, "latency_boxplot_lr_sim.png")
LATENCY_PERCENTILES_PATH = os.path.join(RESULTS_DIR, "latency_percentiles_lr_sim.png")
LATENCY_STABILITY_PATH = os.path.join(RESULTS_DIR, "latency_stability_over_time_lr_sim.png")
PRED_TIME_RUNS_PATH = os.path.join(RESULTS_DIR, "prediction_time_runs_lr_sim.png")
COEFFICIENTS_PATH = os.path.join(RESULTS_DIR, "feature_coefficients_lr_sim.png")
RESOURCE_JSON_PATH = os.path.join(RESULTS_DIR, "resource_monitor_lr_sim.json")

# Publication plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


# =========================================================
# Resource monitor
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
                memory = self.process.memory_info().rss / (1024 * 1024)

                self.timestamps.append(now)
                self.cpu_percents.append(cpu)
                self.memory_mb.append(memory)
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        if self.running:
            return
        self.running = True
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
            return {
                "avg_cpu_percent": 0.0,
                "max_cpu_percent": 0.0,
                "avg_memory_mb": 0.0,
                "max_memory_mb": 0.0,
            }
        return {
            "avg_cpu_percent": float(np.mean(self.cpu_percents)),
            "max_cpu_percent": float(np.max(self.cpu_percents)),
            "avg_memory_mb": float(np.mean(self.memory_mb)),
            "max_memory_mb": float(np.max(self.memory_mb)),
        }


# =========================================================
# Plot functions
# =========================================================
def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BENIGN", "DDoS"],
        yticklabels=["BENIGN", "DDoS"],
        ax=ax
    )
    ax.set_title("Confusion Matrix - Logistic Regression", weight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def save_roc_curve(y_true, y_prob, roc_auc):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}", linewidth=2, color="#4C72B0")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Logistic Regression", weight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PATH)
    plt.close()


def save_metrics_bar_chart(metrics_dict):
    metric_names = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "F2", "MCC"]
    metric_values = [
        metrics_dict["accuracy"],
        metrics_dict["balanced_accuracy"],
        metrics_dict["precision"],
        metrics_dict["recall"],
        metrics_dict["f1_score"],
        metrics_dict["f2_score"],
        metrics_dict["mcc"],
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#E15759"]
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor="black", linewidth=0.7)

    if min(metric_values) < 0:
        ax.set_ylim(-1.05, 1.15)
    else:
        ax.set_ylim(0, 1.15)

    ax.set_ylabel("Score")
    ax.set_title("Key Evaluation Metrics - Logistic Regression", weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, value in zip(bars, metric_values):
        text_y = value + 0.02 if value >= 0 else value - 0.04
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value:.4f}",
            ha="center",
            va=va,
            fontsize=8.5,
            weight="bold"
        )

    plt.tight_layout()
    plt.savefig(METRICS_BAR_PATH)
    plt.close()


def save_prediction_distribution(y_pred):
    labels = ["BENIGN", "DDoS"]
    counts = [int(np.sum(y_pred == 0)), int(np.sum(y_pred == 1))]
    total = sum(counts)
    percentages = [(count / total) * 100 if total else 0 for count in counts]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    bars = ax.bar(labels, counts, color=["#4C72B0", "#C44E52"], edgecolor="black", linewidth=0.7, width=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution - Logistic Regression", weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, count, percentage in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(counts) * 0.015),
            f"{count:,}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold"
        )

    ax.set_ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig(PRED_DIST_PATH)
    plt.close()


def save_resource_usage_plots(timestamps, cpu_values, memory_values):
    if not timestamps:
        print("[WARNING] No resource samples were collected.")
        return

    # Combined CPU and Memory
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax1.plot(timestamps, cpu_values, label="CPU (%)", color="#55A868", linewidth=1.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CPU Usage (%)", color="#55A868")
    ax1.tick_params(axis="y", labelcolor="#55A868")
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    ax2.plot(timestamps, memory_values, label="Memory (MB)", color="#C44E52", linewidth=1.5)
    ax2.set_ylabel("Memory Usage (MB)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, max(np.max(memory_values) * 1.3, 1150))
    ax2.axhline(1024, color="red", linestyle="--", linewidth=1.0, label="1 GB Budget")

    plt.title("Resource Usage Over Time - Logistic Regression", weight="bold")
    fig.tight_layout()
    plt.savefig(RESOURCE_USAGE_PATH)
    plt.close()

    # CPU profile
    plt.figure(figsize=(7.5, 4), dpi=300)
    plt.plot(timestamps, cpu_values, color="#55A868", linewidth=1.5, marker="o", markersize=2)
    plt.axhline(np.mean(cpu_values), color="darkgreen", linestyle="--", label=f"Mean CPU ({np.mean(cpu_values):.1f}%)")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.ylim(0, 105)
    plt.title("CPU Utilization Profile - Logistic Regression", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(CPU_USAGE_PATH)
    plt.close()

    # Memory profile
    plt.figure(figsize=(7.5, 4), dpi=300)
    plt.plot(timestamps, memory_values, color="#C44E52", linewidth=1.5, marker="s", markersize=2)
    plt.axhline(1024, color="red", linestyle="--", linewidth=1.2, label="1 GB Edge RAM Budget")
    plt.axhline(np.mean(memory_values), color="darkred", linestyle=":", label=f"Mean RSS ({np.mean(memory_values):.1f} MB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (MB)")
    plt.ylim(0, max(np.max(memory_values) * 1.3, 1150))
    plt.title("Memory Footprint Profile - Logistic Regression", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(MEMORY_USAGE_PATH)
    plt.close()


def save_latency_plots(latencies_ms, run_prediction_times_sec, pure_stats):
    # 1. Latency distribution
    plt.figure(figsize=(7.5, 4.2), dpi=300)
    sns.histplot(latencies_ms, bins=40, kde=True, color="#4C72B0")
    plt.axvline(pure_stats["mean_latency_ms"], color="red", linestyle="--", label=f"Mean: {pure_stats['mean_latency_ms']:.4f} ms")
    plt.xlabel("Pure Model Interface Latency (ms / sample)")
    plt.ylabel("Frequency")
    plt.title("Pure Single-Sample Latency Distribution (Batch Size = 1) - LR", weight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LATENCY_DIST_PATH)
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(5.5, 4.2), dpi=300)
    sns.boxplot(y=latencies_ms, color="#55A868")
    plt.ylabel("Pure Latency (ms / sample)")
    plt.title("Inference Latency Dispersion (Boxplot) - LR", weight="bold")
    plt.tight_layout()
    plt.savefig(LATENCY_BOXPLOT_PATH)
    plt.close()

    # 3. Percentiles
    labels = ["Mean", "P50 (Median)", "P90", "P95", "P99 (Tail)"]
    vals = [
        pure_stats["mean_latency_ms"],
        pure_stats["p50_latency_ms"],
        pure_stats["p90_latency_ms"],
        pure_stats["p95_latency_ms"],
        pure_stats["p99_latency_ms"]
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    bars = ax.bar(labels, vals, color="#8172B3", edgecolor="black", linewidth=0.7, width=0.5)
    ax.set_ylabel("Pure Inference Latency (ms / sample)")
    ax.set_title("Pure Single-Sample Latency Percentiles (Batch Size = 1) - LR", weight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (max(vals) * 0.02),
            f"{val:.4f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold"
        )

    plt.tight_layout()
    plt.savefig(LATENCY_PERCENTILES_PATH)
    plt.close()

    # 4. Latency stability over samples
    plt.figure(figsize=(8, 4.2), dpi=300)
    plt.plot(np.arange(1, len(latencies_ms) + 1), latencies_ms, linewidth=0.8, color="#C44E52", alpha=0.85)
    plt.axhline(pure_stats["mean_latency_ms"], color="black", linestyle="--", label=f"Mean: {pure_stats['mean_latency_ms']:.4f} ms")
    plt.xlabel("Sample Invocation Index")
    plt.ylabel("Pure Latency (ms)")
    plt.title("Latency Stability Over Successive Invocations - LR", weight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(LATENCY_STABILITY_PATH)
    plt.close()

    # 5. Batch prediction time
    plt.figure(figsize=(6.5, 4.2), dpi=300)
    plt.plot(np.arange(1, len(run_prediction_times_sec) + 1), run_prediction_times_sec, marker="o", linewidth=1.5, color="#1F77B4")
    plt.xlabel("Simulation Run Sequence")
    plt.ylabel("Batch Prediction Time (s)")
    plt.title("Full Test-Set Batch Inference Time Across Repeated Runs - LR", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PRED_TIME_RUNS_PATH)
    plt.close()


def save_feature_coefficients(model):
    try:
        if hasattr(model, "named_steps"):
            lr_classifier = model.named_steps["classifier"]
        else:
            lr_classifier = model

        if not hasattr(lr_classifier, "coef_"):
            print("[WARNING] Logistic Regression coefficients are not available.")
            return

        coefficients = lr_classifier.coef_[0]
        coefficient_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Coefficient": coefficients,
            "Absolute_Coefficient": np.abs(coefficients),
        }).sort_values(by="Absolute_Coefficient", ascending=False)

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        colors = ["#C44E52" if val < 0 else "#4C72B0" for val in coefficient_df["Coefficient"]]
        bars = ax.barh(coefficient_df["Feature"], coefficient_df["Coefficient"], color=colors, edgecolor="black", linewidth=0.7)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Standardized Logistic Regression Coefficient")
        ax.set_ylabel("Engineered Feature")
        ax.set_title("Logistic Regression Feature Coefficients (Effect & Direction)", weight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.invert_yaxis()

        for bar, val in zip(bars, coefficient_df["Coefficient"]):
            offset = 0.05 if val >= 0 else -0.05
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2.0, f"{val:+.4f}", va="center", ha=ha, fontsize=8.5, weight="bold")

        plt.tight_layout()
        plt.savefig(COEFFICIENTS_PATH)
        plt.close()
        print(f"[INFO] Feature coefficient plot saved to: {COEFFICIENTS_PATH}")
    except Exception as error:
        print(f"[WARNING] Could not save feature coefficients: {error}")


# =========================================================
# Pure Model Interface Latency Profiling
# =========================================================
def benchmark_pure_latency(classifier, X_test_scaled, limit=LATENCY_SAMPLE_LIMIT, warmup=WARMUP_SAMPLES):
    print(f"[INFO] Measuring Pure Model Interface Latency (Batch Size = 1, N={limit:,}, Warm-up={warmup})...")

    # 1. Warm-up Phase
    warmup_subset = X_test_scaled[:min(warmup, len(X_test_scaled))]
    for i in range(len(warmup_subset)):
        _ = classifier.predict_proba(warmup_subset[i:i+1])

    # 2. Pure Inference Timing Phase
    np.random.seed(RANDOM_STATE)
    subset_size = min(limit, len(X_test_scaled))
    indices = np.random.choice(len(X_test_scaled), size=subset_size, replace=False)
    X_benchmark = X_test_scaled[indices]

    latencies_ms = np.empty(subset_size, dtype=np.float64)

    for i in range(subset_size):
        x_single = X_benchmark[i:i+1]

        t0 = time.perf_counter_ns()
        _ = classifier.predict_proba(x_single)
        t1 = time.perf_counter_ns()

        latencies_ms[i] = (t1 - t0) / 1_000_000.0  # Convert ns to ms

    stats = {
        "n_samples_evaluated": int(subset_size),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p90_latency_ms": float(np.percentile(latencies_ms, 90)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "throughput_samples_per_sec": float(1000.0 / np.mean(latencies_ms)) if np.mean(latencies_ms) > 0 else 0.0,
    }

    return latencies_ms, stats


# =========================================================
# Load dataset
# =========================================================
print("[INFO] Loading dataset...")

header_df = pd.read_csv(FILE_PATH, nrows=0)
stripped_cols = [col.strip() for col in header_df.columns]
col_mapping = {col.strip(): col for col in header_df.columns}

missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in stripped_cols]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

actual_cols_to_load = [col_mapping[col] for col in RAW_REQUIRED_COLUMNS]

df = pd.read_csv(FILE_PATH, usecols=actual_cols_to_load, low_memory=True)
df.columns = df.columns.str.strip()
df = df[RAW_REQUIRED_COLUMNS].copy()

# =========================================================
# Feature engineering
# =========================================================
print("[INFO] Engineering features...")

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
df["binary_label"] = (df["Label"] != "BENIGN").astype(np.int32)

X = df[FEATURE_COLUMNS].values.astype(np.float32)
y = df["binary_label"].values.astype(np.int32)

print(f"[INFO] Dataset shape after engineering: {X.shape}")
print(f"[INFO] Target features: {FEATURE_COLUMNS}")
print("[INFO] Original class distribution:")
print(df["binary_label"].value_counts())

# =========================================================
# Stratified train/test split
# =========================================================
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X,
    y,
    test_size=TEST_RATIO,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"[INFO] Raw train samples: {len(X_train_raw)}, Test samples: {len(X_test)}")

# =========================================================
# Hybrid downsampling on train only
# Majority:Minority = 20:1
# =========================================================
train_df = pd.DataFrame(X_train_raw, columns=FEATURE_COLUMNS)
train_df["binary_label"] = y_train_raw

minority_df = train_df[train_df["binary_label"] == 0]
majority_df = train_df[train_df["binary_label"] == 1]

minority_count = len(minority_df)
majority_count = len(majority_df)

if minority_count == 0 or majority_count == 0:
    raise ValueError("Training set must contain both classes.")

majority_target_count = min(minority_count * 20, majority_count)
majority_downsampled = majority_df.sample(n=majority_target_count, random_state=RANDOM_STATE)

hybrid_train_df = pd.concat([minority_df, majority_downsampled], axis=0).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train = hybrid_train_df[FEATURE_COLUMNS].values.astype(np.float32)
y_train = hybrid_train_df["binary_label"].values.astype(np.int32)

print(f"[INFO] Hybrid train samples: {len(X_train)}")
print("[INFO] Hybrid train class distribution:")
print(pd.Series(y_train).value_counts())

# =========================================================
# Class weights for Logistic Regression
# =========================================================
n_zero = np.sum(y_train == 0)
n_one = np.sum(y_train == 1)
n_total = len(y_train)

class_weight = {
    0: float(n_total / (2.0 * n_zero)),
    1: float(n_total / (2.0 * n_one)),
}

# =========================================================
# Build Logistic Regression model pipeline
# =========================================================
scaler = StandardScaler()
classifier = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="saga",
    max_iter=5000,
    class_weight=class_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("scaler", scaler),
        ("classifier", classifier),
    ]
)

# =========================================================
# Train
# =========================================================
print("[INFO] Training Hybrid Logistic Regression model...")
train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start

joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved to: {MODEL_PATH}")

# =========================================================
# Evaluate on untouched imbalanced test set
# =========================================================
print("[INFO] Evaluating model on test set...")
pred_start = time.time()
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
pred_time = time.time() - pred_start

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except ValueError:
    roc_auc = float("nan")

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], zero_division=0)

print(f"\n[RESULT] Accuracy:          {acc:.6f}")
print(f"[RESULT] Balanced Accuracy: {balanced_acc:.6f}")
print(f"[RESULT] Precision:         {precision:.6f}")
print(f"[RESULT] Recall:            {recall:.6f}")
print(f"[RESULT] F1-score:          {f1:.6f}")
print(f"[RESULT] F2-score:          {f2:.6f}")
print(f"[RESULT] Macro F1-score:    {macro_f1:.6f}")
print(f"[RESULT] MCC:               {mcc:.6f}")
print(f"[RESULT] ROC-AUC:           {roc_auc:.6f}" if not np.isnan(roc_auc) else "[RESULT] ROC-AUC:           NaN")
print(f"[RESULT] Train Time:        {train_time:.6f} sec")
print(f"[RESULT] Prediction Time:   {pred_time:.6f} sec")
print("\n[RESULT] Confusion Matrix:\n", cm)
print("\n[RESULT] Classification Report:\n", report)

# =========================================================
# Pure Model Interface Latency Profiling (Batch Size = 1)
# =========================================================
X_test_scaled = scaler.transform(X_test).astype(np.float32)
latencies_ms, pure_latency_stats = benchmark_pure_latency(
    classifier,
    X_test_scaled,
    limit=LATENCY_SAMPLE_LIMIT,
    warmup=WARMUP_SAMPLES
)

print("\n[INFO] Pure Model Interface Latency Metrics (Batch Size = 1):")
print(f"  Mean Latency: {pure_latency_stats['mean_latency_ms']:.4f} ms")
print(f"  P50 Latency:  {pure_latency_stats['p50_latency_ms']:.4f} ms")
print(f"  P90 Latency:  {pure_latency_stats['p90_latency_ms']:.4f} ms")
print(f"  P95 Latency:  {pure_latency_stats['p95_latency_ms']:.4f} ms")
print(f"  P99 Latency:  {pure_latency_stats['p99_latency_ms']:.4f} ms")
print(f"  Throughput:   {pure_latency_stats['throughput_samples_per_sec']:.2f} samples/sec")

# =========================================================
# Resource monitoring during repeated batch inference runs
# =========================================================
print("\n[INFO] Monitoring inference resources...")
monitor = ResourceMonitor(interval=MONITOR_INTERVAL)
run_prediction_times_sec = []

monitor.start()
for _ in range(NUM_INFERENCE_RUNS):
    run_start = time.time()
    _ = model.predict(X_test)
    run_prediction_times_sec.append(time.time() - run_start)
monitor.stop()

resource_summary = monitor.summary()

# =========================================================
# Save metrics and reports
# =========================================================
metrics_dict = {
    "model_name": "Logistic Regression Hybrid",
    "training_time_seconds": float(train_time),
    "inference_total_time_seconds": float(pred_time),
    "pure_model_interface_latency": pure_latency_stats,
    "overall_accuracy": float(acc),
    "balanced_accuracy": float(balanced_acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "f2_score": float(f2),
    "macro_f1_score": float(macro_f1),
    "mcc": float(mcc),
    "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
    "class_0_benign": {
        "precision": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, pos_label=0, zero_division=0)),
    },
    "class_1_ddos": {
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    },
    "confusion_matrix": {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    },
    "resource_usage": {
        "inference": {
            "max_cpu_percent": float(np.max(monitor.cpu_percents)) if monitor.cpu_percents else 0.0,
            "max_memory_mb": float(np.max(monitor.memory_mb)) if monitor.memory_mb else 0.0,
        },
        "summary": resource_summary,
    },
    "feature_columns": FEATURE_COLUMNS,
    "raw_required_columns": RAW_REQUIRED_COLUMNS,
    "file_path": FILE_PATH,
    "model_path": MODEL_PATH,
    "test_ratio": TEST_RATIO,
    "random_state": RANDOM_STATE,
    "latency_sample_limit": LATENCY_SAMPLE_LIMIT,
    "num_inference_runs": NUM_INFERENCE_RUNS,
    "train_sampling_ratio_majority_to_minority": 20,
    "scaling": "StandardScaler",
    "solver": "saga",
    "max_iter": 5000,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

with open(RESOURCE_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "timestamps_sec": monitor.timestamps,
            "cpu_percent": monitor.cpu_percents,
            "memory_mb": monitor.memory_mb,
            "prediction_time_runs_sec": run_prediction_times_sec,
            "pure_latencies_ms": latencies_ms.tolist(),
        },
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"[INFO] Metrics saved to: {METRICS_PATH}")
print(f"[INFO] Classification report saved to: {REPORT_PATH}")
print(f"[INFO] Resource trace saved to: {RESOURCE_JSON_PATH}")

# =========================================================
# Generate plots
# =========================================================
print("\n[INFO] Generating plots...")

save_confusion_matrix(cm)

if not np.isnan(roc_auc):
    save_roc_curve(y_test, y_pred_prob, roc_auc)

save_metrics_bar_chart({
    "accuracy": acc,
    "balanced_accuracy": balanced_acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "f2_score": f2,
    "mcc": mcc,
})

save_prediction_distribution(y_pred)
save_resource_usage_plots(monitor.timestamps, monitor.cpu_percents, monitor.memory_mb)
save_latency_plots(latencies_ms, run_prediction_times_sec, pure_latency_stats)
save_feature_coefficients(model)

print("\n[INFO] All plots successfully generated and saved:")
print(f" - {CONFUSION_MATRIX_PATH}")
print(f" - {ROC_CURVE_PATH}")
print(f" - {METRICS_BAR_PATH}")
print(f" - {PRED_DIST_PATH}")
print(f" - {RESOURCE_USAGE_PATH}")
print(f" - {CPU_USAGE_PATH}")
print(f" - {MEMORY_USAGE_PATH}")
print(f" - {LATENCY_DIST_PATH}")
print(f" - {LATENCY_BOXPLOT_PATH}")
print(f" - {LATENCY_PERCENTILES_PATH}")
print(f" - {LATENCY_STABILITY_PATH}")
print(f" - {PRED_TIME_RUNS_PATH}")
print(f" - {COEFFICIENTS_PATH}")

print("\n" + "=" * 75)
print(" LOGISTIC REGRESSION SIMULATION PIPELINE FINISHED SUCCESSFULLY")
print("=" * 75)
