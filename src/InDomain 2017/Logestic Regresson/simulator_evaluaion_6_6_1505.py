# =========================================================
# Logistic Regression Benchmark
# CICIDS2017 In-Domain DDoS Dataset
# Chronological 80/20 Evaluation Protocol
# =========================================================

import json
import os
import threading
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# =========================================================
# Warning Control
# =========================================================
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

# =========================================================
# Matplotlib Styling
# =========================================================
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Liberation Serif",
        ],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# مسیر فایل دیتاست و مدل را در صورت نیاز تغییر دهید
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
MODEL_PATH = "results/logistic_regression_attack_detector.pkl"
OUTPUT_DIR = "results"

TEST_RATIO = 0.20
WARMUP_SAMPLES = 100
RESOURCE_SAMPLE_INTERVAL = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Required Columns
# =========================================================
RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

# ترتیب ویژگی‌ها باید دقیقاً با زمان آموزش مدل یکسان باشد.
FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

TARGET_COL = "Label"

# =========================================================
# Resource Monitoring Storage
# =========================================================
resource_cpu = []
resource_mem = []
resource_timestamps = []

monitor_stop = threading.Event()

# =========================================================
# Resource Monitoring Function
# =========================================================
def monitor_resources(interval=0.1):
    """Monitor CPU utilization and process RSS memory."""
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()

    # Initialize CPU measurement
    psutil.cpu_percent(interval=None)

    while not monitor_stop.is_set():
        try:
            elapsed_time = time.perf_counter() - start_time
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_mb = process.memory_info().rss / (1024 ** 2)

            resource_timestamps.append(float(elapsed_time))
            resource_cpu.append(float(cpu_percent))
            resource_mem.append(float(memory_mb))
        except Exception:
            # خطای مانیتورینگ نباید بنچمارک را متوقف کند.
            pass

        time.sleep(interval)

# =========================================================
# Utility Functions
# =========================================================
def save_current_figure(filename):
    """Save and close the current Matplotlib figure."""
    figure_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close()
    return figure_path

def convert_distribution_to_int_dict(distribution):
    """Convert class distribution to a JSON-safe dictionary."""
    return {str(key): int(value) for key, value in distribution.items()}

def get_model_classes(model):
    """Extract classes_ from a direct estimator or Pipeline."""
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    if hasattr(model, "named_steps"):
        for estimator in reversed(list(model.named_steps.values())):
            classes = getattr(estimator, "classes_", None)
            if classes is not None:
                return np.asarray(classes)

    return None

def get_positive_class_probability(model, x_single, prediction):
    """
    Return probability of the positive class.
    Positive class: 1 = DDoS / non-BENIGN
    """
    if not hasattr(model, "predict_proba"):
        return float(prediction)

    probabilities = model.predict_proba(x_single)
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        return float(prediction)

    classes = get_model_classes(model)

    # One-class model
    if probabilities.shape[1] == 1:
        if classes is not None and len(classes) == 1:
            class_value = str(classes[0]).strip().upper()
            positive_names = {"1", "DDOS", "ATTACK", "NON_BENIGN"}
            if class_value in positive_names:
                return float(probabilities[0, 0])
            return float(1.0 - probabilities[0, 0])
        return float(prediction)

    # Standard binary model
    if classes is not None:
        positive_indices = np.where(classes == 1)[0]
        if len(positive_indices) > 0:
            positive_index = int(positive_indices[0])
            return float(probabilities[0, positive_index])

        normalized_classes = [str(item).strip().upper() for item in classes]
        positive_names = {"1", "DDOS", "ATTACK", "NON_BENIGN"}
        for index, class_name in enumerate(normalized_classes):
            if class_name in positive_names:
                return float(probabilities[0, index])

    # Fallback for standard binary classifiers
    return float(probabilities[0, 1])

def find_logistic_regression_estimator(model):
    """Find the Logistic Regression estimator inside a direct estimator or Pipeline."""
    if hasattr(model, "coef_"):
        return model

    if hasattr(model, "named_steps"):
        estimators = list(model.named_steps.values())
        for estimator in reversed(estimators):
            if hasattr(estimator, "coef_"):
                return estimator
        if len(estimators) > 0:
            return estimators[-1]

    return model

# =========================================================
# Validate Input Files
# =========================================================
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Dataset file was not found:\n{DATA_PATH}")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file was not found:\n{MODEL_PATH}")

# =========================================================
# Load Model
# =========================================================
print("[INFO] Loading Logistic Regression model...")
model = joblib.load(MODEL_PATH)
print(f"[INFO] Loaded model type: {type(model).__name__}")

# =========================================================
# Load Dataset
# =========================================================
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[INFO] Raw dataset shape: {df.shape}")

# =========================================================
# Clean Column Names
# =========================================================
df.columns = df.columns.astype(str).str.strip()

# =========================================================
# Validate Required Columns
# =========================================================
missing_columns = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"Required columns are missing:\n{missing_columns}\n\n"
        f"Available columns are:\n{list(df.columns)}"
    )

# فقط ستون‌های موردنیاز حفظ می‌شوند.
df = df[RAW_REQUIRED_COLUMNS].copy()

# =========================================================
# Feature Extraction
# =========================================================
df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0.0)
    + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0.0)
)
df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

# =========================================================
# Normalize Labels
# =========================================================
df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()

# =========================================================
# Replace Invalid Values
# =========================================================
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================================================
# Remove Invalid Rows
# =========================================================
df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL], inplace=True)
df.reset_index(drop=True, inplace=True)

if len(df) == 0:
    raise ValueError("No valid rows remain after preprocessing.")

print("\n[INFO] Normalized label distribution:")
print(df[TARGET_COL].value_counts(dropna=False))

# =========================================================
# Binary Label Mapping
# =========================================================
# BENIGN     -> 0
# NON-BENIGN -> 1
# =========================================================
df["binary_label"] = (df[TARGET_COL] != "BENIGN").astype(np.int32)

# =========================================================
# Chronological 80/20 Split
# =========================================================
total_samples = int(len(df))

if total_samples < 2:
    raise ValueError("At least two valid samples are required for chronological splitting.")

split_idx = int(total_samples * (1.0 - TEST_RATIO))
split_idx = max(1, split_idx)
split_idx = min(total_samples - 1, split_idx)

df_train = df.iloc[:split_idx].copy()
df_test = df.iloc[split_idx:].copy()

if len(df_train) == 0:
    raise ValueError("The training/development portion is empty.")

if len(df_test) == 0:
    raise ValueError("The chronological test portion is empty.")

# =========================================================
# Prepare Test Features and Labels
# =========================================================
X = df_test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
y = df_test["binary_label"].to_numpy(dtype=np.int32)

if X.ndim != 2:
    raise ValueError("Feature matrix X must be two-dimensional.")

if X.shape[1] != len(FEATURE_COLUMNS):
    raise ValueError("The number of input features does not match FEATURE_COLUMNS.")

if len(X) != len(y):
    raise ValueError("X and y have different numbers of samples.")

# =========================================================
# Print Split Information
# =========================================================
print("\n[INFO] Chronological split completed.")
print(f"[INFO] Total valid samples: {total_samples}")
print(f"[INFO] Training/development samples: {len(df_train)}")
print(f"[INFO] Chronological test samples: {len(df_test)}")
print(f"[INFO] Training ratio: {len(df_train) / total_samples:.4f}")
print(f"[INFO] Test ratio: {len(df_test) / total_samples:.4f}")

# =========================================================
# Class Distributions
# =========================================================
train_class_distribution = df_train["binary_label"].value_counts().sort_index().to_dict()
test_class_distribution = df_test["binary_label"].value_counts().sort_index().to_dict()

print(f"[INFO] Training class distribution: {train_class_distribution}")
print(f"[INFO] Test class distribution: {test_class_distribution}")

if len(np.unique(y)) < 2:
    print("[WARNING] Test set contains only one class. ROC-AUC and ROC curve may be unavailable.")

# =========================================================
# Warm-up Inference
# =========================================================
print("[INFO] Running warm-up inference...")
warmup_count = min(WARMUP_SAMPLES, len(X))

for index in range(warmup_count):
    x_single = X[index:index + 1]
    model.predict(x_single)

# =========================================================
# Start Resource Monitoring
# =========================================================
print("[INFO] Starting single-sample inference benchmark...")
monitor_stop.clear()

monitor_thread = threading.Thread(
    target=monitor_resources,
    kwargs={"interval": RESOURCE_SAMPLE_INTERVAL},
    daemon=True,
)
monitor_thread.start()

# =========================================================
# Pure Single-Sample Inference Benchmark
# =========================================================
all_latencies_ms = []
all_predictions = []
all_probabilities = []

benchmark_start = time.perf_counter()

try:
    for index in range(len(X)):
        x_single = X[index:index + 1]
        inference_start = time.perf_counter()

        prediction_array = model.predict(x_single)
        prediction = int(np.asarray(prediction_array).reshape(-1)[0])

        probability = get_positive_class_probability(
            model=model,
            x_single=x_single,
            prediction=prediction,
        )

        inference_end = time.perf_counter()
        latency_ms = (inference_end - inference_start) * 1000.0

        all_latencies_ms.append(float(latency_ms))
        all_predictions.append(prediction)
        all_probabilities.append(float(probability))

finally:
    benchmark_end = time.perf_counter()
    monitor_stop.set()
    monitor_thread.join(timeout=2.0)

# =========================================================
# Convert Outputs to NumPy Arrays
# =========================================================
latencies = np.asarray(all_latencies_ms, dtype=np.float64)
y_pred = np.asarray(all_predictions, dtype=np.int32)
y_prob = np.asarray(all_probabilities, dtype=np.float64)

# =========================================================
# Validate Benchmark Outputs
# =========================================================
if len(latencies) != len(y):
    raise RuntimeError("Latency count does not match label count.")

if len(y_pred) != len(y):
    raise RuntimeError("Prediction count does not match label count.")

if len(y_prob) != len(y):
    raise RuntimeError("Probability count does not match label count.")

# =========================================================
# Classification Metrics
# =========================================================
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
mcc = matthews_corrcoef(y, y_pred)

try:
    roc_auc = roc_auc_score(y, y_prob)
except ValueError:
    roc_auc = None

cm = confusion_matrix(y, y_pred, labels=[0, 1])

report = classification_report(
    y,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    zero_division=0,
)

print("\n[INFO] Classification report:")
print(report)

# =========================================================
# Confusion Matrix Values
# =========================================================
tn, fp, fn, tp = cm.ravel()
print("[INFO] Confusion matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# =========================================================
# Latency Metrics
# =========================================================
if len(latencies) == 0:
    raise RuntimeError("No latency values were recorded.")

mean_latency_ms = float(np.mean(latencies))
median_latency_ms = float(np.median(latencies))
p50_latency_ms = float(np.percentile(latencies, 50))
p90_latency_ms = float(np.percentile(latencies, 90))
p95_latency_ms = float(np.percentile(latencies, 95))
p99_latency_ms = float(np.percentile(latencies, 99))
min_latency_ms = float(np.min(latencies))
max_latency_ms = float(np.max(latencies))
std_latency_ms = float(np.std(latencies))

total_inference_time_seconds = float(np.sum(latencies) / 1000.0)

if total_inference_time_seconds > 0:
    throughput = float(len(latencies) / total_inference_time_seconds)
else:
    throughput = None

# =========================================================
# Resource Metrics
# =================================================auc = None

cm = confusion_matrix(y, y_pred, labels=[0, 1])

report = classification_report(
    y,
    y_pred,
    labels=[0, 1],
    target_names=["BENIGN", "DDoS"],
    zero_division=0,
)

print("\n[INFO] Classification report:")
print(report)

# =========================================================
# Confusion Matrix Values
# =========================================================
tn, fp, fn, tp = cm.ravel()
print("[INFO] Confusion matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# =========================================================
# Latency Metrics
# =========================================================
if len(latencies) == 0:
    raise RuntimeError("No latency values were recorded.")

mean_latency_ms = float(np.mean(latencies))
median_latency_ms = float(np.median(latencies))
p50_latency_ms = float(np.percentile(latencies, 50))
p90_latency_ms = float(np.percentile(latencies, 90))
p95_latency_ms = float(np.percentile(latencies, 95))
p99_latency_ms = float(np.percentile(latencies, 99))
min_latency_ms = float(np.min(latencies))
max_latency_ms = float(np.max(latencies))
std_latency_ms = float(np.std(latencies))

total_inference_time_seconds = float(np.sum(latencies) / 1000.0)

if total_inference_time_seconds > 0:
    throughput = float(len(latencies) / total_inference_time_seconds)
else:
    throughput = None

# =========================================================
# Resource Metrics
# =================================================
metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "MCC",
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1,
    mcc,
]

if roc_auc is not None:
    metric_names.append("ROC-AUC")
    metric_values.append(roc_auc)

bar_colors = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
]

plt.figure(figsize=(6.2, 3.8))
bars = plt.bar(
    metric_names,
    metric_values,
    color=bar_colors[:len(metric_names)],
    width=0.55,
    edgecolor="black",
    linewidth=0.7,
)

plt.axhline(0, color="gray", linewidth=0.8)
plt.ylim(-1.05 if mcc < 0 else 0.0, 1.08)
plt.title("Classification Performance Metrics (LR)")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score [-]")

for bar, value in zip(bars, metric_values):
    if value >= 0:
        vertical_alignment = "bottom"
        offset = 0.02
    else:
        vertical_alignment = "top"
        offset = -0.05

    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        value + offset,
        f"{value:.4f}",
        ha="center",
        va=vertical_alignment,
        fontsize=9,
    )

save_current_figure("fig_metrics_bar_chart_lr.png")

# =========================================================
# Figure 2: Confusion Matrix
# =========================================================
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

plt.title("Confusion Matrix (LR - In-Domain 2017)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
save_current_figure("fig_confusion_matrix_lr.png")

# =========================================================
# Figure 3: ROC Curve
# =========================================================
if roc_auc is not None and len(np.unique(y)) >= 2:
    fpr, tpr, _ = roc_curve(y, y_prob)

    plt.figure(figsize=(4.5, 3.8))
    plt.plot(
        fpr,
        tpr,
        color="#1f77b4",
        linewidth=1.5,
        label=f"LR (AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random Chance",
    )

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.05)
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_current_figure("fig_roc_curve_lr.png")
else:
    print("[WARNING] ROC curve was not generated because ROC-AUC is unavailable.")

# =========================================================
# Figure 4: Latency Percentiles
# =========================================================
percentile_labels = ["P50", "P90", "P95", "P99"]
percentile_values = [
    p50_latency_ms,
    p90_latency_ms,
    p95_latency_ms,
    p99_latency_ms,
]

plt.figure(figsize=(5.0, 3.8))
bars = plt.bar(
    percentile_labels,
    percentile_values,
    color="#4C72B0",
    width=0.55,
    edgecolor="black",
    linewidth=0.7,
)

plt.title("Inference Latency Percentiles (Batch Size = 1)")
plt.xlabel("Latency Metric")
plt.ylabel("Latency [ms]")

max_percentile_latency = max(percentile_values)
latency_offset = 0.02 if max_percentile_latency == 0 else 0.005 * max_percentile_latency

for bar, value in zip(bars, percentile_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        value + latency_offset,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

save_current_figure("fig_latency_percentiles_lr.png")

# =========================================================
# Figure 5: Latency Distribution
# =========================================================
plt.figure(figsize=(5.0, 3.8))
sns.histplot(
    latencies,
    bins=35,
    kde=True,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.5,
)

plt.axvline(
    mean_latency_ms,
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"Mean = {mean_latency_ms:.3f} ms",
)
plt.axvline(
    p95_latency_ms,
    color="green",
    linestyle=":",
    linewidth=1.2,
    label=f"P95 = {p95_latency_ms:.3f} ms",
)

plt.title("Per-Sample Latency Distribution")
plt.xlabel("Inference Latency [ms]")
plt.ylabel("Sample Count")
plt.legend(loc="upper right")
save_current_figure("fig_latency_distribution_lr.png")

# =========================================================
# Figure 6: Resource Usage
# =========================================================
if len(resource_timestamps) > 0 and len(resource_cpu) > 0 and len(resource_mem) > 0:
    common_length = min(
        len(resource_timestamps),
        len(resource_cpu),
        len(resource_mem),
    )

    timestamps_plot = resource_timestamps[:common_length]
    cpu_plot = resource_cpu[:common_length]
    memory_plot = resource_mem[:common_length]

    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))

    line1, = ax1.plot(
        timestamps_plot,
        cpu_plot,
        color="#1f77b4",
        linewidth=1.2,
        label="CPU Utilization [%]",
    )

    ax1.set_xlabel("Elapsed Time [s]")
    ax1.set_ylabel("CPU Utilization [%]", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    line2, = ax2.plot(
        timestamps_plot,
        memory_plot,
        color="#d62728",
        linewidth=1.2,
        label="Memory (RSS) [MB]",
    )

    ax2.set_ylabel("Memory Usage [MB]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("System Resource Utilization Over Time")
    plt.tight_layout()
    save_current_figure("fig_resource_usage_lr.png")
else:
    print("[WARNING] Resource-monitoring data unavailable. Resource plot was not generated.")

# =========================================================
# Figure 7: Logistic Regression Coefficients
# =========================================================
try:
    lr_estimator = find_logistic_regression_estimator(model)

    if hasattr(lr_estimator, "coef_"):
        coefficients = np.asarray(lr_estimator.coef_)

        if coefficients.ndim == 2:
            coefficients = coefficients[0]

        coefficients = np.abs(coefficients)

        if len(coefficients) == len(FEATURE_COLUMNS):
            coefficient_df = pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Weight": coefficients,
                }
            )

            coefficient_df.sort_values(by="Weight", ascending=True, inplace=True)

            plt.figure(figsize=(5.5, 3.8))
            plt.barh(
                coefficient_df["Feature"],
                coefficient_df["Weight"],
                color="#55A868",
                edgecolor="black",
                linewidth=0.7,
            )

            plt.title("Absolute Feature Coefficients (LR)")
            plt.xlabel("Absolute Weight Value")
            plt.ylabel("Traffic Feature")
            save_current_figure("fig_feature_importance_lr.png")
        else:
            print("[WARNING] Number of coefficients does not match the number of configured features.")
    else:
        print("[WARNING] The loaded model does not expose the coef_ attribute.")

except Exception as error:
    print(f"[WARNING] Could not generate coefficient plot: {error}")

# =========================================================
# Export Figure Captions
# =========================================================
caption_path = os.path.join(OUTPUT_DIR, "figure_captions_lr.txt")

with open(caption_path, "w", encoding="utf-8") as file:
    file.write("Standardized Figure Captions (Times New Roman, 10pt, 300 DPI)\n")
    file.write("=" * 65 + "\n\n")
    file.write(
        "Figure 1: Classification performance metrics "
        "(Accuracy, Precision, Recall, F1-Score, MCC, "
        "and ROC-AUC) for Logistic Regression evaluated "
        "on the chronological test set.\n\n"
    )
    file.write(
        "Figure 2: Confusion matrix of the Logistic "
        "Regression classifier evaluated on the final "
        "20% chronological test set of the CICIDS2017 "
        "in-domain DDoS traffic data.\n\n"
    )
    file.write(
        "Figure 3: Receiver Operating Characteristic "
        "(ROC) curve and corresponding AUC metric for "
        "Logistic Regression on the chronological test set.\n\n"
    )
    file.write(
        "Figure 4: Single-sample pure inference latency "
        "percentiles (P50, P90, P95, and P99) for the "
        "Logistic Regression model.\n\n"
    )
    file.write(
        "Figure 5: Histogram and kernel density estimation "
        "(KDE) of per-sample inference latency [ms].\n\n"
    )
    file.write(
        "Figure 6: Time-series profiling of CPU utilization "
        "[%] and Resident Set Size (RSS) memory consumption "
        "[MB].\n\n"
    )
    file.write(
        "Figure 7: Absolute Logistic Regression coefficient "
        "weights assigned to extracted flow-level network "
        "features.\n"
    )

# =========================================================
# Resource Metrics (این بخش در کد شما جا افتاده بود)
# =========================================================
process = psutil.Process(os.getpid())
current_rss_mb = float(process.memory_info().rss / (1024 ** 2))

if len(resource_mem) > 0:
    peak_rss_mb = float(np.max(resource_mem))
    average_memory_mb = float(np.mean(resource_mem))
else:
    peak_rss_mb = current_rss_mb
    average_memory_mb = current_rss_mb

if len(resource_cpu) > 0:
    average_cpu_percent = float(np.mean(resource_cpu))
    peak_cpu_percent = float(np.max(resource_cpu))
else:
    average_cpu_percent = None
    peak_cpu_percent = None
# =========================================================
# Build JSON Results
# =========================================================
results = {
    "model": "Logistic Regression",
    "dataset": DATA_PATH,
    "model_path": MODEL_PATH,
    "features": FEATURE_COLUMNS,
    "data_split": {
        "strategy": "chronological",
        "train_ratio": float(len(df_train) / total_samples),
        "test_ratio": float(len(df_test) / total_samples),
        "shuffle": False,
        "random_sampling": False,
        "split_index": int(split_idx),
        "total_valid_samples": int(total_samples),
        "training_samples": int(len(df_train)),
        "test_samples": int(len(df_test)),
        "train_class_distribution": convert_distribution_to_int_dict(train_class_distribution),
        "test_class_distribution": convert_distribution_to_int_dict(test_class_distribution),
        "label_mapping": {
            "BENIGN": 0,
            "NON_BENIGN": 1,
        },
    },
    "evaluation_protocol": {
        "batch_size": 1,
        "latency_scope": "pure_model_inference",
        "feature_extraction_in_latency": False,
        "preprocessing_in_latency": False,
        "model_inference_in_latency": True,
        "warmup_samples": int(warmup_count),
        "evaluated_samples": int(len(X)),
        "benchmark_wall_time_seconds": float(benchmark_end - benchmark_start),
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
        "min": min_latency_ms,
        "max": max_latency_ms,
        "std": std_latency_ms,
    },
    "resource_metrics": {
        "peak_rss_mb": peak_rss_mb,
        "average_memory_mb": average_memory_mb,
        "average_cpu_percent": average_cpu_percent,
        "peak_cpu_percent": peak_cpu_percent,
    },
    "throughput_samples_per_second": throughput,
    "confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    },
}

# =========================================================
# Save JSON Results
# =========================================================
json_path = os.path.join(OUTPUT_DIR, "benchmark_lr_indomain2017.json")

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(f"[INFO] Benchmark results saved to: {json_path}")

# =========================================================
# Final Output
# =========================================================
print("\n[INFO] Benchmark completed successfully.")
print(f"[INFO] JSON results saved to:\n{json_path}")
print(f"[INFO] Figures and captions saved in:\n{OUTPUT_DIR}")
