# -*- coding: utf-8 -*-
"""
SHAP Analysis for Logistic Regression Pipeline
(StandardScaler + LogisticRegression)

Compatible with:
logistic_regression_attack_detector.pkl

@author: farshad
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Paths and constants
# =========================================================
DATASET_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"
MODEL_NAME = "logistic_regression_attack_detector.pkl"

TEST_RATIO = 0.2
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, MODEL_NAME)

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

# برای جلوگیری از کندی و مصرف RAM در SHAP
BACKGROUND_SIZE = 1000
EXPLAIN_SIZE = 5000

# =========================================================
# 1. Load dataset
# =========================================================
print("[INFO] Loading dataset for SHAP analysis...")

df = pd.read_csv(DATASET_PATH, low_memory=False)
df.columns = df.columns.str.strip()

missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

df = df[RAW_REQUIRED_COLUMNS].copy()

# =========================================================
# 2. Feature engineering - exactly same as training script
# =========================================================
print("[INFO] Engineering the 4 target features...")

df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
)

df["Packet_Size"] = pd.to_numeric(
    df["Average Packet Size"], errors="coerce"
)

df["Flow_Rate"] = pd.to_numeric(
    df["Flow Packets/s"], errors="coerce"
)

df["Flow_Duration"] = pd.to_numeric(
    df["Flow Duration"], errors="coerce"
)

# دقیقاً مطابق پاکسازی مدل آموزشی
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=FEATURE_COLUMNS + ["Label"]).reset_index(drop=True)

df["Label"] = df["Label"].astype(str).str.strip().str.upper()
df["binary_label"] = (df["Label"] != "BENIGN").astype(np.int32)

# DataFrame نگه داشته شده تا نام ویژگی‌ها در نمودار SHAP نمایش داده شود
X = df[FEATURE_COLUMNS].copy()
y = df["binary_label"].copy()

print(f"[INFO] Dataset shape after engineering: {X.shape}")
print(f"[INFO] Feature columns: {FEATURE_COLUMNS}")

# =========================================================
# 3. Chronological 80/20 split - same as training script
# =========================================================
split_idx = int(len(df) * (1 - TEST_RATIO))

X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()

y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

print(f"[INFO] Train samples: {len(X_train)}")
print(f"[INFO] Test samples: {len(X_test)}")

# =========================================================
# 4. Load trained Pipeline
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Please run the Logistic Regression training script first."
    )

print(f"[INFO] Loading LR Pipeline from: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

# استخراج اجزای Pipeline ذخیره‌شده
scaler = pipeline.named_steps["scaler"]
lr_model = pipeline.named_steps["classifier"]

print("[INFO] Pipeline components loaded successfully.")
print(f"[INFO] Logistic Regression classes: {lr_model.classes_}")

# =========================================================
# 5. Transform data with the fitted training scaler
# IMPORTANT:
# SHAP LinearExplainer باید روی داده‌ی scale شده اجرا شود،
# زیرا Logistic Regression روی داده scale شده آموزش دیده است.
# =========================================================
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=FEATURE_COLUMNS,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=FEATURE_COLUMNS,
    index=X_test.index,
)

# انتخاب Background Data از train set
X_background_scaled = (
    X_train_scaled.sample(
        n=BACKGROUND_SIZE,
        random_state=RANDOM_STATE
    )
    if len(X_train_scaled) > BACKGROUND_SIZE
    else X_train_scaled.copy()
)

# انتخاب نمونه‌ای از test set برای سرعت بالاتر
X_explain_scaled = (
    X_test_scaled.sample(
        n=EXPLAIN_SIZE,
        random_state=RANDOM_STATE
    )
    if len(X_test_scaled) > EXPLAIN_SIZE
    else X_test_scaled.copy()
)

# داده خام متناظر با همان نمونه‌ها؛ فقط برای نمایش بهتر محورهای SHAP
X_explain_original = X_test.loc[X_explain_scaled.index].copy()

print(f"[INFO] Background samples: {len(X_background_scaled)}")
print(f"[INFO] Samples selected for SHAP explanation: {len(X_explain_scaled)}")

# =========================================================
# 6. Calculate SHAP values using LinearExplainer
# =========================================================
print("[INFO] Calculating SHAP values using LinearExplainer...")

# مدل LR برای کلاس 1 (DDoS / Attack) توضیح داده می‌شود.
# خروجی SHAP در Logistic Regression معمولاً در فضای log-odds است.
explainer = shap.LinearExplainer(
    lr_model,
    X_background_scaled
)

shap_values_scaled = explainer(X_explain_scaled)

# =========================================================
# 7. Replace scaled values with original feature values
# این کار محاسبات SHAP را تغییر نمی‌دهد؛
# فقط باعث می‌شود رنگ‌های Beeswarm بر اساس مقادیر واقعی
# Packet_Count، Packet_Size و ... نمایش داده شوند.
# =========================================================
shap_values = shap.Explanation(
    values=shap_values_scaled.values,
    base_values=shap_values_scaled.base_values,
    data=X_explain_original.values,
    feature_names=FEATURE_COLUMNS,
)

# =========================================================
# 8. Generate SHAP plots
# =========================================================
print("[INFO] Generating SHAP plots...")

# -------------------------
# Plot 1: SHAP Beeswarm Plot
# -------------------------
plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_explain_original,
    show=False,
    plot_size=None
)

plt.title(
    "SHAP Summary Plot - Logistic Regression (DDoS Class)",
    fontsize=14,
    pad=20
)

plt.tight_layout()

summary_path = os.path.join(RESULTS_DIR, "lr_shap_summary.png")

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Plot 2: Global Feature Importance
# -------------------------
plt.figure(figsize=(10, 5))

shap.plots.bar(
    shap_values,
    show=False,
    max_display=len(FEATURE_COLUMNS)
)

plt.title(
    "Logistic Regression Feature Importance (Mean |SHAP|)",
    fontsize=14,
    pad=20
)

plt.tight_layout()

bar_path = os.path.join(RESULTS_DIR, "lr_shap_bar.png")

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 9. Save numeric global SHAP importance
# =========================================================
global_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0),
}).sort_values("Mean_Abs_SHAP", ascending=False)

importance_csv_path = os.path.join(
    RESULTS_DIR,
    "lr_shap_global_importance.csv"
)

global_importance.to_csv(
    importance_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n[RESULT] Global SHAP Feature Importance:")
print(global_importance.to_string(index=False))

print("\n[SUCCESS] Logistic Regression SHAP analysis finished.")
print(f"[INFO] Summary plot saved to: {summary_path}")
print(f"[INFO] Bar plot saved to: {bar_path}")
print(f"[INFO] Numeric importance saved to: {importance_csv_path}")
