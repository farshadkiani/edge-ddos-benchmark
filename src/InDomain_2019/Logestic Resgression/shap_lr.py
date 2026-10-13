# -*- coding: utf-8 -*-
"""
SHAP Analysis for Logistic Regression (CICDDoS2019 - Hybrid Model)
Dataset: DrDoS_DNS.csv
@author: farshad
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================================================
# Paths and constants
# =========================================================
DATASET_PATH = "DrDoS_DNS.csv"
RESULTS_DIR = "results"
MODEL_NAME = "logistic_regression_attack_detector_sim.pkl"

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

# پارامترهای سرعت محاسبات
BACKGROUND_SIZE = 1000
EXPLAIN_SIZE = 5000

# =========================================================
# 1. Load dataset (Handling CSV Header Stripping)
# =========================================================
print("[INFO] Loading dataset for SHAP analysis...")

# خواندن هدر برای مپ کردن نام‌های دارای فاصله
header_df = pd.read_csv(DATASET_PATH, nrows=0)
col_mapping = {col.strip(): col for col in header_df.columns}
actual_cols_to_load = [col_mapping[col] for col in RAW_REQUIRED_COLUMNS]

df = pd.read_csv(DATASET_PATH, usecols=actual_cols_to_load, low_memory=True)
df.columns = df.columns.str.strip()
df = df[RAW_REQUIRED_COLUMNS].copy()

# =========================================================
# 2. Feature engineering
# =========================================================
print("[INFO] Engineering the 4 target features...")

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

X = df[FEATURE_COLUMNS].copy()
y = df["binary_label"].copy()

# =========================================================
# 3. Stratified Split (To match evaluation test set)
# =========================================================
# ما فقط به X_test نیاز داریم که مدل روی آن ارزیابی شده است
_, X_test, _, y_test = train_test_split(
    X, y, 
    test_size=TEST_RATIO, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"[INFO] Test samples for analysis: {len(X_test)}")

# =========================================================
# 4. Load trained Pipeline
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print(f"[INFO] Loading LR Hybrid Pipeline: {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

scaler = pipeline.named_steps["scaler"]
lr_model = pipeline.named_steps["classifier"]

# =========================================================
# 5. Transform and Sample
# =========================================================
# تبدیل داده‌های تست به مقیاس استاندارد با استفاده از اسکیلر مدل
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test.values.astype(np.float32)),
    columns=FEATURE_COLUMNS,
    index=X_test.index
)

# انتخاب نمونه برای سرعت (اختیاری)
X_explain_scaled = (
    X_test_scaled.sample(n=EXPLAIN_SIZE, random_state=RANDOM_STATE)
    if len(X_test_scaled) > EXPLAIN_SIZE else X_test_scaled
)

# ایجاد Background از داده‌های تست یا کل داده (برای LinearExplainer)
X_background = X_test_scaled.sample(n=BACKGROUND_SIZE, random_state=RANDOM_STATE)

# =========================================================
# 6. SHAP Calculation
# =========================================================
print("[INFO] Calculating SHAP values (LinearExplainer)...")
explainer = shap.LinearExplainer(lr_model, X_background)
shap_values_raw = explainer(X_explain_scaled)

# نگاشت مقادیر به ویژگی‌های خام برای نمایش بهتر در نمودار
shap_values = shap.Explanation(
    values=shap_values_raw.values,
    base_values=shap_values_raw.base_values,
    data=X_test.loc[X_explain_scaled.index].values,
    feature_names=FEATURE_COLUMNS
)

# =========================================================
# 7. Plotting
# =========================================================
print("[INFO] Saving plots...")

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test.loc[X_explain_scaled.index], show=False)
plt.title("SHAP Summary - LR Hybrid (CICDDoS2019)", fontsize=14, pad=20)
plt.savefig(os.path.join(RESULTS_DIR, "lr_2019_shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# Bar Plot
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values, show=False)
plt.title("Feature Importance - LR Hybrid (Global Mean |SHAP|)", fontsize=14, pad=20)
plt.savefig(os.path.join(RESULTS_DIR, "lr_2019_shap_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] SHAP Analysis for 2019 finished. Results in '{RESULTS_DIR}'")
