# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 2026
Modified for Feature Consistency (Q1 Standard)
@author: farshad
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ۱. تعریف مسیرها
dataset_path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
results_path = "results"
model_name = "xgboost_attack_detector.pkl"

# ایجاد پوشه نتایج در صورت عدم وجود
if not os.path.exists(results_path):
    os.makedirs(results_path)

# ۲. بارگذاری و مهندسی ویژگی‌ها (دقیقاً مشابه Training)
print("[INFO] Loading dataset for SHAP analysis...")

# ستون‌های خام مورد نیاز برای ساخت ۴ ویژگی نهایی
RAW_COLUMNS = [
    "Total Fwd Packets", 
    "Total Backward Packets", 
    "Average Packet Size", 
    "Flow Packets/s", 
    "Flow Duration"
]

# خواندن داده‌ها (استفاده از chunksize یا مشخص کردن ستون‌ها برای سرعت بیشتر)
df = pd.read_csv(dataset_path, low_memory=False)
df.columns = df.columns.str.strip()

print("[INFO] Engineering the 4 target features...")
# ساخت ویژگی‌های مهندسی‌شده
df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0) + 
    pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
)
df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

# انتخاب فقط ۴ ویژگی نهایی که مدل روی آن‌ها آموزش دیده است
FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]
df_final = df[FEATURE_COLUMNS].copy()

# پاکسازی مقادیر نامعتبر
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final = df_final.dropna().reset_index(drop=True)

# ۳. تقسیم داده‌ها (۸۰-۲۰ برای تطابق با مدل)
print(f"[INFO] Dataset shape: {df_final.shape}")
_, X_test = train_test_split(df_final, test_size=0.2, random_state=42)

# ۴. لود کردن مدل آموزش‌دیده
model_path = os.path.join(results_path, model_name)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please run training script first.")

print(f"[INFO] Loading XGBoost model from: {model_path}")
model = joblib.load(model_path)

# ۵. اجرای محاسبات SHAP
print("[INFO] Calculating SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
# استفاده از ۱۰۰۰ نمونه از تست برای سرعت بیشتر (اختیاری)
# X_test_sample = X_test.sample(1000, random_state=42) if len(X_test) > 1000 else X_test
shap_values = explainer(X_test)

# ۶. ترسیم و ذخیره نمودارها با کیفیت بالا (DPI 300)
print("[INFO] Generating and saving plots...")

# نمودار ۱: Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot - XGBoost (4 Features)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "xgb_shap_summary.png"), dpi=300, bbox_inches="tight")
plt.close()

# نمودار ۲: Feature Importance (Bar Plot)
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values, show=False)
plt.title("XGBoost Feature Importance (Global Mean |SHAP|)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "xgb_shap_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"[SUCCESS] SHAP Analysis finished. Plots saved in '{results_path}' folder.")
