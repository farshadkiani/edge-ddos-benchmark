# -*- coding: utf-8 -*-
"""
تطبیق ویژگی‌های SHAP با مدل‌های سبک (Lightweight Features)
Target: 4 Standard Features for Thesis Consistency
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ۱. تنظیمات مسیر
dataset_path = "DrDoS_DNS.csv"
results_path = "results"

# ستون‌های خام مورد نیاز برای ساخت ۴ ویژگی اصلی
RAW_COLUMNS = [
    ' Total Fwd Packets', ' Total Backward Packets', 
    ' Average Packet Size', ' Flow Packets/s', ' Flow Duration', ' Label'
]

# نام‌های استاندارد مورد استفاده در کل پایان‌نامه
FINAL_FEATURES = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]

print("[INFO] Loading and Engineering Features...")
# خواندن ستون‌های خام
df = pd.read_csv(dataset_path, usecols=RAW_COLUMNS)
df.columns = df.columns.str.strip()

# ۲. مهندسی ویژگی (تطبیق دقیق با مدل‌های LSTM و XGBoost قبلی)
df["Packet_Count"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
df["Packet_Size"] = df["Average Packet Size"]
df["Flow_Rate"] = df["Flow Packets/s"]
df["Flow_Duration"] = df["Flow Duration"]

# پاکسازی
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FINAL_FEATURES)
X = df[FINAL_FEATURES]

# ۳. تقسیم‌بندی و نمونه‌برداری (حفظ ترتیب و منطق مدل‌سازی)
_, X_test = train_test_split(X, test_size=0.2, random_state=42)
del df

sample_size = min(5000, len(X_test))
X_test_sampled = X_test.sample(n=sample_size, random_state=42)

# ۴. لود کردن مدل (مطمئن شوید مدل روی همین ۴ ویژگی آموزش دیده است)
os.chdir(results_path)
print(f"[INFO] Loading Model and calculating SHAP for {sample_size} samples...")
model = joblib.load("random_forest_attack_detector_sim.pkl")

# ۵. محاسبات SHAP
explainer = shap.TreeExplainer(model)
shap_results = explainer.shap_values(X_test_sampled)

# مدیریت خروجی مدل‌های چندکلاسه (RF معمولاً لیست برمی‌گرداند)
# ما کلاس ۱ (حمله) را برای تحلیل انتخاب می‌کنیم
if isinstance(shap_results, list):
    actual_shap_values = shap_results[1]
else:
    actual_shap_values = shap_results

# ۶. رسم نمودارهای استاندارد با نام‌های یکسان
print("[INFO] Generating final consistent plots...")


explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_sampled)

# مدیریت مدل‌های چندکلاسه برای RF
if isinstance(shap_values, list):
    # انتخاب کلاس ۱ (حمله)
    actual_shap_obj = shap_values[1]
elif len(shap_values.shape) == 3:
    # ساختار (samples, features, classes)
    actual_shap_obj = shap_values[:, :, 1]
else:
    actual_shap_obj = shap_values

# ۶. رسم نمودارهای نهایی با استایل پایان‌نامه
print("[INFO] Generating plots...")

# الف) نمودار Beeswarm (خلاصه‌ی تأثیرات)
plt.figure(figsize=(10, 6))
shap.plots.beeswarm(actual_shap_obj, show=False)
plt.title("SHAP Summary Plot (Consistent Features)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("shap_summary_plot_consistent.png", dpi=300)
plt.close()

# ب) نمودار میله‌ای (Feature Importance) - اصلاح شده
plt.figure(figsize=(10, 6))
# استفاده از میانگین مقادیر مطلق برای رسم نمودار میله‌ای واقعی
shap.plots.bar(actual_shap_obj, show=False) 
plt.title("Global Feature Importance (SHAP Bar Plot)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("shap_bar_plot_consistent.png", dpi=300)
plt.close()

print(f"[SUCCESS] Both Beeswarm and Bar plots are saved in {results_path}")