import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ۱. تنظیمات مسیر
dataset_path = r"DrDoS_DNS.csv"
results_path = r"results2"

# ۲. خواندن دیتاست به صورت بهینه
print("Loading dataset and fixing column names...")

# برای جلوگیری از خطای نام ستون، ابتدا فقط ردیف اول را می‌خوانیم تا نام دقیق ستون‌ها را بفهمیم
df_sample = pd.read_csv(dataset_path, nrows=1)
# حذف فاصله‌های اضافه از نام تمام ستون‌ها
df_sample.columns = df_sample.columns.str.strip()

# تعریف ۴ ویژگی استاندارد شما بر اساس نام‌های موجود در دیتاست ۲۰۱۹
# دقت کنید: در ۲۰۱۹ نام دقیق ستون 'Packet Length Mean' و 'Total Fwd Packets' است
mapping = {
    'Total Fwd Packets': 'Packet_Count',
    'Packet Length Mean': 'Packet_Size',
    'Flow Packets/s': 'Flow_Rate',
    'Flow Duration': 'Flow_Duration'
}

# پیدا کردن نام‌های اصلی (با فاصله یا بدون فاصله) که در فایل وجود دارند
original_cols_to_read = []
for col in mapping.keys():
    # جستجو در ستون‌های فایل اصلی برای پیدا کردن مشابه (بدون در نظر گرفتن فاصله)
    actual_col = [c for c in pd.read_csv(dataset_path, nrows=0).columns if c.strip() == col][0]
    original_cols_to_read.append(actual_col)

# حالا فقط ستون‌های مورد نیاز را لود می‌کنیم
df = pd.read_csv(dataset_path, usecols=original_cols_to_read)
df.columns = df.columns.str.strip() # حذف فاصله‌ها
df = df.rename(columns=mapping)     # تغییر نام به نام‌های استاندارد پایان‌نامه

# ۳. پاکسازی
print("Cleaning data...")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# ۴. تقسیم داده (حفظ پروتکل ۸۰-۲۰)
print("Splitting dataset...")
_, X_test = train_test_split(df, test_size=0.2, random_state=42)
del df 

# ۵. لود مدل و محاسبات SHAP
if not os.path.exists(results_path):
    os.makedirs(results_path)
os.chdir(results_path)

print("Loading XGBoost model...")
model = joblib.load("xgboost_attack_detector_sim.pkl")

print("Calculating SHAP values (Beeswarm & Bar)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# ۶. ترسیم دو نمودار مورد نظر شما با نام‌های استاندارد
# نمودار اول: Beeswarm (Summary)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Beeswarm Plot - XGBoost (Consistent Features)", fontsize=12)
plt.tight_layout()
plt.savefig("xgb_shap_summary_consistent.png", dpi=300, bbox_inches="tight")
plt.close()

# نمودار دوم: Bar
plt.figure(figsize=(10, 5))
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance Bar Plot - XGBoost", fontsize=12)
plt.tight_layout()
plt.savefig("xgb_shap_bar_consistent.png", dpi=300, bbox_inches="tight")
plt.close()

print("Process finished! Two consistent plots saved in:", results_path)
