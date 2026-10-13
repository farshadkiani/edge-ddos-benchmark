import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# =========================================================
# ۱. تنظیمات مسیرها و بارگذاری مدل
# =========================================================
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "random_forest_attack_detector.pkl")
FILE_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"مدل در مسیر {MODEL_PATH} یافت نشد.")

print("[INFO] Loading model...")
model = joblib.load(MODEL_PATH)

# =========================================================
# ۲. بارگذاری و آماده‌سازی داده‌ها
# =========================================================
print("[INFO] Loading and preparing dataset...")
df = pd.read_csv(FILE_PATH, low_memory=False)
df.columns = df.columns.str.strip()

df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
)
df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=FEATURE_COLUMNS + ["Label"]).reset_index(drop=True)

X = df[FEATURE_COLUMNS].values.astype(np.float32)

TEST_RATIO = 0.2
split_idx = int(len(df) * (1 - TEST_RATIO))
X_test = X[split_idx:]
X_test_subset = X_test[:1000] 
X_test_df = pd.DataFrame(X_test_subset, columns=FEATURE_COLUMNS)

# =========================================================
# ۳. محاسبه مقادیر SHAP
# =========================================================
print(f"[INFO] Computing SHAP values on {len(X_test_df)} samples...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df, approximate=True, check_additivity=False)

# مدیریت ساختار خروجی Random Forest برای کلاس حمله (Index 1)
if isinstance(shap_values, list):
    # اگر خروجی لیست بود (معمولا در نسخه های قدیمی تر)
    shap_values_ddos = shap_values[1]
elif len(shap_values.shape) == 3:
    # اگر خروجی ۳ بعدی بود [samples, features, classes] (نسخه های جدید)
    shap_values_ddos = shap_values[:, :, 1]
else:
    shap_values_ddos = shap_values

# =========================================================
# ۴. رسم و ذخیره نمودارها با استایل درخواستی (اصلاح شده)
# =========================================================
print("[INFO] Generating styled plots...")
os.makedirs(RESULTS_DIR, exist_ok=True)

# محاسبه میانگین قدر مطلق
mean_shap_values = np.abs(shap_values_ddos).mean(0).flatten()
# مرتب‌سازی ایندکس‌ها
inds = np.argsort(mean_shap_values)

# استخراج نام‌ها و مقادیر به صورت لیست ساده
sorted_features = [str(FEATURE_COLUMNS[i]) for i in inds]
sorted_values = [float(mean_shap_values[i]) for i in inds]

plt.figure(figsize=(10, 5))
bars = plt.barh(sorted_features, sorted_values, color="#ff0051", height=0.6)

# اضافه کردن اعداد کنار میله‌ها
max_val = max(sorted_values) if len(sorted_values) > 0 else 1
for bar in bars:
    width = bar.get_width()
    plt.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2, 
             f'+{width:.2f}', va='center', fontsize=11, color="#ff0051", fontweight='bold')

plt.title("SHAP Feature Importance (Random Forest - CICIDS2017)", fontsize=13)
plt.xlabel("mean(|SHAP value|)", fontsize=11)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rf_2017_shap_bar_styled.png"), dpi=300)
plt.close()

# رسم نمودار Summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_ddos, X_test_df, show=False)
plt.savefig(os.path.join(RESULTS_DIR, "rf_2017_shap_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] All plots saved in '{RESULTS_DIR}' folder.")
