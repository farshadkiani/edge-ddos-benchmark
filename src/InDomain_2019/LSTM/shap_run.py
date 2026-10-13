#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# تنظیمات محیطی
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# =========================================================
# ۱. مسیرهای مطلق (جلوگیری از باگ مسیردهی ویندوز و اسپایدر)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "lstm_model_best.h5"
SCALER_PATH = RESULTS_DIR / "scaler_lstm.pkl"
FILE_PATH = BASE_DIR / "DrDoS_DNS.csv"

FEATURES = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]

print(f"[INFO] Checking paths...")
print(f"  - Model Path:  {MODEL_PATH} (Exists: {MODEL_PATH.exists()})")
print(f"  - Scaler Path: {SCALER_PATH} (Exists: {SCALER_PATH.exists()})")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"مدل در مسیر {MODEL_PATH} یافت نشد. لطفاً ابتدا کد آموزش را اجرا کنید.")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"اسکیلر در مسیر {SCALER_PATH} یافت نشد.")

# بارگذاری اسکیلر
print("[INFO] Loading StandardScaler...")
scaler = joblib.load(str(SCALER_PATH))

# بارگذاری مدل با مکانیزم Fallback ایمن
print("[INFO] Loading Trained LSTM Model...")
try:
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("[SUCCESS] Loaded model via load_model.")
except Exception as e:
    print(f"[WARNING] load_model failed ({e}). Reconstructing architecture and loading weights...")
    # بازسازی دقیق معماری کد آموزش شما
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(len(FEATURES), 1), return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.load_weights(str(MODEL_PATH))
    print("[SUCCESS] Model weights loaded successfully into architecture.")

# =========================================================
# ۲. آماده‌سازی داده‌های نمونه
# =========================================================
print("[INFO] Reading sample data from dataset...")
RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

header_df = pd.read_csv(str(FILE_PATH), nrows=0)
col_mapping = {c.strip(): c for c in header_df.columns}
actual_cols_to_load = [col_mapping[c] for c in RAW_REQUIRED_COLUMNS]

df = pd.read_csv(str(FILE_PATH), usecols=actual_cols_to_load, low_memory=False, nrows=50000)
df.columns = df.columns.str.strip()

# مهندسی ویژگی‌ها دقیقاً مطابق پایپ‌لاین آموزش
df["Packet_Count"] = (
    pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
    + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
)
df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna(subset=FEATURES).reset_index(drop=True)

X_raw = df[FEATURES].to_numpy(dtype=np.float32)
X_scaled = scaler.transform(X_raw)

# تغییر بعد به (samples, timesteps=4, features=1)
X_lstm = X_scaled.reshape((X_scaled.shape[0], len(FEATURES), 1))
X_test_sample = X_lstm[-500:]  # انتخاب ۵۰۰ نمونه برای تحلیل

# =========================================================
# ۳. محاسبه Integrated Gradients
# =========================================================
def compute_integrated_gradients(model, input_samples, steps=50):
    baseline = tf.zeros_like(input_samples)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    ig_attributions = []

    for sample_idx in range(len(input_samples)):
        x_sample = input_samples[sample_idx]
        x_base = baseline[sample_idx]

        interpolated_inputs = [x_base + alpha * (x_sample - x_base) for alpha in alphas]
        interpolated_inputs = tf.stack(interpolated_inputs)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            preds = model(interpolated_inputs)

        grads = tape.gradient(preds, interpolated_inputs)
        avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0
        integrated_grad = (x_sample - x_base) * avg_grads
        ig_attributions.append(integrated_grad.numpy())

    return np.array(ig_attributions)

print("[INFO] Computing Integrated Gradients for LSTM...")
ig_attributions = compute_integrated_gradients(model, X_test_sample, steps=50)

# =========================================================
# ۴. رسم و ذخیره نمودار اهمیت ویژگی‌ها
# =========================================================
ig_attributions_squeezed = np.squeeze(ig_attributions, axis=-1)
global_importance = np.mean(np.abs(ig_attributions_squeezed), axis=0)
sorted_idx = np.argsort(global_importance)

plt.figure(figsize=(8, 4.5), dpi=300)
bars = plt.barh(
    [FEATURES[i] for i in sorted_idx],
    global_importance[sorted_idx],
    color="#d90429",
    height=0.6,
    edgecolor="black",
    linewidth=0.5
)

max_val = max(global_importance) if len(global_importance) > 0 else 1.0
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + (max_val * 0.01),
        bar.get_y() + bar.get_height() / 2,
        f"+{width:.4f}",
        va="center",
        fontsize=10,
        color="#d90429",
        fontweight="bold"
    )

plt.title("Integrated Gradients Feature Importance (LSTM - CICDDoS2019)", fontsize=11, weight="bold")
plt.xlabel("Mean Absolute Attribution Score", fontsize=10)
plt.ylabel("Engineered Feature", fontsize=10)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()

output_fig_path = RESULTS_DIR / "lstm_ig_bar_2019.png"
plt.savefig(str(output_fig_path), dpi=300)
plt.close()

print(f"[SUCCESS] Feature attribution analysis completed! Saved in '{output_fig_path}'.")
