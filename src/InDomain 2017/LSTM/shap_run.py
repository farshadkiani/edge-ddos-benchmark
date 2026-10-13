import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# تنظیمات تنسورفلو
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================================================
# ۱. تنظیمات مسیرها
# =========================================================
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "best_lstm_attack_detector.keras")
SCALER_PATH = os.path.join(RESULTS_DIR, "ddos_scaler.pkl")
FILE_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

FEATURE_COLUMNS = ["Packet_Count", "Packet_Size", "Flow_Rate", "Flow_Duration"]
SEQUENCE_LENGTH = 10

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"مدل در مسیر {MODEL_PATH} یافت نشد.")

print("[INFO] Loading LSTM model and Scaler...")
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================================================
# ۲. آماده‌سازی داده‌های تست
# =========================================================
print("[INFO] Preparing test sequences...")
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
df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

X_raw = df[FEATURE_COLUMNS].values.astype(np.float32)
X_scaled = scaler.transform(X_raw)

def create_sequences(data, seq_len):
    X_seq = []
    for i in range(len(data) - seq_len):
        X_seq.append(data[i:i + seq_len])
    return np.array(X_seq, dtype=np.float32)

split_idx = int(len(df) * 0.8)
X_test_all = X_scaled[split_idx:]
X_test_seq = create_sequences(X_test_all, SEQUENCE_LENGTH)[:500] # انتخاب ۵۰ sequence برای تحلیل

# =========================================================
# ۳. تابع محاسبه Integrated Gradients (Native TF)
# =========================================================
def compute_integrated_gradients(model, input_samples, steps=50):
    """
    محاسبه Integrated Gradients بدون نیاز به کتابخانه‌های خارجی
    """
    # خط پایه (Baseline): توالی‌های صفر به همان شکل ورودی
    baseline = tf.zeros_like(input_samples)
    
    # ایجاد مسیرهای گام‌به‌گام (Interpolation)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    
    ig_attributions = []
    
    for sample_idx in range(len(input_samples)):
        x_sample = input_samples[sample_idx] # (seq_len, n_features)
        x_base = baseline[sample_idx]
        
        # ساخت نمونه‌های مسیر مابین baseline و ورودی واقعی
        interpolated_inputs = [x_base + alpha * (x_sample - x_base) for alpha in alphas]
        interpolated_inputs = tf.stack(interpolated_inputs) # (steps+1, seq_len, n_features)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_inputs)
            preds = model(interpolated_inputs)
            
        # محاسبه گرادیان‌ها
        grads = tape.gradient(preds, interpolated_inputs) # (steps+1, seq_len, n_features)
        
        # تقریب انتگرال با قانون ذوزنقه‌ای (Trapezoidal Rule)
        avg_grads = tf.reduce_mean(grads[:-1] + grads[1:], axis=0) / 2.0
        integrated_grad = (x_sample - x_base) * avg_grads
        
        ig_attributions.append(integrated_grad.numpy())
        
    return np.array(ig_attributions)

print("[INFO] Computing Integrated Gradients for LSTM...")
ig_attributions = compute_integrated_gradients(model, X_test_seq, steps=50)
# شکل خروجی: (samples, time_steps, features)

# =========================================================
# ۴. تحلیل اهمیت کلی ویژگی‌ها و بصری‌سازی
# =========================================================
# ۱. میانگین‌گیری روی نمونه‌ها و گام‌های زمانی
global_importance = np.mean(np.abs(ig_attributions), axis=(0, 1))
sorted_idx = np.argsort(global_importance)

# رسم نمودار ستونی Importance
plt.figure(figsize=(10, 5))
bars = plt.barh([FEATURE_COLUMNS[i] for i in sorted_idx], 
                global_importance[sorted_idx], 
                color="#00a896", height=0.6)

max_val = max(global_importance) if len(global_importance) > 0 else 1
for bar in bars:
    width = bar.get_width()
    plt.text(width + (max_val * 0.01), bar.get_y() + bar.get_height()/2, 
             f'+{width:.4f}', va='center', fontsize=11, color="#00a896", fontweight='bold')

plt.title("Integrated Gradients Feature Importance (LSTM - CICIDS2017)", fontsize=13)
plt.xlabel("Mean Absolute Attribution Score", fontsize=11)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lstm_ig_bar.png"), dpi=300)
plt.close()

# ۲. رسم نقشه حرارتی (Heatmap) ویژگی‌ها در طول گام‌های زمانی (Temporal Analysis)
temporal_importance = np.mean(np.abs(ig_attributions), axis=0).T # (features, time_steps)

plt.figure(figsize=(10, 5))
im = plt.imshow(temporal_importance, cmap='YlGnBu', aspect='auto')
plt.colorbar(im, label='Attribution Score')
plt.yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
plt.xticks(range(SEQUENCE_LENGTH), [f't-{SEQUENCE_LENGTH - i}' for i in range(SEQUENCE_LENGTH)])
plt.title("Temporal Attribution Heatmap across 10 Time Steps (LSTM)", fontsize=13)
plt.xlabel("Time Steps in Sequence", fontsize=11)
plt.ylabel("Features", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lstm_ig_temporal_heatmap.png"), dpi=300)
plt.close()

print(f"[SUCCESS] Integrated Gradients analysis completed! Results saved to '{RESULTS_DIR}'.")
