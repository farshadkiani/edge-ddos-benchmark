import os

# TensorFlow env vars must be set before importing tensorflow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =========================================================
# Paths and constants
# =========================================================
FILE_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 20

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

SCALER_PATH = os.path.join(RESULTS_DIR, "ddos_scaler.pkl")
MODEL_PATH = os.path.join(RESULTS_DIR, "lstm_attack_detector_final.keras")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_lstm_attack_detector.keras")
TFLITE_PATH = os.path.join(RESULTS_DIR, "lstm_attack_detector_final.tflite")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_2017.json")

# =========================================================
# Load dataset
# =========================================================
print("[INFO] Loading dataset...")
df = pd.read_csv(FILE_PATH, low_memory=False)
df.columns = df.columns.str.strip()

missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

df = df[RAW_REQUIRED_COLUMNS].copy()

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
print(f"[INFO] Target features (Total 4 columns): {FEATURE_COLUMNS}")
print("[INFO] Class distribution:")
print(df["binary_label"].value_counts())

# =========================================================
# Split train/test without shuffle to preserve ordering
# =========================================================
split_idx = int(len(df) * (1 - TEST_RATIO))

X_train_raw = X[:split_idx]
y_train_raw = y[:split_idx]
X_test_raw = X[split_idx:]
y_test_raw = y[split_idx:]

print(f"[INFO] Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

joblib.dump(scaler, SCALER_PATH)
print(f"[INFO] Scaler saved to: {SCALER_PATH}")

# =========================================================
# Create sequences (sliding window)
# =========================================================
def create_sequences(features, labels, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i + seq_len])
        y_seq.append(labels[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)


print("[INFO] Creating sequences...")
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQUENCE_LENGTH)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQUENCE_LENGTH)

print(f"[INFO] X_train shape: {X_train.shape}")
print(f"[INFO] X_test shape: {X_test.shape}")

# =========================================================
# Build LSTM
# =========================================================
def build_lstm_model(time_steps, n_features):
    model = Sequential(
        [
            Input(shape=(time_steps, n_features)),
            LSTM(32, activation="tanh", recurrent_activation="sigmoid", unroll=True),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_lstm_model(SEQUENCE_LENGTH, X_train.shape[2])
model.summary()

# =========================================================
# Callbacks & training
# =========================================================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

print("[INFO] Training model...")
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
    shuffle=False,
)

model.save(MODEL_PATH)
model = load_model(BEST_MODEL_PATH)

# =========================================================
# Evaluate
# =========================================================
print("[INFO] Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(np.int32)

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
acc = accuracy_score(y_test, y_pred)

print(f"\n[RESULT] Test Loss: {test_loss:.6f}")
print(f"[RESULT] Test Accuracy (Keras): {test_acc:.6f}")
print(f"[RESULT] Accuracy: {acc:.6f}")
print(f"[RESULT] Precision: {precision:.6f}")
print(f"[RESULT] Recall: {recall:.6f}")
print(f"[RESULT] F1-score: {f1:.6f}")

cm = confusion_matrix(y_test, y_pred)
print("\n[RESULT] Confusion Matrix:\n", cm)

report = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], zero_division=0)
print("\n[RESULT] Classification Report:\n", report)

metrics = {
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "test_loss": float(test_loss),
    "test_accuracy_keras": float(test_acc),
    "feature_columns": FEATURE_COLUMNS,
    "sequence_length": SEQUENCE_LENGTH,
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# =========================================================
# TFLite conversion
# =========================================================
print("[INFO] Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"[INFO] Keras model saved to: {MODEL_PATH}")
print(f"[INFO] Best model saved to: {BEST_MODEL_PATH}")
print(f"[INFO] TFLite model saved to: {TFLITE_PATH}")
print(f"[INFO] Metrics saved to: {METRICS_PATH}")
print("[INFO] Pipeline execution finished.")
