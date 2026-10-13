#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LSTM Training Pipeline for CICIDS2017 Friday DDoS
Ablation configuration: WITHOUT Packet_Count

Train domain: Friday DDoS
Validation: contiguous temporal validation partition from Friday

Features used:
- Packet_Size
- Flow_Rate
- Flow_Duration

Feature removed:
- Packet_Count
"""

import os

# TensorFlow env vars must be set before importing tensorflow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =========================================================
# Paths and constants
# =========================================================
FRIDAY_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10
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

# Ablation: WITHOUT Packet_Count
# Actual model input features
FEATURE_COLUMNS = [
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

# For fair preprocessing only: keep same invalid-row filtering policy
# across all ablation variants.
ALL_FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

SCALER_PATH = os.path.join(RESULTS_DIR, "ddos_scaler_no_packet_count.pkl")
MODEL_PATH = os.path.join(RESULTS_DIR, "lstm_attack_detector_no_packet_count.keras")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_lstm_attack_detector.keras")
TFLITE_PATH = os.path.join(RESULTS_DIR, "lstm_attack_detector_without_packet_count.tflite")
THRESHOLD_PATH = os.path.join(RESULTS_DIR, "best_threshold.txt")

# =========================================================
# Helpers
# =========================================================
def create_sequences(features, labels, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_len):
        X_seq.append(features[i:i + seq_len])
        y_seq.append(labels[i + seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int32)


def compute_f2(precision, recall):
    denom = 4 * precision + recall
    if denom == 0:
        return 0.0
    return (5 * precision * recall) / denom


def preprocess_dataframe(file_path):
    print(f"[INFO] Reading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    missing_cols = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df = df[RAW_REQUIRED_COLUMNS].copy()

    df["Packet_Count"] = (
        pd.to_numeric(df["Total Fwd Packets"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Total Backward Packets"], errors="coerce").fillna(0)
    )
    df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
    df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
    df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Packet_Count is created and checked for fairness,
    # but it is NOT used as model input.
    df = df.dropna(subset=ALL_FEATURE_COLUMNS + ["Label"]).reset_index(drop=True)

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    df["binary_label"] = (df["Label"] != "BENIGN").astype(np.int32)

    return df


# =========================================================
# Load & Split Friday Train/Validation
# =========================================================
df_friday = preprocess_dataframe(FRIDAY_PATH)
X_friday = df_friday[FEATURE_COLUMNS].values.astype(np.float32)
y_friday = df_friday["binary_label"].values.astype(np.int32)

print(f"[INFO] Friday dataset shape: {X_friday.shape}")
print(f"[INFO] Used features: {FEATURE_COLUMNS}")
print("[INFO] Friday class distribution:")
print(df_friday["binary_label"].value_counts())

# Find a valid chronological split with both classes in train and validation.
n_rows = len(df_friday)
selected_split_idx = None

for val_fraction in np.arange(0.10, 0.45, 0.01):
    split_idx = int(n_rows * (1.0 - val_fraction))
    y_tr_c = y_friday[:split_idx]
    y_val_c = y_friday[split_idx:]

    if len(np.unique(y_tr_c)) == 2 and len(np.unique(y_val_c)) == 2:
        selected_split_idx = split_idx
        break

if selected_split_idx is None:
    selected_split_idx = int(n_rows * 0.8)
    print("[WARNING] Contiguous split search did not satisfy class diversity. Fallback to 80/20.")
else:
    print(
        f"[INFO] Selected split index: {selected_split_idx} "
        f"(Val Ratio: {1 - (selected_split_idx / n_rows):.2%})"
    )

X_train_raw = X_friday[:selected_split_idx]
y_train_raw = y_friday[:selected_split_idx]
X_val_raw = X_friday[selected_split_idx:]
y_val_raw = y_friday[selected_split_idx:]

# Fit scaler only on Friday training subset.
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)

joblib.dump(scaler, SCALER_PATH)
print(f"[INFO] Scaler saved to: {SCALER_PATH}")

# Build temporal sequences.
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQUENCE_LENGTH)
X_val, y_val = create_sequences(X_val_scaled, y_val_raw, SEQUENCE_LENGTH)

print(f"[INFO] Train sequences: {X_train.shape}")
print(f"[INFO] Validation sequences: {X_val.shape}")

# =========================================================
# Build & Train Model
# =========================================================
def build_lstm_model(time_steps, n_features):
    model = Sequential([
        Input(shape=(time_steps, n_features)),
        LSTM(32, activation="tanh", recurrent_activation="sigmoid", unroll=True),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_lstm_model(SEQUENCE_LENGTH, X_train.shape[2])
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

print("[INFO] Training model on Friday data...")
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
    shuffle=False,
)

model.save(MODEL_PATH)
print(f"[INFO] Final Keras model saved to: {MODEL_PATH}")

# =========================================================
# Threshold Calibration on Friday Validation
# =========================================================
print("[INFO] Calibrating decision threshold on Friday Validation subset...")
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)
y_val_prob = best_model.predict(X_val, verbose=0).ravel()

best_threshold = 0.5
best_f2 = -1.0

for t in np.linspace(0.01, 0.99, 99):
    y_pred_t = (y_val_prob >= t).astype(np.int32)
    p = precision_score(y_val, y_pred_t, zero_division=0)
    r = recall_score(y_val, y_pred_t, zero_division=0)
    f2 = compute_f2(p, r)

    if f2 > best_f2:
        best_f2 = f2
        best_threshold = float(t)

print(f"[SUCCESS] Optimal Threshold found: {best_threshold:.4f} (Validation F2: {best_f2:.6f})")
with open(THRESHOLD_PATH, "w") as f:
    f.write(f"{best_threshold:.4f}")

# =========================================================
# TFLite Conversion
# =========================================================
print("\n[INFO] Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_lower_tensor_list_ops = True

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"[INFO] TFLite model successfully saved to: {TFLITE_PATH}")
print("[INFO] Training pipeline execution finished.")

