# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:18:27 2026

@author: farshad
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

# =========================================================
# Paths and constants
# =========================================================
FILE_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_RATIO = 0.2
RANDOM_STATE = 42

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

MODEL_PATH = os.path.join(RESULTS_DIR, "xgboost_attack_detector.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_xgb_2017.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report_xgb.txt")

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
# Same split style as LSTM and RF for fair comparison
# =========================================================
split_idx = int(len(df) * (1 - TEST_RATIO))

X_train = X[:split_idx]
y_train = y[:split_idx]
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# =========================================================
# Build XGBoost
# =========================================================
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
)

# =========================================================
# Train
# =========================================================
print("[INFO] Training XGBoost model...")
train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start

joblib.dump(model, MODEL_PATH)
print(f"[INFO] Model saved to: {MODEL_PATH}")

# =========================================================
# Evaluate
# =========================================================
print("[INFO] Evaluating model...")
pred_start = time.time()
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
pred_time = time.time() - pred_start

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n[RESULT] Accuracy: {acc:.6f}")
print(f"[RESULT] Precision: {precision:.6f}")
print(f"[RESULT] Recall: {recall:.6f}")
print(f"[RESULT] F1-score: {f1:.6f}")
print(f"[RESULT] ROC-AUC: {roc_auc:.6f}")
print(f"[RESULT] Train Time: {train_time:.6f} sec")
print(f"[RESULT] Prediction Time: {pred_time:.6f} sec")

cm = confusion_matrix(y_test, y_pred)
print("\n[RESULT] Confusion Matrix:\n", cm)

report = classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "DDoS"],
    zero_division=0,
)
print("\n[RESULT] Classification Report:\n", report)

metrics = {
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "train_time_sec": float(train_time),
    "prediction_time_sec": float(pred_time),
    "feature_columns": FEATURE_COLUMNS,
    "raw_required_columns": RAW_REQUIRED_COLUMNS,
    "test_ratio": TEST_RATIO,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": float(scale_pos_weight),
    "model_path": MODEL_PATH,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"[INFO] Metrics saved to: {METRICS_PATH}")
print(f"[INFO] Classification report saved to: {REPORT_PATH}")
print("[INFO] Pipeline execution finished.")
