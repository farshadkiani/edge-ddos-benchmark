#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# =========================
# Paths and constants
# =========================
TRAIN_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(RESULTS_DIR, "random_forest_friday_model_without_packet_count.pkl")
METADATA_PATH = os.path.join(RESULTS_DIR, "train_metadata_rf_without_packet_count.json")
VAL_METRICS_PATH = os.path.join(RESULTS_DIR, "rf_friday_validation_metrics_without_packet_count.json")

# Packet_Count removed
FEATURE_COLUMNS = ["Packet_Size", "Flow_Rate", "Flow_Duration"]

RAW_REQUIRED_COLUMNS = [
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]

RANDOM_STATE = 42
N_ESTIMATORS = 200
VAL_RATIO = 0.20


# =========================
# Helpers
# =========================
def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_and_prepare_data(file_path):
    print(f"[INFO] Loading Friday dataset: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    df.columns = df.columns.str.strip()

    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training file: {missing}")

    df = df[RAW_REQUIRED_COLUMNS].copy()

    # Feature engineering (without Packet_Count)
    df["Packet_Size"] = pd.to_numeric(df["Average Packet Size"], errors="coerce")
    df["Flow_Rate"] = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
    df["Flow_Duration"] = pd.to_numeric(df["Flow Duration"], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before_drop = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["Label"]).reset_index(drop=True)
    after_drop = len(df)
    dropped_rows = before_drop - after_drop

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    y = (df["Label"] != "BENIGN").astype(np.int32).values
    X = df[FEATURE_COLUMNS].values.astype(np.float32)

    stats = {
        "n_rows_raw": int(before_drop),
        "n_rows_clean": int(after_drop),
        "dropped_rows": int(dropped_rows),
        "n_benign": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
    }

    return X, y, stats


def find_valid_temporal_split(y, val_ratio=0.20, min_train_ratio=0.55, min_val_ratio=0.10):
    n = len(y)

    start_idx = int(n * min_train_ratio)
    end_idx = int(n * (1.0 - min_val_ratio))

    if start_idx >= end_idx:
        raise ValueError("Invalid split search range. Check dataset size and ratios.")

    candidate_idx = int(n * (1.0 - val_ratio))
    candidate_idx = max(start_idx, min(candidate_idx, end_idx))

    def has_both_classes(arr):
        return len(np.unique(arr)) >= 2

    if has_both_classes(y[:candidate_idx]) and has_both_classes(y[candidate_idx:]):
        return candidate_idx

    for split_idx in range(start_idx, end_idx + 1):
        if has_both_classes(y[:split_idx]) and has_both_classes(y[split_idx:]):
            return split_idx

    raise ValueError(
        "Could not find a contiguous temporal split where both train and validation "
        "contain both classes."
    )


def evaluate_predictions(y_true, y_prob, threshold=0.50):
    y_pred = (y_prob >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================
# Main training pipeline
# =========================
def train_pipeline():
    ensure_results_dir()

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            f"Make sure the Friday CSV exists in /mnt/data."
        )

    X, y, data_stats = load_and_prepare_data(TRAIN_FILE)

    split_idx = find_valid_temporal_split(y, val_ratio=VAL_RATIO)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"[INFO] Clean samples: {len(X)}")
    print(f"[INFO] Dropped rows: {data_stats['dropped_rows']}")
    print(f"[INFO] Class distribution -> BENIGN: {data_stats['n_benign']}, ATTACK: {data_stats['n_attack']}")
    print(f"[INFO] Temporal split index: {split_idx}")
    print(f"[INFO] Train size: {len(X_train)}, Validation size: {len(X_val)}")

    print("[INFO] Training Random Forest model (without Packet_Count)...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"[INFO] Training completed in {train_time:.2f} s")

    # Validation evaluation at default threshold=0.50
    val_start = time.time()
    val_prob = model.predict_proba(X_val)[:, 1]
    val_time = time.time() - val_start
    avg_val_latency_ms = (val_time / max(len(X_val), 1)) * 1000.0

    val_metrics = evaluate_predictions(y_val, val_prob, threshold=0.50)
    val_metrics["validation_total_inference_time_sec"] = float(val_time)
    val_metrics["validation_avg_latency_ms_per_sample"] = float(avg_val_latency_ms)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")

    # Save metadata
    metadata = {
        "train_file": TRAIN_FILE,
        "model_path": MODEL_PATH,
        "results_dir": RESULTS_DIR,
        "model_type": "RandomForestClassifier",
        "ablation_setting": "without_packet_count",
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": -1,
        "feature_columns": FEATURE_COLUMNS,
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "data_stats": data_stats,
        "temporal_split": {
            "method": "contiguous_temporal_split",
            "validation_ratio_target": VAL_RATIO,
            "split_index": int(split_idx),
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_val)),
        },
        "training_time_sec": float(train_time),
    }
    save_json(metadata, METADATA_PATH)
    print(f"[INFO] Training metadata saved to: {METADATA_PATH}")

    # Save validation metrics
    save_json(val_metrics, VAL_METRICS_PATH)
    print(f"[INFO] Validation metrics saved to: {VAL_METRICS_PATH}")

    # Optional: feature importance
    if hasattr(model, "feature_importances_"):
        fi = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        fi_path = os.path.join(RESULTS_DIR, "feature_importance_rf_without_packet_count.json")
        save_json(fi, fi_path)
        print(f"[INFO] Feature importance saved to: {fi_path}")

    print("[INFO] Validation results at threshold=0.50:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print("[INFO] Friday RF training pipeline without Packet_Count completed successfully.")


if __name__ == "__main__":
    train_pipeline()
