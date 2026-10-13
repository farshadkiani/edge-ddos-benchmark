#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import itertools
from typing import Any, Dict, Tuple

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


# =========================================================
# Paths and constants
# =========================================================

TRAIN_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "random_forest_friday_model.pkl",
)


# =========================================================
# Feature configuration
# =========================================================

FEATURE_COLUMNS = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Average Packet Size",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]


# =========================================================
# Experiment configuration
# =========================================================

RANDOM_STATE = 42
VAL_RATIO = 0.20
BETA = 2.0

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"],
}

THRESHOLD_GRID = np.linspace(
    0.01,
    0.99,
    99,
)


# =========================================================
# Utility functions
# =========================================================

def ensure_results_dir() -> None:
    """Create the results directory if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# Data preparation
# =========================================================

def load_and_prepare_data(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    missing_columns = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    # Numeric conversion
    fwd_packets = pd.to_numeric(df["Total Fwd Packets"], errors="coerce")
    backward_packets = pd.to_numeric(df["Total Backward Packets"], errors="coerce")
    packet_size = pd.to_numeric(df["Average Packet Size"], errors="coerce")
    flow_rate = pd.to_numeric(df["Flow Packets/s"], errors="coerce")
    flow_duration = pd.to_numeric(df["Flow Duration"], errors="coerce")

    df["Packet_Count"] = fwd_packets + backward_packets
    df["Packet_Size"] = packet_size
    df["Flow_Rate"] = flow_rate
    df["Flow_Duration"] = flow_duration

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for column in FEATURE_COLUMNS:
        df.loc[df[column] < 0, column] = np.nan

    df["Label"] = df["Label"].astype("string").str.strip().str.upper()

    valid_label_mask = (df["Label"].notna() & (df["Label"] != "") & (df["Label"] != "NAN") & (df["Label"] != "NONE"))
    df = df.loc[valid_label_mask].copy()
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    y = (df["Label"] != "BENIGN").astype(np.int32).to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    return X, y


# =========================================================
# Temporal split
# =========================================================

def find_valid_temporal_split(y: np.ndarray, val_ratio: float = VAL_RATIO) -> int:
    n_samples = len(y)
    target_split = int(round(n_samples * (1.0 - val_ratio)))

    def has_both_classes(values: np.ndarray) -> bool:
        return len(np.unique(values)) == 2

    for split_idx in range(int(n_samples * 0.55), int(n_samples * 0.90)):
        if has_both_classes(y[:split_idx]) and has_both_classes(y[split_idx:]):
            if abs(split_idx - target_split) < 100:
                return int(split_idx)
    return target_split


# =========================================================
# Metrics
# =========================================================

def evaluate_predictions(y_true: np.ndarray, y_probability: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_prediction = (y_probability >= float(threshold)).astype(np.int32)
    return {"f2": float(fbeta_score(y_true, y_prediction, beta=BETA, zero_division=0))}


# =========================================================
# Grid Search Logic
# =========================================================

def generate_parameter_combinations():
    parameter_names = list(PARAM_GRID.keys())
    parameter_values = [PARAM_GRID[name] for name in parameter_names]
    for values in itertools.product(*parameter_values):
        yield dict(zip(parameter_names, values))

def parameter_key(parameters: Dict[str, Any]):
    return tuple((key, str(parameters[key])) for key in sorted(parameters.keys()))

def is_current_model_better(current_f2: float, current_threshold: float, current_parameters: Dict[str, Any],
                            best_f2: float, best_threshold: float, best_parameters: Dict[str, Any]) -> bool:
    if best_f2 is None: return True
    if current_f2 > best_f2: return True
    if np.isclose(current_f2, best_f2, atol=1e-12):
        if current_threshold < best_threshold: return True
        return parameter_key(current_parameters) < parameter_key(best_parameters)
    return False

# =========================================================
# Main training pipeline
# =========================================================

def train_pipeline() -> None:
    ensure_results_dir()
    X, y = load_and_prepare_data(TRAIN_FILE)
    split_idx = find_valid_temporal_split(y=y)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_validation, y_validation = X[split_idx:], y[split_idx:]

    best_model = None
    best_parameters = None
    best_threshold = None
    best_f2 = None

    parameter_combinations = list(generate_parameter_combinations())

    for index, parameters in enumerate(parameter_combinations, start=1):
        model = RandomForestClassifier(**parameters, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_validation)[:, 1]
        
        # Grid Search for best threshold
        for threshold in THRESHOLD_GRID:
            metrics = evaluate_predictions(y_validation, val_prob, threshold)
            
            if is_current_model_better(metrics["f2"], threshold, parameters, best_f2, best_threshold, best_parameters):
                best_model = model
                best_parameters = parameters.copy()
                best_threshold = float(threshold)
                best_f2 = float(metrics["f2"])

    joblib.dump(best_model, MODEL_PATH)
    print(f"[INFO] Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_pipeline()
