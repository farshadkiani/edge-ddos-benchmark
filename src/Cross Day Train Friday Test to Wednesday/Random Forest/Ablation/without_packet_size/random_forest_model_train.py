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


# ============================================================
# Paths and constants
# ============================================================

TRAIN_FILE = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

RESULTS_DIR = "results"

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "random_forest_friday_model_without_packet_size.pkl"
)

METADATA_PATH = os.path.join(
    RESULTS_DIR,
    "train_metadata_rf_without_packet_size.json"
)

VAL_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "rf_friday_validation_metrics_without_packet_size.json"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    RESULTS_DIR,
    "feature_importance_rf_without_packet_size.json"
)


# ============================================================
# Ablation configuration
# Packet_Size is intentionally removed.
# ============================================================

FEATURE_COLUMNS = [
    "Packet_Count",
    "Flow_Rate",
    "Flow_Duration",
]


# Canonical/raw feature names after header normalization
RAW_REQUIRED_COLUMNS = [
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Packets/s",
    "Flow Duration",
    "Label",
]


RANDOM_STATE = 42
N_ESTIMATORS = 200
VAL_RATIO = 0.20


# ============================================================
# General helper functions
# ============================================================

def ensure_results_dir():
    """Create the output directory if it does not already exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(data, path):
    """Save Python dictionary/list data as formatted JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_column_name(column_name):
    """
    Normalize a CSV column name.

    Handles:
    - UTF-8 BOM character
    - leading/trailing spaces
    - multiple internal spaces
    - non-breaking spaces
    """
    column_name = str(column_name)

    column_name = column_name.replace("\ufeff", "")
    column_name = column_name.replace("\xa0", " ")
    column_name = " ".join(column_name.split())

    return column_name.strip()


def resolve_train_file():
    """
    Search for the Friday CSV file in common Docker/local locations.
    """
    candidate_paths = [
        TRAIN_FILE,
        os.path.join("/app", TRAIN_FILE),
        os.path.join("/mnt/data", TRAIN_FILE),
        os.path.join(os.getcwd(), TRAIN_FILE),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "\n[ERROR] Friday training dataset was not found.\n"
        f"Expected file name: {TRAIN_FILE}\n"
        "Checked paths:\n- "
        + "\n- ".join(candidate_paths)
    )


def detect_csv_format(file_path):
    """
    Detect CSV encoding and delimiter robustly.

    CICIDS2017 is normally comma-separated, but some downloaded or
    exported variants may use semicolons or tab delimiters.
    """
    encodings_to_try = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin1",
    ]

    separators_to_try = [
        ",",
        ";",
        "\t",
    ]

    best_result = None
    best_column_count = 0

    for encoding in encodings_to_try:
        for separator in separators_to_try:
            try:
                header_df = pd.read_csv(
                    file_path,
                    nrows=0,
                    sep=separator,
                    encoding=encoding,
                    engine="python",
                )

                column_count = len(header_df.columns)

                if column_count > best_column_count:
                    best_column_count = column_count
                    best_result = {
                        "encoding": encoding,
                        "separator": separator,
                        "raw_columns": list(header_df.columns),
                    }

            except Exception:
                continue

    if best_result is None or best_column_count <= 1:
        raise ValueError(
            "\n[ERROR] Could not reliably detect the CSV delimiter "
            "or encoding.\n"
            "Please verify that the input file is a valid CSV file."
        )

    return best_result


def get_required_raw_column_mapping(raw_columns):
    """
    Map canonical required names to the actual CSV names.

    Example:
        ' Total Fwd Packets ' -> 'Total Fwd Packets'

    Returns:
        dict:
            canonical_name -> actual_name_in_csv
    """
    normalized_to_actual = {}

    for raw_column in raw_columns:
        normalized_name = normalize_column_name(raw_column)

        if normalized_name not in normalized_to_actual:
            normalized_to_actual[normalized_name] = raw_column

    missing_columns = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in normalized_to_actual
    ]

    if missing_columns:
        available_columns = sorted(normalized_to_actual.keys())

        error_message = (
            "\n[ERROR] Required columns were not found in the CSV file.\n"
            f"Missing columns: {missing_columns}\n\n"
            "Available normalized columns detected in the file:\n"
        )

        for column in available_columns:
            error_message += f"  - {column}\n"

        raise ValueError(error_message)

    return {
        canonical_name: normalized_to_actual[canonical_name]
        for canonical_name in RAW_REQUIRED_COLUMNS
    }


# ============================================================
# Data loading and feature engineering
# ============================================================

def load_and_prepare_data(file_path):
    """
    Load Friday data and construct the selected ablation feature set.

    Retained features:
        - Packet_Count = Total Fwd Packets + Total Backward Packets
        - Flow_Rate = Flow Packets/s
        - Flow_Duration = Flow Duration

    Removed feature:
        - Packet_Size
    """
    print(f"[INFO] Loading Friday dataset: {file_path}")

    csv_info = detect_csv_format(file_path)

    encoding = csv_info["encoding"]
    separator = csv_info["separator"]
    raw_columns = csv_info["raw_columns"]

    separator_display = {
        ",": "comma (,)",
        ";": "semicolon (;)",
        "\t": "tab (\\t)",
    }.get(separator, repr(separator))

    print(f"[INFO] Detected encoding: {encoding}")
    print(f"[INFO] Detected delimiter: {separator_display}")
    print(f"[INFO] Number of detected CSV columns: {len(raw_columns)}")

    column_mapping = get_required_raw_column_mapping(raw_columns)

    actual_usecols = list(column_mapping.values())

    print("[INFO] Resolved source-column mapping:")

    for canonical_name, actual_name in column_mapping.items():
        print(f"  {canonical_name} -> {repr(actual_name)}")

    # Read only the required raw source columns.
    df = pd.read_csv(
        file_path,
        usecols=actual_usecols,
        sep=separator,
        encoding=encoding,
        engine="c",
        low_memory=False,
    )

    # Rename actual/raw file headers to canonical headers.
    rename_mapping = {
        actual_name: canonical_name
        for canonical_name, actual_name in column_mapping.items()
    }

    df.rename(columns=rename_mapping, inplace=True)

    # Ensure all canonical columns are present after rename.
    missing_after_rename = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_after_rename:
        raise ValueError(
            "\n[ERROR] Required columns are missing after CSV loading "
            "and normalization.\n"
            f"Missing: {missing_after_rename}\n"
            f"Loaded columns: {list(df.columns)}"
        )

    # Convert numeric source columns.
    fwd_packets = pd.to_numeric(
        df["Total Fwd Packets"],
        errors="coerce",
    )

    bwd_packets = pd.to_numeric(
        df["Total Backward Packets"],
        errors="coerce",
    )

    flow_rate = pd.to_numeric(
        df["Flow Packets/s"],
        errors="coerce",
    )

    flow_duration = pd.to_numeric(
        df["Flow Duration"],
        errors="coerce",
    )

    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Feature engineering for the "without Packet_Size" variant
    # --------------------------------------------------------
    prepared = pd.DataFrame({
        "Packet_Count": fwd_packets + bwd_packets,
        "Flow_Rate": flow_rate,
        "Flow_Duration": flow_duration,
        "Label": labels,
    })

    # Replace infinity introduced by Flow Packets/s values.
    prepared.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    before_drop = len(prepared)

    prepared.dropna(
        subset=FEATURE_COLUMNS + ["Label"],
        inplace=True,
    )

    prepared.reset_index(drop=True, inplace=True)

    after_drop = len(prepared)
    dropped_rows = before_drop - after_drop

    X = prepared[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    # CICIDS2017 labeling:
    # BENIGN -> 0
    # any non-BENIGN label (DDoS here) -> 1
    y = (
        prepared["Label"].to_numpy() != "BENIGN"
    ).astype(np.int32)

    stats = {
        "n_rows_raw": int(before_drop),
        "n_rows_clean": int(after_drop),
        "dropped_rows": int(dropped_rows),
        "n_benign": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
        "detected_encoding": encoding,
        "detected_delimiter": separator_display,
    }

    return X, y, stats


# ============================================================
# Temporal validation split
# ============================================================

def find_valid_temporal_split(
    y,
    val_ratio=0.20,
    min_train_ratio=0.55,
    min_val_ratio=0.10,
):
    """
    Find a contiguous temporal split while ensuring both partitions
    contain both binary classes.

    No shuffling is performed.
    """
    n = len(y)

    start_idx = int(n * min_train_ratio)
    end_idx = int(n * (1.0 - min_val_ratio))

    if start_idx >= end_idx:
        raise ValueError(
            "Invalid temporal split range. "
            "Check dataset size and split ratios."
        )

    candidate_idx = int(n * (1.0 - val_ratio))

    candidate_idx = max(
        start_idx,
        min(candidate_idx, end_idx),
    )

    def has_both_classes(values):
        return len(np.unique(values)) >= 2

    if (
        has_both_classes(y[:candidate_idx])
        and has_both_classes(y[candidate_idx:])
    ):
        return candidate_idx

    # If the target split does not contain both classes,
    # search a valid contiguous alternative.
    for split_idx in range(start_idx, end_idx + 1):
        if (
            has_both_classes(y[:split_idx])
            and has_both_classes(y[split_idx:])
        ):
            return split_idx

    raise ValueError(
        "Could not find a contiguous temporal split where both "
        "training and validation sets contain both classes."
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true, y_prob, threshold=0.50):
    """
    Calculate classification and ranking metrics.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "f2": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2,
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Main training pipeline
# ============================================================

def train_pipeline():
    ensure_results_dir()

    train_file = resolve_train_file()

    X, y, data_stats = load_and_prepare_data(train_file)

    split_idx = find_valid_temporal_split(
        y,
        val_ratio=VAL_RATIO,
    )

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    print("\n[INFO] Dataset summary:")
    print(f"  Clean samples: {len(X)}")
    print(f"  Dropped rows: {data_stats['dropped_rows']}")
    print(f"  BENIGN samples: {data_stats['n_benign']}")
    print(f"  ATTACK samples: {data_stats['n_attack']}")
    print(f"  Temporal split index: {split_idx}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    print("\n[INFO] Training Random Forest model...")
    print("[INFO] Ablation setting: without Packet_Size")
    print(f"[INFO] Retained features: {FEATURE_COLUMNS}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0,
    )

    train_start_time = time.time()

    model.fit(X_train, y_train)

    training_time_sec = time.time() - train_start_time

    print(
        f"[INFO] Training completed in "
        f"{training_time_sec:.2f} seconds."
    )

    # Validation inference at default threshold tau = 0.50.
    validation_start_time = time.time()

    val_prob = model.predict_proba(X_val)[:, 1]

    validation_total_inference_time_sec = (
        time.time() - validation_start_time
    )

    validation_avg_latency_ms_per_sample = (
        validation_total_inference_time_sec
        / max(len(X_val), 1)
    ) * 1000.0

    val_metrics = evaluate_predictions(
        y_true=y_val,
        y_prob=val_prob,
        threshold=0.50,
    )

    val_metrics["validation_total_inference_time_sec"] = float(
        validation_total_inference_time_sec
    )

    val_metrics["validation_avg_latency_ms_per_sample"] = float(
        validation_avg_latency_ms_per_sample
    )

    # Save trained model.
    joblib.dump(model, MODEL_PATH)

    print(f"[INFO] Model saved: {MODEL_PATH}")

    # Save metadata.
    metadata = {
        "train_file": train_file,
        "results_dir": RESULTS_DIR,
        "model_path": MODEL_PATH,
        "model_type": "RandomForestClassifier",
        "ablation_setting": "without_packet_size",
        "removed_feature": "Packet_Size",
        "retained_features": FEATURE_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": -1,
        "data_stats": data_stats,
        "temporal_split": {
            "method": "contiguous_temporal_split",
            "validation_ratio_target": VAL_RATIO,
            "split_index": int(split_idx),
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_val)),
        },
        "training_time_sec": float(training_time_sec),
    }

    save_json(metadata, METADATA_PATH)

    print(f"[INFO] Metadata saved: {METADATA_PATH}")

    # Save validation metrics.
    save_json(val_metrics, VAL_METRICS_PATH)

    print(f"[INFO] Validation metrics saved: {VAL_METRICS_PATH}")

    # Save feature importance.
    feature_importance = dict(
        zip(
            FEATURE_COLUMNS,
            model.feature_importances_.tolist(),
        )
    )

    save_json(
        feature_importance,
        FEATURE_IMPORTANCE_PATH,
    )

    print(
        "[INFO] Feature importance saved: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    print("\n============================================================")
    print("[INFO] Friday validation metrics at threshold = 0.50")
    print("============================================================")

    for key, value in val_metrics.items():
        print(f"{key}: {value}")

    print("\n[INFO] Random Forest training pipeline completed.")
    print("[INFO] Ablation setting: Packet_Size removed.")


if __name__ == "__main__":
    train_pipeline()
