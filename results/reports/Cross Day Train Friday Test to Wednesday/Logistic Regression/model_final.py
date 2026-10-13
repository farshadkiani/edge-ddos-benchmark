import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BETA = 2.0
VAL_RATIO = 0.20

DATASET_CANDIDATES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

RESULTS_DIR = Path("results")

MODEL_FEATURES = [
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


def find_dataset():
    for candidate in DATASET_CANDIDATES:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path

    search_roots = [Path("."), Path("/app"), Path("/data")]

    for root in search_roots:
        if root.exists():
            matches = list(root.rglob("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        "Dataset not found: Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def create_binary_labels(label_series):
    labels = label_series.astype(str).str.strip().str.upper()
    return np.where(labels == "BENIGN", 0, 1).astype(np.int8)


def prepare_features(df):
    missing_columns = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "Required raw columns are missing: " + ", ".join(missing_columns)
        )

    y = create_binary_labels(df["Label"])

    X_df = pd.DataFrame({
        "Packet_Count": (
            pd.to_numeric(df["Total Fwd Packets"], errors="coerce")
            + pd.to_numeric(df["Total Backward Packets"], errors="coerce")
        ),
        "Packet_Size": pd.to_numeric(df["Average Packet Size"], errors="coerce"),
        "Flow_Rate": pd.to_numeric(df["Flow Packets/s"], errors="coerce"),
        "Flow_Duration": pd.to_numeric(df["Flow Duration"], errors="coerce"),
    })

    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    valid_rows = X_df.notna().all(axis=1)
    X_df = X_df.loc[valid_rows, MODEL_FEATURES]
    y = y[valid_rows.to_numpy()]

    X = X_df.to_numpy(dtype=np.float32)

    if X.shape[0] == 0:
        raise ValueError("No valid samples remained after preprocessing.")

    if X.shape[1] != 4:
        raise RuntimeError(f"Expected 4 features, got {X.shape[1]}.")

    return X, y, MODEL_FEATURES.copy(), valid_rows


def find_valid_temporal_split(y, target_ratio=0.80, min_train_ratio=0.55, max_train_ratio=0.90):
    y = np.asarray(y, dtype=np.int8)
    n = len(y)

    if n < 2:
        raise ValueError("Not enough samples to create a temporal split.")

    target_idx = int(n * target_ratio)
    target_idx = max(1, min(n - 1, target_idx))

    min_idx = max(1, int(n * min_train_ratio))
    max_idx = min(n - 1, int(n * max_train_ratio))

    def is_valid_split(split_idx):
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        return (
            len(y_train) > 0
            and len(y_val) > 0
            and len(np.unique(y_train)) >= 2
            and len(np.unique(y_val)) >= 2
        )

    if is_valid_split(target_idx):
        return target_idx

    for split_idx in range(min_idx, max_idx + 1):
        if is_valid_split(split_idx):
            return split_idx

    raise ValueError(
        "No valid temporal split found where both train and validation contain both classes."
    )


def evaluate_predictions(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    y_pred = (y_prob >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_squared = BETA ** 2
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = (
        (1.0 + beta_squared) * precision * recall /
        (beta_squared * precision + recall)
        if (beta_squared * precision + recall) > 0 else 0.0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "mcc": float(mcc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def calibrate_f2_threshold(y_true, y_probs, beta=2.0):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot calibrate threshold with empty arrays.")

    thresholds = np.linspace(0.01, 0.99, 99)

    best_metrics = None
    best_f2 = -1.0

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        beta_squared = beta ** 2
        f2 = (
            (1.0 + beta_squared) * precision * recall /
            (beta_squared * precision + recall)
            if (beta_squared * precision + recall) > 0 else 0.0
        )

        # strict > : in ties, smaller threshold stays selected
        if f2 > best_f2:
            best_f2 = f2
            best_metrics = {
                "threshold": float(threshold),
                "f2_score": float(f2),
                "precision": float(precision),
                "recall": float(recall),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }

    if best_metrics is None:
        raise RuntimeError("Threshold calibration failed.")

    return best_metrics


def train_cross_day_lr():
    start_time = time.time()

    print("[*] Starting Logistic Regression Training on Friday dataset...", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()
    print(f"[+] Loading training data from: {dataset_path}", flush=True)

    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"[+] Raw dataset shape: {df.shape}", flush=True)

    df = clean_column_names(df)

    X, y, feature_columns, valid_rows = prepare_features(df)

    print(f"[+] Valid samples after cleaning: {len(X)}", flush=True)
    print(f"[+] Features count: {len(feature_columns)}", flush=True)
    print(f"[+] Model features ({X.shape[1]} only): {', '.join(feature_columns)}", flush=True)
    print(f"[+] Benign samples: {int(np.sum(y == 0))}", flush=True)
    print(f"[+] Attack samples: {int(np.sum(y == 1))}", flush=True)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Training data contains only one class. Logistic Regression requires at least two classes."
        )

    split_idx = find_valid_temporal_split(y, target_ratio=1.0 - VAL_RATIO)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    print(f"[+] Temporal split index: {split_idx}", flush=True)
    print(f"[+] Train samples: {len(X_train)}", flush=True)
    print(f"[+] Validation samples: {len(X_val)}", flush=True)
    print(f"[+] Train benign: {int(np.sum(y_train == 0))}, train attack: {int(np.sum(y_train == 1))}", flush=True)
    print(f"[+] Val benign: {int(np.sum(y_val == 0))}, val attack: {int(np.sum(y_val == 1))}", flush=True)

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training split contains only one class. Logistic Regression requires at least two classes."
        )
    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation split contains only one class. Threshold calibration requires both classes."
        )

    print("[*] Scaling features with StandardScaler on train split only...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val_scaled = scaler.transform(X_val).astype(np.float32, copy=False)
    print("[+] Feature scaling completed.", flush=True)

    print("[*] Fitting Logistic Regression model (solver=lbfgs)...", flush=True)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=300,
            tol=1e-4,
            C=1.0,
            class_weight=None,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_scaled, y_train)

    convergence_warnings = [
        warning for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence_warnings:
        print(
            "[WARNING] Logistic Regression did not fully converge. Consider increasing max_iter to 500.",
            flush=True,
        )

    print(f"[+] Training completed in {model.n_iter_[0]} iteration(s).", flush=True)

    print("[*] Calibrating decision threshold on validation split using F2-score...", flush=True)
    y_val_probs = model.predict_proba(X_val_scaled)[:, 1].astype(np.float64)

    calibration = calibrate_f2_threshold(y_val, y_val_probs, beta=BETA)
    best_threshold = calibration["threshold"]

    print(f"[+] Optimal threshold: {calibration['threshold']:.8f}", flush=True)
    print(f"[+] Best F2-score: {calibration['f2_score']:.6f}", flush=True)
    print(f"[+] Calibration precision: {calibration['precision']:.6f}", flush=True)
    print(f"[+] Calibration recall: {calibration['recall']:.6f}", flush=True)

    val_metrics_default = evaluate_predictions(y_val, y_val_probs, threshold=0.50)
    val_metrics_calibrated = evaluate_predictions(y_val, y_val_probs, threshold=best_threshold)

    model_path = RESULTS_DIR / "lr_model.joblib"
    scaler_path = RESULTS_DIR / "scaler_lr.joblib"
    metadata_path = RESULTS_DIR / "lr_metadata.json"
    summary_path = RESULTS_DIR / "lr_training_summary.txt"

    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    metadata = {
        "model_type": "LogisticRegression",
        "solver": "lbfgs",
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": RANDOM_STATE,
        "beta": BETA,
        "dataset": str(dataset_path),
        "dataset_type": "Friday training / Wednesday testing",
        "temporal_split_method": "contiguous 80/20 split on Friday clean rows",
        "target_validation_ratio": VAL_RATIO,
        "split_index": int(split_idx),
        "label_mapping": {
            "BENIGN": 0,
            "ALL_NON_BENIGN": 1,
        },
        "n_samples": int(len(X)),
        "n_features": int(len(feature_columns)),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_benign": int(np.sum(y == 0)),
        "n_attack": int(np.sum(y == 1)),
        "n_train_benign": int(np.sum(y_train == 0)),
        "n_train_attack": int(np.sum(y_train == 1)),
        "n_val_benign": int(np.sum(y_val == 0)),
        "n_val_attack": int(np.sum(y_val == 1)),
        "features": feature_columns,
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "best_threshold": float(best_threshold),
        "calibration": calibration,
        "validation_metrics_default_0_50": val_metrics_default,
        "validation_metrics_calibrated": val_metrics_calibrated,
        "valid_rows": int(np.sum(valid_rows.to_numpy())),
        "dropped_rows": int(len(valid_rows) - np.sum(valid_rows.to_numpy())),
        "training_time_seconds": float(time.time() - start_time),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("Logistic Regression Training Summary\n")
        file.write("=" * 45 + "\n")
        file.write(f"Dataset: {dataset_path}\n")
        file.write(f"Samples: {len(X)}\n")
        file.write(f"Train samples: {len(X_train)}\n")
        file.write(f"Validation samples: {len(X_val)}\n")
        file.write(f"Features: {len(feature_columns)}\n")
        file.write(f"Feature names: {', '.join(feature_columns)}\n")
        file.write(f"Benign samples: {int(np.sum(y == 0))}\n")
        file.write(f"Attack samples: {int(np.sum(y == 1))}\n")
        file.write(f"Train benign: {int(np.sum(y_train == 0))}\n")
        file.write(f"Train attack: {int(np.sum(y_train == 1))}\n")
        file.write(f"Val benign: {int(np.sum(y_val == 0))}\n")
        file.write(f"Val attack: {int(np.sum(y_val == 1))}\n")
        file.write("Solver: lbfgs\n")
        file.write("Max iterations: 300\n")
        file.write(f"Temporal split index: {split_idx}\n")
        file.write(f"Best threshold: {best_threshold:.8f}\n")
        file.write(f"Best F2-score: {calibration['f2_score']:.6f}\n")
        file.write(f"Precision: {calibration['precision']:.6f}\n")
        file.write(f"Recall: {calibration['recall']:.6f}\n")
        file.write(f"Training time: {time.time() - start_time:.2f} seconds\n")

    print(f"[+] Saved model: {model_path}", flush=True)
    print(f"[+] Saved scaler: {scaler_path}", flush=True)
    print(f"[+] Saved metadata: {metadata_path}", flush=True)
    print(f"[+] Saved summary: {summary_path}", flush=True)
    print(f"[+] Total execution time: {time.time() - start_time:.2f} seconds", flush=True)
    print("[+] Logistic Regression training completed successfully.", flush=True)


if __name__ == "__main__":
    train_cross_day_lr()
