import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BETA = 2.0

DATASET_CANDIDATES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

RESULTS_DIR = (
    Path("results")
    / "ablation_crossday"
    / "logistic_regression"
    / "without_flow_duration"
)

ALL_MODEL_FEATURES = [
    "Packet_Count",
    "Packet_Size",
    "Flow_Rate",
    "Flow_Duration",
]

REMOVED_FEATURE = "Flow_Duration"

MODEL_FEATURES = [
    feature
    for feature in ALL_MODEL_FEATURES
    if feature != REMOVED_FEATURE
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
            matches = list(
                root.rglob(
                    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
                )
            )

            if matches:
                return matches[0]

    raise FileNotFoundError(
        "Dataset not found: "
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    )


def clean_column_names(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def create_binary_labels(label_series):
    labels = label_series.astype(str).str.strip().str.lower()

    return np.where(
        labels.str.contains("benign", na=False),
        0,
        1,
    ).astype(np.int8)


def prepare_features(df):
    missing_columns = [
        col
        for col in RAW_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Required raw columns are missing: "
            + ", ".join(missing_columns)
        )

    y = create_binary_labels(df["Label"])

    X_all_df = pd.DataFrame({
        "Packet_Count": (
            pd.to_numeric(
                df["Total Fwd Packets"],
                errors="coerce",
            )
            +
            pd.to_numeric(
                df["Total Backward Packets"],
                errors="coerce",
            )
        ),
        "Packet_Size": pd.to_numeric(
            df["Average Packet Size"],
            errors="coerce",
        ),
        "Flow_Rate": pd.to_numeric(
            df["Flow Packets/s"],
            errors="coerce",
        ),
        "Flow_Duration": pd.to_numeric(
            df["Flow Duration"],
            errors="coerce",
        ),
    })

    X_all_df = X_all_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # برای مقایسه منصفانه با مدل Full و سایر Ablationها،
    # تنها ردیف‌هایی حفظ می‌شوند که هر چهار ویژگی اصلی معتبر هستند.
    valid_rows = X_all_df.notna().all(axis=1)

    X_all_df = X_all_df.loc[
        valid_rows,
        ALL_MODEL_FEATURES,
    ]

    y = y[valid_rows.to_numpy()]

    # فقط سه ویژگی باقی‌مانده وارد مدل می‌شوند:
    # Packet_Count, Packet_Size, Flow_Rate
    X_df = X_all_df[MODEL_FEATURES]

    X = X_df.to_numpy(dtype=np.float32)

    if X.shape[0] == 0:
        raise ValueError(
            "No valid samples remained after preprocessing."
        )

    if X.shape[1] != 3:
        raise RuntimeError(
            f"Expected 3 features after removing "
            f"{REMOVED_FEATURE}, got {X.shape[1]}."
        )

    return X, y, MODEL_FEATURES.copy(), valid_rows


def calibrate_f2_threshold(y_true, y_probs, beta=2.0):
    y_true = np.asarray(y_true, dtype=np.int8)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    if len(y_true) != len(y_probs):
        raise ValueError(
            "y_true and y_probs must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calibrate threshold with empty arrays."
        )

    order = np.argsort(-y_probs, kind="mergesort")

    y_sorted = y_true[order]
    probs_sorted = y_probs[order]

    total_positives = np.sum(y_sorted, dtype=np.int64)

    if total_positives == 0:
        warnings.warn(
            "No positive samples were found. Using threshold=0.5."
        )

        return {
            "threshold": 0.5,
            "f2_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    tp = np.cumsum(y_sorted, dtype=np.int64)
    fp = np.cumsum(1 - y_sorted, dtype=np.int64)
    fn = total_positives - tp

    beta_squared = beta ** 2

    numerator = (1.0 + beta_squared) * tp

    denominator = (
        (1.0 + beta_squared) * tp
        + beta_squared * fn
        + fp
    )

    f_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )

    best_idx = int(np.argmax(f_scores))

    best_threshold = float(probs_sorted[best_idx])
    best_f2 = float(f_scores[best_idx])

    best_tp = int(tp[best_idx])
    best_fp = int(fp[best_idx])
    best_fn = int(fn[best_idx])

    precision = (
        best_tp / (best_tp + best_fp)
        if (best_tp + best_fp) > 0
        else 0.0
    )

    recall = (
        best_tp / (best_tp + best_fn)
        if (best_tp + best_fn) > 0
        else 0.0
    )

    return {
        "threshold": best_threshold,
        "f2_score": best_f2,
        "precision": precision,
        "recall": recall,
        "tp": best_tp,
        "fp": best_fp,
        "fn": best_fn,
    }


def train_cross_day_lr_ablation():
    start_time = time.time()

    print(
        "[*] Starting Logistic Regression Ablation Training "
        "on Friday dataset...",
        flush=True,
    )

    print(
        f"[*] Removed feature: {REMOVED_FEATURE}",
        flush=True,
    )

    print(
        f"[*] Remaining features: {', '.join(MODEL_FEATURES)}",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset()

    print(
        f"[+] Loading training data from: {dataset_path}",
        flush=True,
    )

    df = pd.read_csv(dataset_path, low_memory=False)

    print(
        f"[+] Raw dataset shape: {df.shape}",
        flush=True,
    )

    df = clean_column_names(df)

    X, y, feature_columns, valid_rows = prepare_features(df)

    print(
        f"[+] Valid samples after cleaning: {len(X)}",
        flush=True,
    )

    print(
        f"[+] Features count: {len(feature_columns)}",
        flush=True,
    )

    print(
        f"[+] Model features ({X.shape[1]} only): "
        f"{', '.join(feature_columns)}",
        flush=True,
    )

    print(
        f"[+] Removed feature: {REMOVED_FEATURE}",
        flush=True,
    )

    print(
        f"[+] Benign samples: {int(np.sum(y == 0))}",
        flush=True,
    )

    print(
        f"[+] Attack samples: {int(np.sum(y == 1))}",
        flush=True,
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Training data contains only one class. "
            "Logistic Regression requires at least two classes."
        )

    print(
        "[*] Scaling features with StandardScaler...",
        flush=True,
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X).astype(
        np.float32,
        copy=False,
    )

    print("[+] Feature scaling completed.", flush=True)

    print(
        "[*] Fitting Logistic Regression model (solver=lbfgs)...",
        flush=True,
    )

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

        model.fit(X_scaled, y)

    convergence_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, ConvergenceWarning)
    ]

    if convergence_warnings:
        print(
            "[WARNING] Logistic Regression did not fully converge. "
            "Consider increasing max_iter to 500.",
            flush=True,
        )

    print(
        f"[+] Training completed in "
        f"{model.n_iter_[0]} iteration(s).",
        flush=True,
    )

    print(
        "[*] Calibrating decision threshold using "
        "vectorized F2-score...",
        flush=True,
    )

    y_probs = model.predict_proba(X_scaled)[:, 1].astype(
        np.float64
    )

    calibration = calibrate_f2_threshold(
        y,
        y_probs,
        beta=BETA,
    )

    best_threshold = calibration["threshold"]

    print(
        f"[+] Optimal threshold: "
        f"{calibration['threshold']:.8f}",
        flush=True,
    )

    print(
        f"[+] Best F2-score: "
        f"{calibration['f2_score']:.6f}",
        flush=True,
    )

    print(
        f"[+] Calibration precision: "
        f"{calibration['precision']:.6f}",
        flush=True,
    )

    print(
        f"[+] Calibration recall: "
        f"{calibration['recall']:.6f}",
        flush=True,
    )

    model_path = (
        RESULTS_DIR
        / "lr_model_without_flow_duration.joblib"
    )

    scaler_path = (
        RESULTS_DIR
        / "scaler_lr_without_flow_duration.joblib"
    )

    metadata_path = (
        RESULTS_DIR
        / "lr_metadata_without_flow_duration.json"
    )

    summary_path = (
        RESULTS_DIR
        / "lr_training_summary_without_flow_duration.txt"
    )

    joblib.dump(model, model_path, compress=3)
    joblib.dump(scaler, scaler_path, compress=3)

    metadata = {
        "model_type": "LogisticRegression",
        "experiment_type": "leave_one_feature_out_ablation",
        "ablation_configuration": "without_flow_duration",
        "removed_feature": REMOVED_FEATURE,
        "solver": "lbfgs",
        "max_iter": 300,
        "tol": 1e-4,
        "C": 1.0,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "beta": BETA,
        "dataset": str(dataset_path),
        "dataset_type": "Friday training / Wednesday testing",
        "label_mapping": {
            "BENIGN": 0,
            "ALL_NON_BENIGN": 1,
        },
        "n_samples": int(len(X)),
        "n_features": int(len(feature_columns)),
        "n_benign": int(np.sum(y == 0)),
        "n_attack": int(np.sum(y == 1)),
        "all_features": ALL_MODEL_FEATURES,
        "used_features": feature_columns,
        "raw_required_columns": RAW_REQUIRED_COLUMNS,
        "best_threshold": float(best_threshold),
        "calibration": calibration,
        "valid_rows": int(np.sum(valid_rows.to_numpy())),
        "dropped_rows": int(
            len(valid_rows)
            - np.sum(valid_rows.to_numpy())
        ),
        "training_time_seconds": float(
            time.time() - start_time
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(
            "Logistic Regression Ablation Training Summary\n"
        )
        file.write("=" * 55 + "\n")
        file.write(f"Dataset: {dataset_path}\n")
        file.write(
            "Experiment: Leave-one-feature-out ablation\n"
        )
        file.write(
            f"Removed feature: {REMOVED_FEATURE}\n"
        )
        file.write(
            f"Used features: {', '.join(feature_columns)}\n"
        )
        file.write(f"Samples: {len(X)}\n")
        file.write(f"Features: {len(feature_columns)}\n")
        file.write(
            f"Benign samples: {int(np.sum(y == 0))}\n"
        )
        file.write(
            f"Attack samples: {int(np.sum(y == 1))}\n"
        )
        file.write("Solver: lbfgs\n")
        file.write("Max iterations: 300\n")
        file.write(
            f"Best threshold: {best_threshold:.8f}\n"
        )
        file.write(
            f"Best F2-score: "
            f"{calibration['f2_score']:.6f}\n"
        )
        file.write(
            f"Precision: "
            f"{calibration['precision']:.6f}\n"
        )
        file.write(
            f"Recall: "
            f"{calibration['recall']:.6f}\n"
        )
        file.write(
            f"Training time: "
            f"{time.time() - start_time:.2f} seconds\n"
        )

    print(f"[+] Saved model: {model_path}", flush=True)
    print(f"[+] Saved scaler: {scaler_path}", flush=True)
    print(f"[+] Saved metadata: {metadata_path}", flush=True)
    print(f"[+] Saved summary: {summary_path}", flush=True)

    print(
        f"[+] Total execution time: "
        f"{time.time() - start_time:.2f} seconds",
        flush=True,
    )

    print(
        "[+] Logistic Regression ablation training "
        "completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    train_cross_day_lr_ablation()
