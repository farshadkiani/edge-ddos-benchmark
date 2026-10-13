import os
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
REMOVED_FEATURE = "Packet_Size"
OUTPUT_DIR = os.path.join("results", "xgb_without_packet_size")

# Aliases support both engineered and CICIDS2017 raw column names.
FEATURE_ALIASES = {
    "Packet_Count": ["Packet_Count", "Total Fwd Packets", "Total Backward Packets"],
    "Packet_Size": ["Packet_Size", "Average Packet Size", "Packet Length Mean",
                     "Total Length of Fwd Packets", "Total Length of Bwd Packets"],
    "Flow_Rate": ["Flow_Rate", "Flow Bytes/s", "Flow Packets/s"],
    "Flow_Duration": ["Flow_Duration", "Flow Duration"],
}

def clean_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    drop_cols = ["Flow ID", "Source IP", "Source Port", "Destination IP",
                 "Destination Port", "Protocol", "Timestamp"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if "Label" not in df.columns:
        raise ValueError("Column 'Label' not found in dataset.")
    df["Label"] = df["Label"].apply(lambda x: 1 if "ddos" in str(x).lower() else 0)
    return df

def get_columns_to_remove(columns, feature_group):
    candidates = FEATURE_ALIASES[feature_group]
    found = [c for c in candidates if c in columns]
    if not found:
        raise ValueError(f"No columns found for {feature_group}. Available columns: {list(columns)}")
    return found

def main():
    data_path = DATA_PATH if os.path.exists(DATA_PATH) else os.path.join("/mnt/data", DATA_PATH)
    df = clean_and_prepare_data(data_path)
    X = df.drop(columns=["Label"])
    y = df["Label"].to_numpy()
    removed_columns = get_columns_to_remove(X.columns, REMOVED_FEATURE)
    X = X.drop(columns=removed_columns)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    model = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100,
        verbosity=1, n_jobs=-1, random_state=42, tree_method="hist", eval_metric="logloss")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    probs = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f2 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        preds = (probs >= threshold).astype(int)
        score = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if score > best_f2:
            best_threshold, best_f2 = float(threshold), float(score)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(OUTPUT_DIR, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_xgb.joblib"))
    metadata = {"model":"XGBoost", "removed_feature":REMOVED_FEATURE,
        "removed_columns":removed_columns, "features":list(X.columns),
        "best_threshold":best_threshold, "validation_f2_score":best_f2,
        "training_time_seconds":train_time, "temporal_split":"80% train / 20% validation"}
    with open(os.path.join(OUTPUT_DIR, "xgb_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Completed: removed={REMOVED_FEATURE}, threshold={best_threshold:.4f}, F2={best_f2:.4f}")

if __name__ == "__main__":
    main()
