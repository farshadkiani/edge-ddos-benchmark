# train_friday_xgb.py
import os
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler
import joblib  # این کتابخانه برای ذخیره و بارگذاری مدل کافی است

def clean_and_prepare_data(filepath):
    print(f"[*] Loading training data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # استانداردسازی نام ستون‌ها (حذف فضاهای خالی)
    df.columns = df.columns.str.strip()
    
    # حذف ستون‌های شناسه و زمانی غیرمرتبط با یادگیری ماشین
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # جایگزینی مقادیر بی‌نهایت و نامعتبر با NaN و سپس پر کردن یا حذف آن‌ها
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # کدگذاری برچسب‌ها (DDoS به 1 و Benign به 0)
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 1 if 'ddos' in str(x).lower() else 0)
    else:
        raise ValueError("Column 'Label' not found in dataset.")
        
    return df

def main():
    # مسیر داده ورودی (روز جمعه)
    data_path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join("/mnt/data", data_path)
        
    df = clean_and_prepare_data(data_path)
    
    X = df.drop(columns=['Label'])
    y = df['Label'].values
    
    # ۱. تقسیم‌بندی زمانی منصفانه (Temporal Split) - ۸۰ درصد آموزش، ۲۰ درصد اعتبارسنجی
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"[*] Data split completed:")
    print(f"    Train size: {X_train.shape[0]} samples")
    print(f"    Validation size: {X_val.shape[0]} samples")
    
    # نرمال‌سازی داده‌ها
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # ذخیره آبجکت scaler برای فاز تست
    os.makedirs("results", exist_ok=True)
    joblib.dump(scaler, "results/scaler_xgb.joblib")
    
    # ۲. آموزش مدل XGBoost با تنظیمات بهینه برای Edge (عمق درخت محدود برای سرعت بالا)
    print("[*] Training XGBoost model...")
    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        verbosity=1,
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_duration = time.time() - start_time
    print(f"[+] Training completed in {train_duration:.2f} seconds.")
    
    # ۳. کالیبراسیون آستانه تصمیم‌گیری (Threshold Calibration) بر اساس F2-score روی داده‌های Validation
    print("[*] Calibrating decision threshold based on F2-score...")
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    
    best_threshold = 0.5
    best_f2 = 0.0
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t
            
    print(f"[+] Optimal threshold found: {best_threshold:.4f} (F2-score: {best_f2:.4f})")
    
    # ذخیره مدل و متادیتا
    joblib.dump(model, "results/xgb_model.joblib")
    
    metadata = {
        "best_threshold": float(best_threshold),
        "validation_f2_score": float(best_f2),
        "training_time_seconds": float(train_duration),
        "features": list(X.columns)
    }
    
    with open("results/xgb_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("[+] Model, scaler, and metadata saved to 'results/' directory successfully.")

if __name__ == "__main__":
    main()
