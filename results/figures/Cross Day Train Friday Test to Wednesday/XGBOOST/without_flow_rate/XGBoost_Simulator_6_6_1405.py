import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve, matthews_corrcoef
)

# ===================== EXPERIMENT CONFIGURATION =====================
ABLATION_NAME = "without_flow_rate"
MODEL_DIR = os.path.join("results", "xgb_" + ABLATION_NAME)
OUTPUT_DIR = os.path.join("results", "xgb_" + ABLATION_NAME + "_wednesday")
TEST_FILENAME = "Wednesday-workingHours.pcap_ISCX.csv"
CHUNK_SIZE = 50000
LATENCY_SAMPLE_COUNT = 5000

# ==============================================================================
# 1. Global Matplotlib Academic Styling (Times New Roman, Font Size = 10, DPI = 300)
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "savefig.dpi": 300,
    "figure.dpi": 300
})


def get_resource_usage():
    """محاسبه حافظه اشغالی پروسه (MB) و درصد استفاده از CPU."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=None)
    return mem_mb, cpu_percent


def compute_metrics(y_true, y_probs, threshold, roc_auc_val):
    """محاسبه معیارهای هفت‌گانه طبقه‌بندی و مقادیر ماتریس درهم‌ریختگی."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        "threshold": float(threshold),
        "y_pred": y_pred,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2-Score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "Specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_val),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }


def save_confusion_matrix(y_true, y_pred, title, caption_text, path, cmap):
    """رسم ماتریس درهم‌ریختگی با استانداردهای دانشگاهی."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap, cbar=False,
        annot_kws={"size": 10},
        xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'], ax=ax
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metric_bar(metrics, title, caption_text, path):
    """رسم نمودار میله‌ای معیارهای عملکرد."""
    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'MCC', 'ROC-AUC']
    vals = [metrics[x] for x in names]
    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#34495e', '#f39c12']
    
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    bars = ax.bar(names, vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score (Normalized [0, 1])', fontsize=10)
    ax.set_xlabel('Evaluation Metric', fontsize=10)
    ax.set_title(title, fontsize=10)
    
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=10)
        
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_threshold_comparison_metrics(default_metrics, calibrated_metrics, cal_thresh, caption_text, path):
    """رسم نمودار میله‌ای مقایسه‌ای آستانه پیش‌فرض در برابر آستانه کالیبره‌شده."""
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    vals_default = [default_metrics[name] for name in names]
    vals_calibrated = [calibrated_metrics[name] for name in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    bars1 = ax.bar(x - width / 2, vals_default, width, label='Default Threshold (0.50)', color='#3498db', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, vals_calibrated, width, label=f'Calibrated Threshold ({cal_thresh:.4f})', color='#e67e22', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Score (Normalized [0, 1])', fontsize=10)
    ax.set_xlabel('Evaluation Metric', fontsize=10)
    ax.set_title('Performance Comparison: Default vs. Calibrated Threshold - XGBoost', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim([0.0, 1.12])
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.015,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9
            )

    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def save_probability_distribution(y_true, y_probs, threshold, is_calibrated, path):
    """رسم هیستوگرام توزیع احتمال پیش‌بینی به تفکیک کلاس."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    
    benign_probs = y_probs[y_true == 0]
    ddos_probs = y_probs[y_true == 1]

    sns.histplot(benign_probs, bins=40, kde=True, color='#34495e', label='Normal (Benign)', stat="count", ax=ax, alpha=0.5, edgecolor='black')
    sns.histplot(ddos_probs, bins=40, kde=True, color='#e74c3c', label='DDoS Attack', stat="count", ax=ax, alpha=0.5, edgecolor='black')

    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, label=f'Decision Threshold = {threshold:.4f}')
    
    ax.set_xlabel('Predicted Attack Probability', fontsize=10)
    ax.set_ylabel('Sample Count', fontsize=10)
    ax.set_xlim([-0.05, 1.05])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)

    threshold_desc = f"Calibrated Threshold = {threshold:.4f}" if is_calibrated else f"Threshold = {threshold:.4f}"
    caption_text = f"Figure: Predicted probability distribution by class ({threshold_desc})."
    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    def model_path(name): return os.path.join(MODEL_DIR, name)
    def output_path(name): return os.path.join(OUTPUT_DIR, name)

    print(f"[*] Loading model and artifacts from: {MODEL_DIR}")
    model = joblib.load(model_path('xgb_model.joblib'))
    scaler = joblib.load(model_path('scaler_xgb.joblib'))
    with open(model_path('xgb_metadata.json'), encoding='utf-8') as f:
        metadata = json.load(f)

    calibrated_threshold = float(metadata.get('best_threshold', 0.5))
    features = metadata['features']
    
    # پیدا کردن مسیر فایل داده‌های روز چهارشنبه
    candidates = [
        TEST_FILENAME,
        os.path.join("..", TEST_FILENAME),
        os.path.join("data", TEST_FILENAME),
        os.path.join("/mnt/data", TEST_FILENAME)
    ]
    test_path = None
    for cand in candidates:
        if os.path.exists(cand):
            test_path = cand
            break

    if not test_path:
        raise FileNotFoundError(f"Wednesday test dataset '{TEST_FILENAME}' not found.")

    print(f"[*] Evaluating {ABLATION_NAME} on 100% of Wednesday dataset using chunks...")

    y_true_all, probs_all = [], []
    memory_profile, cpu_profile, chunk_times = [], [], []
    latency_benchmark_samples = []

    # بخش اول: استنباط دسته‌ای سریع روی ۱۰۰٪ داده‌ها جهت محاسبه متریک‌های کیفی
    for idx, chunk in enumerate(pd.read_csv(test_path, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        if 'Label' not in chunk.columns:
            continue
        
        y = chunk['Label'].apply(lambda x: 0 if 'benign' in str(x).lower() else 1).values
        missing = [c for c in features if c not in chunk.columns]
        if missing:
            raise KeyError(f'Missing test features: {missing}')
        
        chunk_X = chunk[features].values
        chunk_X_scaled = scaler.transform(chunk_X)
        
        start_chunk_time = time.perf_counter()
        probs = model.predict_proba(chunk_X_scaled)[:, 1]
        chunk_dt = time.perf_counter() - start_chunk_time
        
        mem, cpu = get_resource_usage()
        memory_profile.append(mem)
        cpu_profile.append(cpu)
        chunk_times.append(chunk_dt)
        
        y_true_all.extend(y)
        probs_all.extend(probs)
        
        # ذخیره نمونه‌ها برای بنچمارک دقیق تأخیر تک‌نمونه‌ای
        if len(latency_benchmark_samples) < LATENCY_SAMPLE_COUNT:
            needed = LATENCY_SAMPLE_COUNT - len(latency_benchmark_samples)
            latency_benchmark_samples.extend(chunk_X_scaled[:needed])
            
        print(f"    Chunk {idx + 1:02d} | Rows Processed: {len(y_true_all):,} | RSS Memory: {mem:.2f} MB | CPU: {cpu:.1f}%")

    # بخش دوم: بنچمارک تأخیر استنباط تکی (Pure Single-Sample Latency با Batch Size = 1)
    print(f"\n[*] Running Pure Single-Sample Latency Benchmark on {len(latency_benchmark_samples)} samples (Batch Size = 1)...")
    
    # مرحله Warm-up برای رفع اثر Cold Start پردازنده
    warmup_dummy = np.zeros((1, len(features)))
    for _ in range(50):
        _ = model.predict_proba(warmup_dummy)

    sample_latencies_ms = []
    for x_vec in latency_benchmark_samples:
        x_single = x_vec.reshape(1, -1)
        
        t0 = time.perf_counter()
        _ = model.predict_proba(x_single)
        t1 = time.perf_counter()
        
        sample_latencies_ms.append((t1 - t0) * 1000.0)

    y_true = np.asarray(y_true_all)
    probs = np.asarray(probs_all)
    sample_latencies_ms = np.asarray(sample_latencies_ms)

    # بخش سوم: محاسبه متریک‌ها و صدک‌های تأخیر
    roc_auc = roc_auc_score(y_true, probs)
    p, r, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(r, p)
    fpr, tpr, _ = roc_curve(y_true, probs)
    
    d = compute_metrics(y_true, probs, 0.5, roc_auc)
    c = compute_metrics(y_true, probs, calibrated_threshold, roc_auc)

    latency_stats = {
        'Mean-Latency-ms': float(np.mean(sample_latencies_ms)),
        'P50-Latency-ms': float(np.percentile(sample_latencies_ms, 50)),
        'P90-Latency-ms': float(np.percentile(sample_latencies_ms, 90)),
        'P95-Latency-ms': float(np.percentile(sample_latencies_ms, 95)),
        'P99-Latency-ms': float(np.percentile(sample_latencies_ms, 99)),
    }

    shared = {
        'ROC-AUC': float(roc_auc),
        'PR-AUC': float(pr_auc),
        **latency_stats,
        'Peak-Memory-MB': float(np.max(memory_profile)),
        'Avg-CPU-Usage-Percent': float(np.mean(cpu_profile))
    }

    results = {
        'experiment': ABLATION_NAME,
        'features': features,
        'default_threshold_0.5': {k: v for k, v in {**d, **shared}.items() if k != 'y_pred'},
        'calibrated_threshold': {k: v for k, v in {**c, **shared}.items() if k != 'y_pred'}
    }
    
    json_path = output_path('xgb_wednesday_crossday_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # ==============================================================================
    # 4. ترسیم و ذخیره‌سازی نمودارها با رعایت استانداردهای گرافیکی
    # ==============================================================================
    
    # ۱. ماتریس‌های درهم‌ریختگی
    save_confusion_matrix(
        y_true, d['y_pred'], 'Confusion Matrix - Default Threshold (0.50)',
        'Figure: Confusion matrix of XGBoost under default threshold (0.50).',
        output_path('confusion_matrix_xgb_default.png'), 'Blues'
    )
    save_confusion_matrix(
        y_true, c['y_pred'], f'Confusion Matrix - Calibrated Threshold (T={calibrated_threshold:.4f})',
        'Figure: Confusion matrix of XGBoost under calibrated operating threshold.',
        output_path('confusion_matrix_xgb_calibrated.png'), 'Oranges'
    )
    
    # ۲. نمودارهای میله‌ای معیارهای هفت‌گانه
    save_metric_bar(
        d, 'Performance Metrics - Default Threshold (0.50)',
        'Figure: Performance metrics summary for default threshold (0.50).',
        output_path('metrics_bar_chart_xgb_default.png')
    )
    save_metric_bar(
        c, f'Performance Metrics - Calibrated Threshold (T={calibrated_threshold:.4f})',
        f'Figure: Performance metrics summary for calibrated threshold (T={calibrated_threshold:.4f}).',
        output_path('metrics_bar_chart_xgb_calibrated.png')
    )
    
    # ۳. مقایسه مستقیم میله‌ای دو آستانه
    save_threshold_comparison_metrics(
        d, c, calibrated_threshold,
        'Figure: Side-by-side metric comparison between default and calibrated thresholds for XGBoost.',
        output_path('metrics_bar_chart_xgb.png')
    )

    # ۴. منحنی‌های ROC و PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Baseline')
    axes[0].plot(fpr, tpr, color='#28536b', lw=2.0, label=f'XGBoost (AUC = {roc_auc:.4f})')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=10)
    axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=10)
    axes[0].set_title('(a) Receiver Operating Characteristic (ROC)', fontsize=10)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    axes[1].plot(r, p, color='#19647e', lw=2.0, label=f'XGBoost (PR-AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=10)
    axes[1].set_ylabel('Precision', fontsize=10)
    axes[1].set_title('(b) Precision-Recall (PR) Curve', fontsize=10)
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    fig.text(0.5, -0.05, 'Figure: Discrimination performance (ROC and PR curves) on Wednesday cross-day dataset.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path('xgb_wednesday_roc_pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ۵. نمودار صدک‌های تأخیر
    percentiles = [50, 90, 95, 99]
    perc_vals = [latency_stats[f"P{p}-Latency-ms"] for p in percentiles]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f"P{p}" for p in percentiles], perc_vals, color='#1abc9c', edgecolor='black', linewidth=0.8, width=0.45)
    ax.set_ylabel('Inference Latency (ms)', fontsize=10)
    ax.set_xlabel('Latency Percentile', fontsize=10)
    ax.set_title('Single-Sample Pure Inference Latency Percentiles (Batch Size = 1, N = 5000)', fontsize=10)
    for bar, val in zip(bars, perc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., val + 0.01 * max(perc_vals),
                f'{val:.4f} ms', ha='center', va='bottom', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    fig.text(0.5, -0.05, 'Figure: Tail latency analysis across percentiles (P50, P90, P95, P99) for single-sample inference.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path('latency_percentiles_xgb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ۶. نمودار توزیع احتمال (Default & Calibrated)
    save_probability_distribution(
        y_true, probs, threshold=0.5, is_calibrated=False,
        path=output_path('xgb_probability_distribution_default.png')
    )
    save_probability_distribution(
        y_true, probs, threshold=calibrated_threshold, is_calibrated=True,
        path=output_path('xgb_probability_distribution_calibrated.png')
    )

    # ۷. پروفایل مصرف حافظه با خط بودجه ۱ گیگابایت
    timeline_ratio = np.linspace(0, 1, len(memory_profile))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(timeline_ratio, memory_profile, color='#ba5555', lw=2.0, label='RSS Memory')
    ax.axhline(1024, color='red', linestyle='--', lw=1.8, label='1 GB Edge RAM Budget')
    ax.set_ylim([100, 1080])
    ax.set_xlabel('Normalized Timeline (Ratio)', fontsize=10)
    ax.set_ylabel('RAM Usage (MB)', fontsize=10)
    ax.set_title('Memory Utilization Footprint During Evaluation', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.text(0.5, -0.05, 'Figure: Memory utilization footprint during streaming cross-day evaluation.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path('memory_usage_over_time_xgb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ۸. پروفایل مصرف CPU
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(timeline_ratio, cpu_profile, color='#4e9a68', lw=1.8, label='CPU Utilization')
    ax.set_xlabel('Normalized Timeline (Ratio)', fontsize=10)
    ax.set_ylabel('CPU Usage (%)', fontsize=10)
    ax.set_title('CPU Utilization Profile Over Evaluation Runs', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.text(0.5, -0.05, 'Figure: CPU utilization percentage profile tracked during real-time inference runs.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path('cpu_usage_over_time_xgb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n[+] Evaluation completed successfully. All artifacts saved to: {OUTPUT_DIR}')


if __name__ == '__main__':
    evaluate()
