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
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, matthews_corrcoef, auc
)

# ==============================================================================
# 1. تنظیمات ظاهری مت‌پلات‌لیب (Times New Roman، اندازه فونت 10، DPI=300)
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
    """محاسبه تمام معیارهای ارزیابی روی ۱۰۰٪ داده‌ها با آستانه مشخص."""
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0
        spec = 0.0

    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "threshold": float(threshold),
        "y_pred": y_pred,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "F2-Score": float(f2),
        "MCC": float(mcc),
        "ROC-AUC": float(roc_auc_val),
        "Specificity": float(spec),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def save_confusion_matrix(y_true, y_pred, title, caption_text, output_path, cmap):
    """رسم و ذخیره ماتریس درهم‌ریختگی به همراه زیرنویس علمی."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        cbar=False,
        annot_kws={"size": 10},
        xticklabels=['Normal', 'DDoS'],
        yticklabels=['Normal', 'DDoS'],
        ax=ax
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_bar_chart(metrics, title, caption_text, output_path):
    """رسم نمودار میله‌ای معیارهای عملکرد برای یک آستانه خاص."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    metric_vals = [metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#34495e', '#f39c12']
    bars = ax.bar(metric_names, metric_vals, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

    ax.set_ylim([0.0, 1.08])
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Evaluation Metric', fontsize=10)
    ax.set_title(title, fontsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.02,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_threshold_comparison_metrics(default_metrics, calibrated_metrics, cal_thresh, caption_text, output_path):
    """رسم نمودار مقایسه میله‌ای دوتایی بین آستانه پیش‌فرض و کالیبره‌شده."""
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "MCC", "ROC-AUC"]
    vals_default = [default_metrics[name] for name in metric_names]
    vals_calibrated = [calibrated_metrics[name] for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    bars1 = ax.bar(x - width / 2, vals_default, width, label='Default Threshold (0.50)', color='#3498db', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, vals_calibrated, width, label=f'Calibrated Threshold ({cal_thresh:.4f})', color='#e67e22', edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Score', fontsize=10)
    ax.set_xlabel('Evaluation Metric', fontsize=10)
    ax.set_title('Performance Comparison: Default vs. Calibrated Threshold - XGBoost', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
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
                ha='center',
                va='bottom',
                fontsize=9
            )

    fig.text(0.5, -0.05, caption_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_probability_distribution(y_true, y_probs, threshold, is_calibrated, output_path):
    """رسم و ذخیره توزیع احتمال هر کلاس به همراه نمودار KDE و خط آستانه تصمیم‌گیری."""
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def test_cross_day_xgboost():
    print("[*] Starting Cross-Day Real-Time Evaluation for XGBoost...")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    def out(filename):
        return os.path.join(output_dir, filename)

    # 1. لود مدل، اسکیلر و متادیتا
    model_path = out("xgb_model.joblib") if os.path.exists(out("xgb_model.joblib")) else "xgb_model.joblib"
    scaler_path = out("scaler_xgb.joblib") if os.path.exists(out("scaler_xgb.joblib")) else "scaler_xgb.joblib"
    meta_path = out("xgb_metadata.json") if os.path.exists(out("xgb_metadata.json")) else "xgb_metadata.json"

    print(f"[*] Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"[*] Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"[*] Loading metadata from: {meta_path}")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    calibrated_threshold = metadata.get("best_threshold", 0.5)
    features = metadata["features"]
    default_threshold = 0.5

    # 2. پیدا کردن مسیر فایل داده‌های روز چهارشنبه
    candidates = [
        "Wednesday-workingHours.pcap_ISCX.csv",
        os.path.join("..", "Wednesday-workingHours.pcap_ISCX.csv"),
        os.path.join("data", "Wednesday-workingHours.pcap_ISCX.csv")
    ]
    data_path = None
    for cand in candidates:
        if os.path.exists(cand):
            data_path = cand
            break

    if not data_path:
        raise FileNotFoundError("Wednesday-workingHours.pcap_ISCX.csv dataset not found in current or adjacent directories.")

    print(f"[*] Processing 100% of Wednesday dataset using chunks: {data_path}")

    CHUNK_SIZE = 50000
    all_y_true = []
    all_y_probs = []
    
    memory_profile = []
    cpu_profile = []
    latency_benchmark_samples = []
    TARGET_LATENCY_SAMPLES = 5000

    # 3. پردازش استریمینگ داده‌ها روی ۱۰۰٪ نمونه‌ها
    for idx, chunk in enumerate(pd.read_csv(data_path, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk.columns = chunk.columns.str.strip()
        
        # پیش‌پردازش تمیزکاری مقادیر نامعتبر
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk.dropna(inplace=True)

        if chunk.empty:
            continue

        y_chunk = chunk['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1).values
        X_chunk = chunk[features]

        # تبدیل ویژگی‌ها و پیش‌بینی احتمالات
        chunk_X_scaled = scaler.transform(X_chunk)
        chunk_probs = model.predict_proba(chunk_X_scaled)[:, 1]

        all_y_true.extend(y_chunk)
        all_y_probs.extend(chunk_probs)

        # جمع‌آوری نمونه برای بنچمارک تاخیر
        if len(latency_benchmark_samples) < TARGET_LATENCY_SAMPLES:
            needed = TARGET_LATENCY_SAMPLES - len(latency_benchmark_samples)
            latency_benchmark_samples.extend(chunk_X_scaled[:needed])

        mem_after, cpu_after = get_resource_usage()
        memory_profile.append(mem_after)
        cpu_profile.append(cpu_after)

        print(f"    Processed Chunk {idx + 1} | Cumulative Samples: {len(all_y_true)}")

    # 4. ارزیابی تاخیر تک‌نمونه‌ای (Batch Size = 1)
    print(f"\n[*] Running Pure Single-Sample Latency Benchmark on {len(latency_benchmark_samples)} samples (Batch Size = 1)...")

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

    sample_latencies_ms = np.array(sample_latencies_ms)
    y_true = np.array(all_y_true)
    y_probs = np.array(all_y_probs)

    # 5. محاسبه معیارهای ارزیابی
    unique_classes = np.unique(y_true)
    has_two_classes = len(unique_classes) > 1

    if has_two_classes:
        roc_auc = roc_auc_score(y_true, y_probs)
        ap_score = average_precision_score(y_true, y_probs)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall_vals, precision_vals)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
    else:
        roc_auc, ap_score, pr_auc = 0.0, 0.0, 0.0
        precision_vals, recall_vals = np.array([0.0, 1.0]), np.array([0.0, 0.0])
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    default_metrics = compute_metrics(y_true, y_probs, default_threshold, roc_auc)
    calibrated_metrics = compute_metrics(y_true, y_probs, calibrated_threshold, roc_auc)

    latency_stats = {
        "Mean-Latency-ms": float(np.mean(sample_latencies_ms)),
        "P50-Latency-ms": float(np.percentile(sample_latencies_ms, 50)),
        "P90-Latency-ms": float(np.percentile(sample_latencies_ms, 90)),
        "P95-Latency-ms": float(np.percentile(sample_latencies_ms, 95)),
        "P99-Latency-ms": float(np.percentile(sample_latencies_ms, 99)),
    }

    shared_results = {
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "AP-Score": float(ap_score),
        **latency_stats,
        "Peak-Memory-MB": float(np.max(memory_profile)),
        "Avg-CPU-Usage-Percent": float(np.mean(cpu_profile))
    }

    results = {
        "default_threshold_0.5": {
            **{k: v for k, v in default_metrics.items() if k != "y_pred"},
            **shared_results
        },
        "calibrated_threshold": {
            **{k: v for k, v in calibrated_metrics.items() if k != "y_pred"},
            **shared_results
        }
    }

    print("\n" + "=" * 55)
    print("--- XGBOOST REAL-TIME EVALUATION RESULTS ---")
    print("=" * 55)
    for k, v in shared_results.items():
        if isinstance(v, float):
            print(f"| {k:<30} | {v:.6f} |")
        else:
            print(f"| {k:<30} | {v} |")
    print("=" * 55)

    with open(out("xgboost_wednesday_crossday_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # ==============================================================================
    # 6. رسم و خروجی نمودارهای علمی
    # ==============================================================================
    
    # ماتریس‌های درهم‌ریختگی
    save_confusion_matrix(
        y_true, default_metrics["y_pred"],
        'Confusion Matrix - XGBoost (Threshold = 0.50)',
        'Figure: Confusion matrix of XGBoost under default threshold (0.50).',
        out("confusion_matrix_xgboost_default.png"), cmap="Blues"
    )

    save_confusion_matrix(
        y_true, calibrated_metrics["y_pred"],
        f'Confusion Matrix - XGBoost (Calibrated Threshold = {calibrated_threshold:.4f})',
        'Figure: Confusion matrix of XGBoost under calibrated threshold.',
        out("confusion_matrix_xgboost_calibrated.png"), cmap="Oranges"
    )

    # نمودارهای میله‌ای عملکرد
    save_metrics_bar_chart(
        default_metrics,
        'Performance Metrics Summary - XGBoost (Threshold = 0.50)',
        'Figure: Overall classification metrics for XGBoost (Threshold = 0.50).',
        out("metrics_bar_chart_xgboost_default.png")
    )

    save_metrics_bar_chart(
        calibrated_metrics,
        f'Performance Metrics Summary - XGBoost (Calibrated Threshold = {calibrated_threshold:.4f})',
        f'Figure: Overall classification metrics for XGBoost under calibrated threshold ({calibrated_threshold:.4f}).',
        out("metrics_bar_chart_xgboost_calibrated.png")
    )

    save_threshold_comparison_metrics(
        default_metrics, calibrated_metrics, calibrated_threshold,
        'Figure: Side-by-side metric comparison between default and calibrated thresholds for XGBoost.',
        out("metrics_bar_chart_xgboost.png")
    )

    # نمودارهای دوقلوی ROC و PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Baseline')
    if has_two_classes:
        axes[0].plot(fpr, tpr, color='#28536b', lw=2.0, label=f'XGBoost (AUC = {roc_auc:.4f})')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=10)
    axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=10)
    axes[0].set_title('(a) Receiver Operating Characteristic (ROC)', fontsize=10)
    axes[0].legend(loc="lower right", frameon=True, fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    if has_two_classes:
        axes[1].plot(recall_vals, precision_vals, color='#19647e', lw=2.0, label=f'XGBoost (AP = {ap_score:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=10)
    axes[1].set_ylabel('Precision', fontsize=10)
    axes[1].set_title('(b) Precision-Recall (PR) Curve', fontsize=10)
    axes[1].legend(loc="lower left", frameon=True, fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    fig.text(0.5, -0.05, 'Figure: Discrimination performance (ROC and PR curves) on Wednesday cross-day dataset.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(out("fig_roc_pr_curves_xgboost.png"), dpi=300, bbox_inches='tight')
    plt.savefig(out("xgb_wednesday_roc_pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # پروفایل زمانی مصرف CPU
    timeline_ratio = np.linspace(0, 1, len(cpu_profile))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(timeline_ratio, cpu_profile, color='#4e9a68', lw=1.8, label='CPU Utilization')
    ax.set_xlabel('Normalized Timeline (Ratio)', fontsize=10)
    ax.set_ylabel('CPU Usage (%)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.text(0.5, -0.05, 'Figure: CPU utilization percentage profile over the evaluation duration.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(out("xgb_cpu_usage_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # پروفایل ردپای RAM با سقف بودجه 1GB
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(timeline_ratio, memory_profile, color='#ba5555', lw=2.0, label='RSS Memory')
    ax.axhline(y=1024, color='red', linestyle='--', lw=1.8, label='1 GB Edge RAM Budget')
    ax.set_ylim([100, 1080])
    ax.set_xlabel('Normalized Timeline (Ratio)', fontsize=10)
    ax.set_ylabel('RAM Usage (MB)', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=10)
    fig.text(0.5, -0.05, 'Figure: Memory utilization footprint during streaming cross-day evaluation.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(out("xgb_memory_usage_over_time.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # توزیع احتمال برای آستانه پیش‌فرض 0.50
    save_probability_distribution(
        y_true, y_probs,
        threshold=default_threshold,
        is_calibrated=False,
        output_path=out("xgb_probability_distribution_default.png")
    )

    # توزیع احتمال برای آستانه کالیبره‌شده
    save_probability_distribution(
        y_true, y_probs,
        threshold=calibrated_threshold,
        is_calibrated=True,
        output_path=out("xgb_probability_distribution_calibrated.png")
    )

    # نمودار صدک‌های تاخیر
    percentiles = [50, 90, 95, 99]
    perc_vals = [latency_stats[f"P{p}-Latency-ms"] for p in percentiles]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([f"P{p}" for p in percentiles], perc_vals, color='#1abc9c', edgecolor='black', linewidth=0.8, width=0.45)
    ax.set_ylabel('Inference Latency (ms)', fontsize=10)
    ax.set_xlabel('Latency Percentile', fontsize=10)
    ax.set_title('Single-Sample Latency Percentiles - XGBoost (Batch Size = 1, N = 5000)', fontsize=10)

    for bar, val in zip(bars, perc_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            val + 0.01 * max(perc_vals),
            f'{val:.4f} ms',
            ha='center',
            va='bottom',
            fontsize=10
        )
    ax.grid(True, linestyle=':', alpha=0.5, axis='y')
    fig.text(0.5, -0.05, 'Figure: Latency percentiles (P50, P90, P95, P99) on 5,000 single-sample inferences.', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(out("latency_percentiles_xgboost.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[+] All XGBoost figures and JSON results successfully exported to: {output_dir}")


if __name__ == "__main__":
    test_cross_day_xgboost()
