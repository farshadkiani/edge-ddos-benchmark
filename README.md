# Edge DDoS Benchmark: Machine Learning & Deep Learning Evaluation on Edge Devices

This repository contains the complete benchmarking pipeline, ablation studies, and evaluation framework for lightweight DDoS attack detection on resource-constrained edge devices (targeted for edge environments like Raspberry Pi 4). 

The experimental evaluations span **four baseline models** (Random Forest, XGBoost, LSTM, and Logistic Regression) evaluated across **four major experimental setups**:
1. **Feature-Group Ablation Study** (Impact of dropping `Flow_Duration`, `Flow_Rate`, `Packet_Count`, and `Packet_Size`)
2. **Cross-Day Generalization** (Train on Friday DDoS $\to$ Test on Wednesday unknown traffic with Threshold Calibration)
3. **In-Domain 2017 Evaluation** (CICIDS2017 Friday temporal split)
4. **In-Domain 2019 Evaluation** (CICDDoS2019 DNS flood streaming simulation)

---

## 📋 Table of Contents
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [How to Run (Docker Workflow)](#how-to-run-docker-workflow)
- [Directory Structure](#directory-structure)
- [Detailed Model Inputs & Outputs by Scenario](#detailed-model-inputs--outputs-by-scenario)
  - [1. Feature-Group Ablation Study](#1-feature-group-ablation-study)
  - [2. Cross-Day Evaluation (Friday → Wednesday)](#2-cross-day-evaluation-friday--wednesday)
  - [3. In-Domain Evaluation (CICIDS2017)](#3-in-domain-evaluation-cicids2017)
  - [4. In-Domain Simulation (CICDDoS2019)](#4-in-domain-simulation-cicddos2019)

---

## ⚙️ Prerequisites & Environment Setup

- **Docker Desktop** installed and running.
- **Hardware Profile (Edge Simulator)**: Containers run with restricted resources (`--cpus="1.0" --memory="1g"`) to simulate Raspberry Pi 4 edge behavior.
- **Datasets**:
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` ([CICIDS2017 Dataset Official Download Page](https://www.unb.ca/cic/datasets/ids-2017.html))
  - `Wednesday-workingHours.pcap_ISCX.csv` ([CICIDS2017 Dataset Official Download Page](https://www.unb.ca/cic/datasets/ids-2017.html))
  - `DrDoS_DNS.csv` ([CICDDoS2019 Dataset Official Download Page](https://www.unb.ca/cic/datasets/ddos-2019.html))

---

## 🚀 How to Run (Docker Workflow)

The workspace structure requires the training/testing scripts, datasets, and configuration files to reside in the working directory:
- **`Dockerfile`**: Defines the training environment, installs dependencies from `requirements.txt`, and executes the designated training script via `CMD`.
- **`Dockerfile.sim`**: Defines the resource-constrained edge simulation environment for single-sample streaming inference (`Batch Size = 1`) and latency benchmarking.
- **`requirements.txt`**: Specifies exact dependencies (e.g., `scikit-learn`, `xgboost`, `tensorflow`, `pandas`, `psutil`).

### Step-by-Step Instructions:

1. **Launch Docker Desktop**.
2. **Open Terminal / Command Prompt (CMD)** and navigate to the project directory:
   ```cmd
   cd path\to\edge-ddos-benchmark
   ```
3. **Model Training (Container Execution)**:
   - Configure the target script in `Dockerfile` (e.g., `CMD ["python", "train_rf_without_flow_rate.py"]`).
   - Build and run the training container:
     ```cmd
     docker build -t ddos-trainer .
     docker run --rm -v "%cd%/results:/app/results" ddos-trainer
     ```
4. **Model Testing & Latency Benchmarking (Edge Simulation)**:
   - Configure the target simulator script in `Dockerfile.sim` (e.g., `CMD ["python", "rf_crossday_ram_optimized_no_flow_rate.py"]`).
   - Build and run the evaluation container under constrained CPU & RAM:
     ```cmd
     docker build -f Dockerfile.sim -t ddos-simulator .
     docker run --rm -v "%cd%/results:/app/results" --cpus="1.0" --memory="1g" ddos-simulator
     ```
   *(Note: For In-Domain 2019, only the simulation step (step 4) is executed).*
5. All produced metrics, metadata, JSON reports, and 300 DPI publication-ready figures are automatically mapped and stored in the **`results/`** directory.

---

## 📁 Directory Structure

```text
├── Dockerfile                      # Training container recipe
├── Dockerfile.sim                  # Edge latency & memory benchmark container recipe
├── requirements.txt                # Python package dependencies
├── src/                            # Pipeline source code (train, evaluate, calibrate)
│   ├── ablation/
│   ├── cross_day/
│   └── indomain/
├── results/                        # Generated output artifacts (ignored in git except metrics)
│   ├── figures/                    # 300 DPI Times New Roman academic figures
│   └── *.json                      # Structured metrics, hardware profiles, and latency stats
```

---

## 📊 Detailed Model Inputs & Outputs by Scenario

### 1. Feature-Group Ablation Study

Evaluates performance degradation when key feature groups are omitted. Models are trained on Friday and evaluated across Wednesday using temporal threshold calibration ($F_2$-Score).

| Model / Scenario | Training Inputs | Training Outputs (Saved Models / Metadata) | Testing Inputs | Evaluation Outputs (`results/`) |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** <br>`without_flow_duration` | Friday CSV | `xgb_model.joblib`<br>`scaler_xgb.joblib`<br>`xgb_metadata.json` | Wednesday CSV<br>Trained weights & scaler | `wednesday_eval_default.json`<br>`wednesday_eval_calibrated.json`<br>`wednesday_eval_comparison.json`<br>Publication figures |
| **XGBoost** <br>`without_flow_rate` | Friday CSV | `xgb_model.joblib`<br>`scaler_xgb.joblib`<br>`xgb_metadata.json` | Wednesday CSV<br>Trained weights & scaler | Evaluation JSON (Default vs. Calibrated metrics, Latency P50–P99, Peak RAM, CPU %)<br>Figures |
| **XGBoost** <br>`without_packet_count` | Friday CSV | `xgb_model.joblib`<br>`scaler_xgb.joblib`<br>`xgb_metadata.json` | Wednesday CSV<br>Trained weights & scaler | `xgb_wednesday_crossday_results.json`<br>Classification metrics, Latency percentiles, RAM/CPU stats |
| **XGBoost** <br>`without_packet_size` | Friday CSV | `xgb_model.joblib`<br>`scaler_xgb.joblib`<br>`xgb_metadata.json` | Wednesday CSV<br>Trained weights & scaler | `xgb_wednesday_crossday_results.json`<br>Metrics, Latency percentiles, RAM/CPU stats |
| **Random Forest** <br>`without_flow_duration` | Friday CSV | `random_forest_friday_model_without_flow_duration.pkl`<br>`train_metadata_rf_without_flow_duration.json`<br>`rf_friday_validation_metrics_without_flow_duration.json`<br>`feature_importance_rf_without_flow_duration.json` | Friday CSV (20% Val for calibration)<br>Wednesday CSV<br>Trained PKL model | `rf_cross_day_evaluation_results_no_flow_duration.json`<br>Optimal $F_2$ threshold, Dual-threshold metrics, Latency percentiles & Throughput, Figures |
| **Random Forest** <br>`without_packet_size` | Friday CSV | `random_forest_friday_model_without_packet_size.pkl`<br>`train_metadata_rf_without_packet_size.json`<br>`rf_friday_validation_metrics_without_packet_size.json`<br>`feature_importance_rf_without_packet_size.json` | Trained PKL model<br>Friday CSV (20% Val)<br>Wednesday CSV | `rf_cross_day_evaluation_results.json`<br>Complete metrics, calibration specs, real-time latency ($P_{50}-P_{99}$), Figures |
| **Random Forest** <br>`without_packet_count` | Friday CSV | `random_forest_friday_model_without_packet_count.pkl`<br>`train_metadata_rf_without_packet_count.json`<br>`rf_friday_validation_metrics_without_packet_count.json`<br>`feature_importance_rf_without_packet_count.json` | Trained PKL model<br>Friday CSV (20% Val)<br>Wednesday CSV | `rf_wednesday_crossday_results_no_packet_count.json`<br>Traffic stats, calibration results, dual-threshold metrics, hardware benchmark, Figures |
| **Random Forest** <br>`without_flow_rate` | Friday CSV | `random_forest_friday_model_without_flow_rate.pkl`<br>`train_metadata_rf_without_flow_rate.json`<br>`rf_friday_validation_metrics_without_flow_rate.json`<br>`feature_importance_rf_without_flow_rate.json` | Trained PKL model<br>Friday CSV (20% Val)<br>Wednesday CSV | `rf_wednesday_crossday_results_no_flow_rate.json`<br>Ablation metadata, calibrated threshold, Single-sample latency (P50–P99), ROC-AUC/PR-AUC, Figures |
| **LSTM** <br>`without_flow_duration` | Friday CSV | `ddos_scaler_without_flow_duration.pkl`<br>`best_lstm_attack_detector_without_flow_duration.keras`<br>`best_threshold_without_flow_duration.txt`<br>`lstm_attack_detector_without_flow_duration.tflite` | Wednesday CSV<br>Trained Keras & Scaler | `metrics_cross_dataset_without_flow_duration.json`<br>Resource profiles (CPU, RAM, model size), Latency (Mean, P50–P99), Confusion matrix & metrics, Figures |
| **LSTM** <br>`without_flow_rate` | Friday CSV | `ddos_scaler_without_flow_rate.pkl`<br>`best_lstm_attack_detector_without_flow_rate.keras`<br>`best_threshold_without_flow_rate.txt`<br>`lstm_attack_detector_without_flow_rate.tflite` | Friday CSV<br>Wednesday CSV<br>Trained Keras & Scaler | `metrics_cross_dataset_without_flow_rate.json`<br>Hardware metrics, latency analysis, dual-threshold evaluation, Figures |
| **LSTM** <br>`without_packet_count` | Friday CSV | `ddos_scaler.pkl`<br>`best_lstm_attack_detector.keras`<br>`best_threshold.txt`<br>`lstm_attack_detector_final.tflite` | Wednesday CSV<br>Trained Keras & Scaler | `metrics_2017.json`<br>Resource metrics (`peak_rss_mb`, CPU), latency analysis, dual-threshold metrics, Figures |
| **LSTM** <br>`without_packet_size` | Friday CSV | `ddos_scaler.pkl`<br>`best_lstm_attack_detector.keras`<br>`best_threshold.txt`<br>`lstm_attack_detector_final.tflite` | Wednesday CSV<br>Trained Keras & Scaler | `metrics_2017.json`<br>Resource profiles, latency analysis, dual-threshold metrics, Figures |
| **Logistic Regression** <br>`without_flow_duration` | Friday CSV | `lr_model_without_flow_duration.joblib`<br>`scaler_lr_without_flow_duration.joblib`<br>`lr_metadata_without_flow_duration.json`<br>`lr_training_summary_without_flow_duration.txt` | Trained weights, scaler, metadata<br>Wednesday CSV | `lr_without_flow_duration_crossday_results.json`<br>Metrics at both thresholds, latency stats (Mean, P50–P99, FPS), RAM & CPU profiles, Figures |
| **Logistic Regression** <br>`without_flow_rate` | Friday CSV | `lr_model_without_flow_rate.joblib`<br>`scaler_lr_without_flow_rate.joblib`<br>`lr_metadata_without_flow_rate.json`<br>`lr_training_summary_without_flow_rate.txt` | Trained weights, scaler, metadata<br>Wednesday CSV | `lr_without_flow_rate_crossday_results.json`<br>Metrics at both thresholds, latency stats (P50–P99, FPS), RAM & CPU profiles, Figures |
| **Logistic Regression** <br>`without_packet_count` | Friday CSV | `lr_model_without_packet_count.joblib`<br>`scaler_lr_without_packet_count.joblib`<br>`lr_metadata_without_packet_count.json`<br>`lr_training_summary_without_packet_count.txt` | Trained weights, scaler, metadata<br>Wednesday CSV | `lr_without_packet_count_wednesday_crossday_results.json`<br>Dual-threshold metrics, Single-sample latency (P50–P99), RAM & CPU profiles, Figures |
| **Logistic Regression** <br>`without_packet_size` | Friday CSV | `lr_model_without_packet_size.joblib`<br>`scaler_lr_without_packet_size.joblib`<br>`lr_metadata_without_packet_size.json`<br>`lr_training_summary_without_packet_size.txt` | Wednesday CSV<br>Trained weights, scaler, metadata | `lr_without_packet_size_wednesday_crossday_results.json`<br>Metrics at both thresholds, latency stats, RAM & CPU profiles, Figures |

---

### 2. Cross-Day Evaluation (Friday → Wednesday)

Complete feature set cross-day evaluation under realistic distribution shifts.

* **Logistic Regression (LR)**:
  - **Input Model**: Friday CSV
  - **Output Model**: `lr_model.joblib`, `scaler_lr.joblib`, `lr_metadata.json`
  - **Input Test**: Wednesday CSV, Friday CSV (optional for fallback calibration), trained model artifacts
  - **Output Test**: `lr_wednesday_crossday_results.json` (Calibrated threshold, $F_2$ score, dual-threshold classification metrics, 50,000-sample latency stats, peak/avg RAM & CPU), Evaluation figures.
* **LSTM**:
  - **Input Model**: Friday CSV
  - **Output Model**: `best_lstm_attack_detector.keras`, `lstm_attack_detector_final.tflite`, `ddos_scaler.pkl`, `best_threshold.txt`
  - **Input Test**: Wednesday CSV, trained Keras model, scaler, threshold file
  - **Output Test**: `metrics_2017.json` (Resource metrics, peak RSS MB, latency percentiles P50–P99, dual-threshold metrics), Evaluation figures.
* **Random Forest (RF)**:
  - **Input Model**: Friday CSV
  - **Output Model**: `random_forest_friday_model.pkl`
  - **Input Test**: Friday CSV (last 20% temporal split for $F_2$ calibration), Wednesday CSV, trained model PKL
  - **Output Test**: `rf_crossday_evaluation_summary.json` (Raspberry Pi 4 hardware target profile, class distribution, calibrated threshold, latency stats, dual-threshold metrics, ROC-AUC, PR-AUC), Evaluation figures.
* **XGBoost**:
  - **Input Model**: Friday CSV
  - **Output Model**: `xgb_model.joblib`, `scaler_xgb.joblib`, `xgb_metadata.json`
  - **Input Test**: Wednesday CSV, trained model, scaler, metadata
  - **Output Test**: `xgboost_wednesday_crossday_results.json` (Complete metrics, P50–P99 latency, Peak RAM, Avg CPU), Evaluation figures.

---

### 3. In-Domain Evaluation (CICIDS2017)

Evaluates in-domain detection performance using a strict temporal split on Friday data.

* **Logistic Regression (LR)**:
  - **Input Model**: Friday CSV
  - **Output Model**: `logistic_regression_attack_detector.pkl`, `classification_report_lr.txt`, `metrics_lr_2017.json`
  - **Input Test**: Friday CSV, trained PKL detector
  - **Output Test**: `benchmark_lr_indomain2017.json` (Temporal split statistics, classification metrics, confusion matrix, latency stats, throughput, system resource usage), `figure_captions_lr.txt`, Figures.
* **LSTM**:
  - **Input Model**: Friday CSV
  - **Output Model**: `lstm_attack_detector_final.keras`, `best_lstm_attack_detector.keras`, `lstm_attack_detector_final.tflite`, `ddos_scaler.pkl`
  - **Input Test**: Friday CSV, trained TFLite model, scaler
  - **Output Test**: `benchmark_lstm_indomain2017.json` (Evaluation protocol: Batch size=1, Seq length=10, latency stats, throughput, peak RAM, CPU usage), `figure_captions_lstm.txt`, Figures.
* **Random Forest (RF)**:
  - **Input Model**: Friday CSV
  - **Output Model**: `random_forest_attack_detector.pkl`
  - **Input Test**: Friday CSV, trained PKL detector
  - **Output Test**: `benchmark_rf_indomain2017.json` (Classification metrics, real-time latency stats $P_{50}-P_{99}$, throughput, peak RSS RAM, CPU usage), `benchmark_report_rf_indomain2017.txt`, Figures.
* **XGBoost**:
  - **Input Model**: Friday CSV
  - **Output Model**: `xgboost_attack_detector.pkl`
  - **Input Test**: Friday CSV, trained PKL detector
  - **Output Test**: `benchmark_xgb_indomain2017.json` (Classification metrics, confusion matrix, latency stats, throughput, peak RAM, CPU usage), `benchmark_report_xgb_indomain2017.txt`, `figure_captions_xgb.txt`, Figures.

---

### 4. In-Domain Simulation (CICDDoS2019)

Real-time streaming simulation using `DrDoS_DNS.csv` under edge hardware constraints.

* **Logistic Regression (LR)**:
  - **Input Test**: `DrDoS_DNS.csv`
  - **Output Test**: `logistic_regression_attack_detector_sim.pkl`, `metrics_lr_sim.json` (Accuracy, Balanced Acc, Precision, Recall, F1, F2, MCC, ROC-AUC, latency stats, resource summary), `classification_report_lr_sim.txt`, `resource_monitor_lr_sim.json`, Figures.
* **LSTM**:
  - **Input Test**: `DrDoS_DNS.csv`
  - **Output Test**: `lstm_model_best.h5`, `scaler_lstm.pkl`, `metrics_lstm.json` (Comprehensive metrics, confusion matrix, nanosecond-level latency stats P50–P99, hardware summary), `classification_report_lstm.txt`, `resource_monitor_lstm.json`, Figures.
* **RF**:
  - **Input Test**: `DrDoS_DNS.csv`
  - **Output Test**: `random_forest_attack_detector_sim.pkl`, `scaler_rf_sim.pkl`, `metrics_rf_sim.json` (Classification metrics, confusion matrix, millisecond single-sample latency P50–P99, hardware monitoring), `classification_report_rf_sim.txt`, `resource_monitor_rf_sim.json`, Figures.
* **XGBoost**:
  - **Input Test**: `DrDoS_DNS.csv`
  - **Output Test**: `xgboost_attack_detector_sim.pkl`, `scaler_xgb_sim.pkl`, `metrics_xgb_sim.json` (Classification metrics, confusion matrix, real-time latency percentiles), `resource_monitor_xgb_sim.json`, Figures.
```
