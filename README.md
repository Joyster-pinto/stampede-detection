# Hybrid Optical Flow and Temporal Attention Network for Robust Early Stampede Detection in Public Crowd Surveillance

A real-time, explainable crowd anomaly detection system that identifies early-stage panic and stampede events in surveillance video using dense optical flow features and a BiLSTM+Attention classifier.

---

## Overview

Stampede events in crowded public spaces pose severe risks to human safety. This project proposes a vision-based detection pipeline that:

- Extracts **9 optical-flow statistics** (3 adopted + 6 novel) from dense Farnebäck flow fields
- Models temporal dynamics using a **Bidirectional LSTM with Additive Attention**
- Provides **explainability** via permutation feature importance and attention-weight saliency maps
- Achieves **real-time performance** at 54.5 FPS on a standard laptop GPU

## Key Results

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
|-------|:-----------:|:------------:|:---------:|:-----------:|:-----------:|
| LSTM | 99.18 | 99.12 | 98.25 | 98.68 | 99.95 |
| GRU | 99.09 | 100.00 | 97.08 | 98.52 | 99.66 |
| BiLSTM | 99.46 | 99.71 | 98.54 | 99.12 | 99.99 |
| **BiLSTM+Attention** | **99.55** | **100.00** | **98.54** | **99.27** | **99.97** |

- **Zero false alarms** on the UMN test set (100% precision)
- **Early detection**: Anomalies detected 0.39s before annotated onset on average (up to 7.5s lead time)
- **Cross-dataset transfer**: Generalizes to UCSD Ped2 and CUHK Avenue without retraining

## Project Structure

```
├── main.py                          # Main pipeline runner
├── run_pipeline.py                  # Simplified pipeline execution
├── requirements.txt                 # Python dependencies
│
├── src/                             # Core source code
│   ├── config.py                    # Central configuration & hyperparameters
│   ├── preprocess.py                # Video loading & frame preprocessing
│   ├── optical_flow.py              # Farnebäck dense optical flow computation
│   ├── features.py                  # 9-feature extraction (3 baseline + 6 novel)
│   ├── dataset.py                   # Sequence construction & data loading
│   ├── train.py                     # Model training with early stopping
│   ├── evaluate.py                  # Evaluation metrics & visualization
│   └── models/                      # Neural network architectures
│       ├── lstm_model.py            # Unidirectional LSTM
│       ├── gru_model.py             # Unidirectional GRU
│       ├── bilstm_model.py          # Bidirectional LSTM
│       └── bilstm_attention.py      # BiLSTM + Additive Attention (proposed)
│
├── experiment_ablation.py           # Experiment 2: Feature ablation study
├── experiment_cross_dataset.py      # Experiment 3: Zero-shot domain transfer
├── experiment_early_detection.py    # Experiment 4: Detection latency analysis
├── experiment_robustness.py         # Experiment 5: Camera degradation robustness
├── experiment_cost_analysis.py      # Experiment 6: Computational cost profiling
├── experiment_explainability.py     # Experiment 7: Feature importance & attention
│
└── Experiment Output/               # Experiment results, figures, and reports
    ├── Experiment_1_Model_Comparison/
    ├── Experiment_2_Feature_Ablation/
    ├── Experiment_3_CrossDataset/
    ├── Experiment_4_Early_Detection/
    ├── Experiment_5_Robustness/
    ├── Experiment_6_Cost_Analysis/
    └── Experiment_7_Explainability/
```

## Features Extracted

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | Entropy (H) | Baseline | Disorder in movement directions |
| 2 | Temporal Occupancy Variation (TOV) | Baseline | Frame-to-frame change in active motion region |
| 3 | Kernel Density Estimation (KDE) | Baseline | Peak density of flow magnitude distribution |
| 4 | Mean Magnitude (M̄) | **Proposed** | Average crowd speed |
| 5 | Std. Magnitude (σ_M) | **Proposed** | Speed variability |
| 6 | Skewness (γ_M) | **Proposed** | Asymmetry of speed distribution |
| 7 | Motion Coherence (C) | **Proposed** | Fraction moving in a shared direction |
| 8 | Crowd Density (ρ) | **Proposed** | Fraction of pixels with active motion |
| 9 | Direction Variance (σ²_θ) | **Proposed** | Angular scatter of motion vectors |

## Model Architecture

The proposed classifier consists of:
1. **2-layer Bidirectional LSTM** (128 hidden units per direction) for temporal encoding
2. **Additive (Bahdanau) Attention** for saliency-guided sequence aggregation
3. **Classification Head** with Batch Normalization, Dropout (0.3), and Sigmoid output

Total parameters: **554,817**

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Joyster-pinto/stampede-detection.git
cd stampede-detection

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Download the following datasets and place them in the `datasets/` directory:

| Dataset | Link | Usage |
|---------|------|-------|
| UMN Crowd Activity | [Download](http://mha.cs.umn.edu/proj_events.shtml) | Training & Testing |
| UCSD Pedestrian (Ped2) | [Download](http://www.svcl.ucsd.edu/projects/anomaly/) | Cross-dataset Evaluation |
| CUHK Avenue | [Download](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/) | Cross-dataset Evaluation |

```
datasets/
├── UMN/          # UMN video sequences
├── UCSD/         # UCSD Ped2 sequences
└── Avenue/       # CUHK Avenue sequences
```

## Usage

### Run the Full Pipeline

```bash
# Run everything: preprocessing → feature extraction → training → evaluation
python main.py --step all

# Run individual steps
python main.py --step features     # Extract optical flow features
python main.py --step train        # Train all model variants
python main.py --step evaluate     # Evaluate and generate comparison plots
```

### Run Individual Experiments

```bash
python experiment_ablation.py           # Feature ablation study
python experiment_cross_dataset.py      # Zero-shot cross-dataset transfer
python experiment_early_detection.py    # Early detection latency analysis
python experiment_robustness.py         # Camera degradation robustness
python experiment_cost_analysis.py      # Computational cost profiling
python experiment_explainability.py     # Feature importance & attention maps
```

### Use Baseline Features Only

```bash
python main.py --step all --baseline-only
```

## Hardware & Software Environment

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4050 (6 GB VRAM) |
| CPU | Intel Core i7-13650HX |
| RAM | 16 GB DDR5 |
| OS | Windows 11 |
| Python | 3.12 |
| PyTorch | 2.0 |
| OpenCV | 4.8 |

## License

This project is for academic and research purposes.
