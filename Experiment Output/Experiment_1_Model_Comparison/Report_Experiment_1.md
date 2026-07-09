# Experiment 1: Model Comparison (Baseline vs Proposed)

## Objective
To determine the most effective deep learning architecture for classifying normal vs. stampede behaviors based on dense optical flow features, comparing the base paper's LSTM against our proposed BiLSTM + Attention model.

## Results on Test Set
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| LSTM (Baseline) | 99.18% | 99.12% | 98.25% | 98.68% | 99.95% |
| GRU | 99.09% | 100.00% | 97.08% | 98.52% | 99.66% |
| BiLSTM | 99.46% | 99.71% | 98.54% | 99.12% | 99.99% |
| **BiLSTM + Attention (Proposed)** | **99.55%** | **100.00%** | **98.54%** | **99.27%** | **99.97%** |

## Conclusion
The proposed `BiLSTM + Attention` model outperforms the base paper's unidirectional LSTM model across all major metrics, achieving a 99.55% accuracy and perfect 100% precision. The bidirectional context and attention mechanism successfully address the limitations of the baseline approach.

## Assets Included in this Folder
- `roc_curves_comparison.png`
- Confusion matrices for all models
- Training history graphs for all models
- `model_comparison.json` (Raw metric data)
