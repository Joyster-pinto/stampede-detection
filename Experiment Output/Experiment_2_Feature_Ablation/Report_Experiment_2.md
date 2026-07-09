# Experiment 2: Feature Ablation Study

## Objective
To prove the effectiveness of the 6 novel features introduced in our pipeline (mean magnitude, standard deviation of magnitude, skewness, motion coherence, direction variance, and crowd density) against the 3 features used in the base paper.

## Tested Cases
- **Case A:** Entropy only
- **Case B:** Entropy + TOV
- **Case C (Baseline):** Entropy + TOV + KDE (Base paper's exact features)
- **Case D (Proposed):** All 9 features (3 Baseline + 6 Novel)

## Ablation Results (Using BiLSTM+Attention)
| Case | Features Included | Accuracy | Precision | Recall | F1-Score |
|------|-------------------|----------|-----------|--------|----------|
| Case A | Entropy | 68.93% | 0.00% | 0.00% | 0.00% |
| Case B | Entropy + TOV | 84.78% | 82.05% | 65.31% | 72.73% |
| Case C | Entropy + TOV + KDE | 96.47% | 96.63% | 91.84% | 94.17% |
| **Case D** | **All 9 Features** | **99.64%** | **100.00%** | **98.83%** | **99.41%** |

## Conclusion
Adding our 6 novel features resulted in a massive performance boost. The model's F1-score jumped from 94.17% (using only the base paper's features) to 99.41% (using all features). This confirms that our novel features successfully capture complex stampede dynamics (like rapid multidirectional scattering) far better than the baseline.

## Assets Included in this Folder
- `ablation_comparison.png` (Visual bar chart of the results)
- Training histories for each feature case
- `ablation_results.json` (Raw metric data)
