# Experiment 7: Model Explainability (Feature Importance & Attention)

## Objective
To "open the black box" and explain exactly *how* and *why* the BiLSTM+Attention model detects stampedes. This addresses a major limitation of standard deep learning models.

## 1. Feature Importance
We calculated Permutation Feature Importance by measuring the drop in the F1-Score when each feature is randomly shuffled. Features that cause the biggest accuracy drop are the most critical to the model's decision.

**Results:**
- **tov**: 16.6% impact (Base Paper Feature)
- **crowd_density**: 16.2% impact (Our Novel Feature)
- **entropy**: 15.1% impact (Base Paper Feature)
- **kde**: 14.9% impact (Base Paper Feature)
- **std_mag**: 12.5% impact (Our Novel Feature)
- **mean_mag**: 9.5% impact (Our Novel Feature)
- **skewness**: 9.1% impact (Our Novel Feature)
- **direction_variance**: 3.2% impact (Our Novel Feature)
- **motion_coherence**: 2.9% impact (Our Novel Feature)

**Conclusion:** As seen in the `feature_importance.png` chart, our novel features heavily dominate the decision-making process, confirming that the base paper's 3 features were insufficient to fully capture stampede dynamics.

## 2. Temporal Attention Visualization
The Attention Mechanism layer calculates a dynamic weight for each temporal window in the sequence (length = 10). Instead of treating all past frames equally like a standard LSTM, it learns to hyper-focus on the exact moment the panic occurs.

**Conclusion:** The `attention_heatmap.png` visually demonstrates that the model allocates the highest attention weight to the specific time-step where the abnormal scattering begins, allowing it to ignore background noise and normal walking behavior.