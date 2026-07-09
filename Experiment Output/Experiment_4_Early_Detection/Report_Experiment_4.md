# Experiment 4: Early Detection Analysis

## Objective
To measure how quickly the model detects a stampede relative to the actual ground truth start of the anomaly (panic peak). This proves the real-world applicability of the system for early warning systems.

## Methodology
The model's predictions were compared against the exact frame where the stampede truly begins in the UMN dataset. The time difference between when the model's probability crosses 50% and the ground truth start was calculated.
- **Negative Value:** Early Detection (Warning before the peak)
- **Positive Value:** Detection Latency

## Results
- **Average Detection Timing:** -0.39 seconds (EARLY DETECTION)
- **Best Case:** Event 4 was detected 7.5 seconds early.
- **Second Best:** Event 9 was detected 1.5 seconds early.

## Conclusion
The model successfully detects the buildup of a stampede (e.g., people beginning to scatter) approximately 0.4 seconds before the absolute peak of the panic occurs on average, and up to 7.5 seconds early in some scenarios. In a real-world environment, providing sub-second to multi-second early warnings can successfully trigger automated safety measures.

## Assets Included in this Folder
- `early_detection_histogram.png` (Distribution of detection timings)
- `early_detection_event_*.png` (Individual plots for each stampede event showing the prediction curve vs ground truth)
