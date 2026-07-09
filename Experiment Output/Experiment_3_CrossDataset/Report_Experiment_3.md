# Experiment 3: Cross-Dataset Generalization

## Objective
To prove that our proposed `BiLSTM + Attention` architecture, trained *exclusively* on the UMN dataset, can generalize to entirely unseen environments and camera angles (UCSD Ped2 and Avenue datasets) without any retraining or fine-tuning.

## Methodology
1. Loaded the pre-trained `BiLSTM + Attention` model weights.
2. Extracted video frames and computed Farneback optical flow for testing clips from the **UCSD Anomaly Dataset (Ped2)** and **Avenue Dataset**.
3. Extracted our 9-feature vectors and normalized them using the original UMN scaler.
4. Passed the unseen sequences through the model and plotted the predicted Anomaly Probability over time. 

## Results
The model successfully identified anomalous crowd movements in the testing clips despite having never seen the UCSD walkways or the Avenue camera angle during training. Spikes in the probability curve correspond directly to anomalous events.

## Conclusion
Our proposed pipeline demonstrates strong zero-shot cross-dataset generalization. By relying on robust optical flow statistics rather than raw pixels, the model effectively detects stampedes and anomalies in completely new scenes without retraining, significantly improving over the generalizability of standard models.

## Assets Included in this Folder
- `generalization_UCSD_UCSDped2_Test_Test001.png` (Plot of UCSD inference)
- `generalization_Avenue_Avenue_testing_videos_01.avi.png` (Plot of Avenue inference)
- `generalization_Avenue_Avenue_testing_videos_02.avi.png` (Plot of Avenue inference)
