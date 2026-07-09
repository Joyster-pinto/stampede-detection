"""
experiment_early_detection.py - Step 12: Early Detection Analysis
===================================================================
Analyzes how quickly the model detects a stampede relative to the 
actual ground truth start of the anomaly. 

A negative delay means early detection (warning before the peak).
A positive delay means detection latency.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.models import BiLSTMAttentionClassifier
from sklearn.preprocessing import StandardScaler

def run_early_detection():
    Config.ensure_dirs()
    
    print("Loading UMN features and ground truth...")
    feat_path = os.path.join(Config.FEATURES_DIR, 'UMN_features_full.pkl')
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
        
    X_features = data['X']
    y_labels = data['y']
    
    # We will simulate processing a video sequentially
    # Since UMN is a concatenated set of sequences, we'll look at 
    # the transition points where label changes from 0 to 1.
    
    # 1. Normalize Features
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_features.reshape(-1, X_features.shape[2])).reshape(X_features.shape)
    
    # 2. Load Model
    model = BiLSTMAttentionClassifier(input_size=9, hidden_size=Config.HIDDEN_SIZE).to(Config.DEVICE)
    model.load_state_dict(torch.load(os.path.join(Config.MODEL_DIR, "BiLSTM_Attention_best.pt"), map_location=Config.DEVICE))
    model.eval()
    
    # 3. Predict on all sequences
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(Config.DEVICE)
    with torch.no_grad():
        outputs, _ = model(X_tensor, return_attention=True)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # 4. Find Anomaly Start Points
    # A new stampede starts when y changes from 0 to 1
    # We pad with 0 to detect the very first change if it happens at index 0
    y_diff = np.diff(np.pad(y_labels, (1, 0), constant_values=0))
    stampede_starts = np.where(y_diff == 1)[0]
    
    # Assume 1 window = 0.5 seconds (depends on stride/fps, but generally Config.SEQUENCE_LENGTH / FPS)
    # At 30 FPS, if window=15 frames, that's 0.5s per window
    window_duration_sec = 0.5 
    
    detection_delays = []
    
    print(f"Found {len(stampede_starts)} stampede events in UMN.")
    
    for i, start_idx in enumerate(stampede_starts):
        # Look at a window around the stampede start (e.g. 20 windows before, 20 after)
        lookback = 40
        lookforward = 40
        
        start_search = max(0, start_idx - lookback)
        end_search = min(len(probs), start_idx + lookforward)
        
        # Find where model probability first crosses 0.5 in this search window
        search_probs = probs[start_search:end_search]
        detections = np.where(search_probs > 0.5)[0]
        
        if len(detections) > 0:
            # First detection index relative to the whole sequence
            first_detect_idx = start_search + detections[0]
            
            # Delay in windows
            delay_windows = first_detect_idx - start_idx
            
            # Delay in seconds
            delay_seconds = delay_windows * window_duration_sec
            detection_delays.append(delay_seconds)
            print(f"Event {i+1}: True Start={start_idx}, Detected={first_detect_idx} -> Delay: {delay_seconds:.2f} seconds")
            
            # Plot individual event
            plt.figure(figsize=(8, 4))
            plt.plot(np.arange(start_search, end_search), search_probs, 'b-', label='Model Probability')
            plt.axvline(x=start_idx, color='r', linestyle='-', label='True Stampede Start')
            plt.axvline(x=first_detect_idx, color='g', linestyle='--', label=f'Model Detection ({delay_seconds:.2f}s)')
            plt.axhline(y=0.5, color='gray', linestyle=':')
            plt.title(f'Event {i+1} Early Detection Analysis')
            plt.xlabel('Time Window')
            plt.ylabel('Probability')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(Config.FIGURE_DIR, f'early_detection_event_{i+1}.png'))
            plt.close()
        else:
            print(f"Event {i+1}: Missed detection in search window.")
            
    if detection_delays:
        avg_delay = np.mean(detection_delays)
        status = "EARLY DETECTION" if avg_delay < 0 else "LATENCY"
        print(f"\nAverage Detection Timing: {avg_delay:.2f} seconds ({status})")
        
        # Plot Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(detection_delays, bins=10, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(x=avg_delay, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {avg_delay:.2f}s')
        plt.title('Distribution of Detection Delays')
        plt.xlabel('Detection Delay (Seconds)\n<0 = Early Detection | >0 = Latency')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.FIGURE_DIR, 'early_detection_histogram.png'))
        plt.close()

if __name__ == "__main__":
    run_early_detection()
