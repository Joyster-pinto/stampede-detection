"""
experiment_cross_dataset.py - Step 11: Cross-Dataset Generalization
=====================================================================
Evaluates the BiLSTM+Attention model (trained purely on UMN) 
on unseen datasets (UCSD and Avenue) to demonstrate generalization.

Since ground truth labels for these datasets require complex parsing,
we will evaluate generalization qualitatively by plotting the predicted 
anomaly scores over time for sample clips. Spikes indicate detected anomalies.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.preprocess import load_ucsd_clips, load_avenue_clips
from src.optical_flow import compute_flow_sequence
from src.features import extract_all_features
from src.dataset import create_sequences
from src.models import BiLSTMAttentionClassifier

def evaluate_clip(model, clip_name, frames, fps, dataset_name):
    print(f"\nProcessing {dataset_name} clip: {clip_name} ({len(frames)} frames)")
    
    # Step 1: Optical Flow
    mags, angs = compute_flow_sequence(frames)
    
    # Step 2: Feature Extraction
    features, names = extract_all_features(mags, angs, fps, use_novel=True)
    if len(features) == 0:
        print("Clip too short.")
        return
        
    # Step 3: Create Sequences (dummy labels)
    dummy_labels = np.zeros(len(features))
    X_seq, _ = create_sequences(features, dummy_labels, seq_len=Config.SEQUENCE_LENGTH, stride=1)
    
    if len(X_seq) == 0:
        print("Not enough sequences.")
        return
        
    # Step 3.5: Normalize features
    # Since we need the scaler, we load the UMN features to fit the scaler
    feat_path = os.path.join(Config.FEATURES_DIR, 'UMN_features_full.pkl')
    import pickle
    from sklearn.preprocessing import StandardScaler
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)
    X_umn = data['X']
    scaler = StandardScaler()
    scaler.fit(X_umn.reshape(-1, X_umn.shape[2]))
    
    X_seq_norm = scaler.transform(X_seq.reshape(-1, X_seq.shape[2])).reshape(X_seq.shape)
    
    # Step 4: Inference
    model.eval()
    X_tensor = torch.tensor(X_seq_norm, dtype=torch.float32).to(Config.DEVICE)
    
    with torch.no_grad():
        outputs, _ = model(X_tensor, return_attention=True)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        
    # Pad probs to match feature length
    pad_len = Config.SEQUENCE_LENGTH - 1
    probs_padded = np.pad(probs, (pad_len, 0), mode='edge')
    
    # Plot Anomaly Scores over time
    plt.figure(figsize=(10, 4))
    plt.plot(probs_padded, color='red', linewidth=2, label='Predicted Anomaly Score')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
    plt.title(f'Cross-Dataset Generalization: {dataset_name} ({clip_name})\nModel Trained ONLY on UMN')
    plt.xlabel('Time Window')
    plt.ylabel('Anomaly Probability')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_name = f"generalization_{dataset_name}_{clip_name.replace('/', '_')}.png"
    save_path = os.path.join(Config.FIGURE_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to {save_path}")

def run_cross_dataset():
    Config.ensure_dirs()
    
    print(f"Device: {Config.DEVICE}")

    # Load pre-trained model
    model_path = os.path.join(Config.MODEL_DIR, "BiLSTM_Attention_best.pt")
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return

    # We know the input size is 9 (all features)
    model = BiLSTMAttentionClassifier(input_size=9, hidden_size=Config.HIDDEN_SIZE).to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    print("Successfully loaded BiLSTM_Attention model trained on UMN.")

    # 1. Test on UCSD
    ucsd_dir = r'D:\Research\datasets\UCSD\UCSD_Anomaly_Dataset.v1p2'
    ucsd_clips = load_ucsd_clips(ucsd_dir, subset="UCSDped2")
    # Evaluate 2 test clips
    test_clips = [c for c in ucsd_clips if c[3]] # c[3] is is_test
    for clip in test_clips[:2]:
        evaluate_clip(model, clip[0], clip[1], clip[2], "UCSD")

    # 2. Test on Avenue
    avenue_dir = r'D:\Research\datasets\Avenue\Avenue Dataset'
    avenue_clips = load_avenue_clips(avenue_dir)
    # Evaluate 2 test clips
    test_clips = [c for c in avenue_clips if c[3]]
    for clip in test_clips[:2]:
        evaluate_clip(model, clip[0], clip[1], clip[2], "Avenue")

    print("\nExperiment 3 Complete!")

if __name__ == "__main__":
    run_cross_dataset()
