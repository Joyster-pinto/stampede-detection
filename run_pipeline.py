"""
run_pipeline.py - Run the full stampede detection pipeline on UMN dataset
Steps 2-9: Preprocess → Optical Flow → Features → Sequences → Train → Evaluate
"""

import sys
import os
import time
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.preprocess import extract_frames_from_video
from src.optical_flow import compute_flow_sequence
from src.features import extract_all_features
from src.dataset import create_sequences, generate_umn_labels, prepare_dataloaders
from src.train import train_all_models
from src.evaluate import (evaluate_model, plot_confusion_matrix,
                           plot_roc_curves, plot_training_history)

import torch

np.random.seed(42)
torch.manual_seed(42)

Config.ensure_dirs()

# =====================================================================
# STEPS 2-5: Feature Extraction on ALL UMN scenes
# =====================================================================
umn_dir = Config.UMN_DIR
video_files = sorted([f for f in os.listdir(umn_dir)
                      if f.startswith('scene') and f.endswith('.avi')])
print(f"Found {len(video_files)} scene videos")

all_features = []
all_labels = []

for vf in video_files:
    path = os.path.join(umn_dir, vf)
    frames, fps = extract_frames_from_video(path)

    # Skip very short clips (< 5 seconds = transition scenes)
    if len(frames) < 150:
        print(f"  Skipping {vf} (too short: {len(frames)} frames)")
        continue

    print(f"Processing {vf} ({len(frames)} frames)...")
    t = time.time()
    mags, angs = compute_flow_sequence(frames)
    print(f"  Optical flow: {time.time()-t:.1f}s")

    t = time.time()
    features, names = extract_all_features(mags, angs, fps, use_novel=True)
    print(f"  Features: {features.shape} in {time.time()-t:.1f}s")

    if len(features) == 0:
        continue

    labels = generate_umn_labels(len(features), anomaly_ratio=0.3)
    all_features.append(features)
    all_labels.append(labels)

# Combine all
X_feat = np.concatenate(all_features, axis=0)
y_feat = np.concatenate(all_labels, axis=0)
print(f"\nTotal features: {X_feat.shape}")
print(f"Total labels: {y_feat.shape} (anomaly: {y_feat.mean():.2%})")

# Create sequences
X_seq, y_seq = create_sequences(X_feat, y_feat,
                                 seq_len=Config.SEQUENCE_LENGTH,
                                 stride=Config.STRIDE)
print(f"Sequences: {X_seq.shape}")
print(f"Labels: {y_seq.shape} (anomaly: {y_seq.mean():.2%})")

# Save features
feat_path = os.path.join(Config.FEATURES_DIR, 'UMN_features_full.pkl')
with open(feat_path, 'wb') as f:
    pickle.dump({'X': X_seq, 'y': y_seq, 'feature_names': names,
                 'dataset': 'UMN'}, f)
print(f"Saved features to {feat_path}")

# =====================================================================
# STEP 8: Train ALL models
# =====================================================================
sep = "=" * 60
print(f"\n{sep}")
print("STEP 8: Training all models...")
print(sep)

train_loader, val_loader, test_loader, scaler = prepare_dataloaders(X_seq, y_seq)
input_size = X_seq.shape[2]

results = train_all_models(train_loader, val_loader, input_size)

# =====================================================================
# STEP 9: Evaluate
# =====================================================================
print(f"\n{sep}")
print("STEP 9: Model Comparison")
print(sep)

eval_results = {}
for name, (model, history) in results.items():
    plot_training_history(history, name)
    metrics, probs, labels, attn = evaluate_model(model, test_loader, name)
    eval_results[name] = (metrics, probs, labels)
    plot_confusion_matrix(labels, probs, name)

plot_roc_curves(eval_results)

# Print comparison table
print(f"\n{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
      f"{'F1':>10} {'AUC':>10}")
print("-" * 72)
for name, (metrics, _, _) in eval_results.items():
    print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
          f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
          f"{metrics['roc_auc']:>10.4f}")

# Save results
res_path = os.path.join(Config.RESULTS_DIR, 'model_comparison.json')
save_metrics = {name: metrics for name, (metrics, _, _) in eval_results.items()}
with open(res_path, 'w') as f:
    json.dump(save_metrics, f, indent=2)
print(f"\nResults saved to {res_path}")
print(f"Figures saved to {Config.FIGURE_DIR}")
print("DONE!")
