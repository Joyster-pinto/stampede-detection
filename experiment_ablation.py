"""
experiment_ablation.py - Step 10: Feature Ablation Study
=========================================================
Train the proposed BiLSTM+Attention model using different subsets of features
to demonstrate that the new features improve performance over the base paper.

Case A: Entropy only
Case B: Entropy + TOV
Case C: Entropy + TOV + KDE (Base paper baseline)
Case D: All old + new features (Proposed)
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import prepare_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_history

def run_ablation_study():
    Config.ensure_dirs()
    
    print(f"Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    feat_path = os.path.join(Config.FEATURES_DIR, 'UMN_features_full.pkl')
    if not os.path.exists(feat_path):
        print(f"Error: Features file not found at {feat_path}")
        print("Please run the main pipeline first to extract features.")
        return

    print("Loading cached features...")
    with open(feat_path, 'rb') as f:
        data = pickle.load(f)

    X_seq = data['X']
    y_seq = data['y']
    feature_names = data['feature_names']
    
    print(f"Loaded {X_seq.shape[0]} sequences. All Features: {feature_names}")

    cases = {
        "Case_A_Entropy": [0],
        "Case_B_Ent_TOV": [0, 1],
        "Case_C_Baseline": [0, 1, 2],
        "Case_D_Proposed": list(range(len(feature_names)))
    }

    results = {}

    for case_name, feat_indices in cases.items():
        print(f"\n{'='*60}")
        print(f"Running Ablation Case: {case_name}")
        used_feats = [feature_names[i] for i in feat_indices]
        print(f"Features ({len(used_feats)}): {used_feats}")
        print(f"{'='*60}")

        # Slice features
        X_case = X_seq[:, :, feat_indices]
        input_size = X_case.shape[2]

        # Prepare dataloaders
        train_loader, val_loader, test_loader, scaler = prepare_dataloaders(X_case, y_seq)

        # Train Proposed Model
        model_name = "BiLSTM_Attention"
        save_path = os.path.join(Config.MODEL_DIR, f"{model_name}_{case_name}_best.pt")
        
        # Reset seed for fair comparison
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
        
        model, history = train_model(model_name, train_loader, val_loader, input_size, save_path)
        
        # Evaluate
        plot_training_history(history, f"{model_name}_{case_name}")
        metrics, probs, labels, attn = evaluate_model(model, test_loader, f"{model_name}_{case_name}")
        
        results[case_name] = metrics

    # Print comparison table
    print(f"\n{'='*72}")
    print("FEATURE ABLATION RESULTS (BiLSTM+Attention)")
    print(f"{'='*72}")
    print(f"{'Case':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 72)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
              f"{metrics['roc_auc']:>10.4f}")

    # Save results to JSON
    res_path = os.path.join(Config.RESULTS_DIR, 'ablation_results.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {res_path}")

    # Plot bar chart for comparison
    labels_names = list(results.keys())
    accs = [results[n]['accuracy'] for n in labels_names]
    f1s = [results[n]['f1_score'] for n in labels_names]
    aucs = [results[n]['roc_auc'] for n in labels_names]

    x = np.arange(len(labels_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, accs, width, label='Accuracy')
    ax.bar(x, f1s, width, label='F1-Score')
    ax.bar(x + width, aucs, width, label='ROC-AUC')

    ax.set_ylabel('Scores')
    ax.set_title('Feature Ablation Comparison (BiLSTM+Attention)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_names)
    ax.legend(loc='lower right')
    
    # Zoom in on top part of the chart to show differences better
    min_score = min(min(accs), min(f1s), min(aucs))
    ax.set_ylim(max(0.0, min_score - 0.1), 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.FIGURE_DIR, 'ablation_comparison.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    run_ablation_study()
