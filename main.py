"""
main.py - Main Pipeline Runner
================================
Runs the complete stampede detection pipeline step by step.
Usage:
    python main.py --step all          # Run everything
    python main.py --step preprocess   # Step 2 only
    python main.py --step features     # Steps 3-5
    python main.py --step train        # Step 8
    python main.py --step evaluate     # Step 9
    python main.py --step experiments  # Steps 9-15
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import pickle
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.preprocess import load_umn_videos
from src.optical_flow import compute_flow_sequence
from src.features import extract_all_features
from src.dataset import (create_sequences, generate_umn_labels,
                          prepare_dataloaders)
from src.train import train_all_models, train_model
from src.evaluate import (evaluate_model, plot_confusion_matrix,
                           plot_roc_curves, plot_training_history)


def set_seed(seed=None):
    """Set random seeds for reproducibility."""
    if seed is None:
        seed = Config.SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def step_extract_features(dataset="UMN", use_novel=True):
    """Steps 2-5: Preprocess → Optical Flow → Features → Sequences.

    Processes all videos and saves extracted features to disk.
    """
    Config.ensure_dirs()
    features_file = os.path.join(Config.FEATURES_DIR,
                                  f"{dataset}_features_{'full' if use_novel else 'baseline'}.pkl")

    if os.path.exists(features_file):
        print(f"Loading cached features from {features_file}")
        with open(features_file, 'rb') as f:
            return pickle.load(f)

    print(f"\n{'='*60}")
    print(f"STEP 2-5: Processing {dataset} dataset")
    print(f"{'='*60}")

    # Step 2: Load and preprocess videos
    print("\n[Step 2] Loading and preprocessing videos...")
    if dataset == "UMN":
        videos = load_umn_videos()
    else:
        raise ValueError(f"Dataset {dataset} not yet implemented. Start with UMN.")

    all_features = []
    all_labels = []

    for video_name, frames, fps in tqdm(videos, desc="Processing videos"):
        if len(frames) < 4:
            print(f"  Skipping {video_name} (too few frames: {len(frames)})")
            continue

        # Step 3: Compute optical flow
        print(f"\n[Step 3] Computing optical flow for {video_name}...")
        magnitudes, angles = compute_flow_sequence(frames)
        print(f"  Computed {len(magnitudes)} flow frames")

        # Step 4: Extract features
        print(f"[Step 4] Extracting features...")
        feature_matrix, feature_names = extract_all_features(
            magnitudes, angles, fps, use_novel=use_novel
        )
        if len(feature_matrix) == 0:
            print(f"  Skipping {video_name} (no features extracted)")
            continue
        print(f"  Feature matrix shape: {feature_matrix.shape}")
        print(f"  Features: {feature_names}")

        # Generate labels for UMN (last ~30% is anomaly)
        labels = generate_umn_labels(len(feature_matrix), anomaly_ratio=0.3)

        all_features.append(feature_matrix)
        all_labels.append(labels)

    # Concatenate all videos
    X_features = np.concatenate(all_features, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)

    # Step 5: Create temporal sequences
    print(f"\n[Step 5] Creating temporal sequences...")
    X_seq, y_seq = create_sequences(X_features, y_labels,
                                     seq_len=Config.SEQUENCE_LENGTH,
                                     stride=Config.STRIDE)
    print(f"  Sequences: {X_seq.shape}")
    print(f"  Labels: {y_seq.shape} (anomaly ratio: {y_seq.mean():.2%})")

    data = {
        "X": X_seq,
        "y": y_seq,
        "feature_names": feature_names,
        "dataset": dataset,
    }

    # Cache features
    os.makedirs(os.path.dirname(features_file), exist_ok=True)
    with open(features_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved features to {features_file}")

    return data


def step_train(data, model_names=None):
    """Step 8: Train models."""
    if model_names is None:
        model_names = ["LSTM", "GRU", "BiLSTM", "BiLSTM_Attention"]

    X, y = data["X"], data["y"]
    input_size = X.shape[2]

    print(f"\n{'='*60}")
    print(f"STEP 8: Training models")
    print(f"Input shape: {X.shape} | Features: {input_size}")
    print(f"{'='*60}")

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(X, y)

    # Train all models
    results = train_all_models(train_loader, val_loader, input_size)

    # Save training histories
    for name, (model, history) in results.items():
        plot_training_history(history, name)

    return results, test_loader, scaler, input_size


def step_evaluate(results, test_loader):
    """Step 9: Evaluate all models."""
    print(f"\n{'='*60}")
    print(f"STEP 9: Model Comparison (Experiment 1)")
    print(f"{'='*60}")

    eval_results = {}
    for name, (model, history) in results.items():
        metrics, probs, labels, attn = evaluate_model(model, test_loader, name)
        eval_results[name] = (metrics, probs, labels)

        # Plot confusion matrix for each model
        plot_confusion_matrix(labels, probs, name)

    # Plot ROC curves for all models
    plot_roc_curves(eval_results)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'AUC':>10}")
    print("-" * 72)
    for name, (metrics, _, _) in eval_results.items():
        print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} "
              f"{metrics['roc_auc']:>10.4f}")

    # Save results to JSON
    Config.ensure_dirs()
    results_path = os.path.join(Config.RESULTS_DIR, "model_comparison.json")
    save_metrics = {name: metrics for name, (metrics, _, _) in eval_results.items()}
    with open(results_path, 'w') as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Stampede Detection Pipeline")
    parser.add_argument("--step", default="all",
                        choices=["all", "features", "train", "evaluate"],
                        help="Pipeline step to run")
    parser.add_argument("--dataset", default="UMN",
                        choices=["UMN", "UCSD", "Avenue"],
                        help="Dataset to use")
    parser.add_argument("--novel", action="store_true", default=True,
                        help="Use novel features (default: True)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Use only baseline features (Entropy, TOV, KDE)")
    args = parser.parse_args()

    set_seed()
    use_novel = not args.baseline_only

    print(f"Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.step in ["all", "features"]:
        data = step_extract_features(args.dataset, use_novel)

    if args.step in ["all", "train"]:
        if args.step == "train":
            data = step_extract_features(args.dataset, use_novel)
        results, test_loader, scaler, input_size = step_train(data)

    if args.step in ["all", "evaluate"]:
        if args.step == "evaluate":
            data = step_extract_features(args.dataset, use_novel)
            results, test_loader, scaler, input_size = step_train(data)
        eval_results = step_evaluate(results, test_loader)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"Figures saved to: {Config.FIGURE_DIR}")
    print(f"Results saved to: {Config.RESULTS_DIR}")
    print(f"Models saved to: {Config.MODEL_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
