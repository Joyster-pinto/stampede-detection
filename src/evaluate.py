"""
evaluate.py - Step 9: Evaluation & Metrics
============================================
Computes: Accuracy, Precision, Recall, F1-score, ROC-AUC
Generates: Confusion matrix, ROC curves
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from .config import Config


def get_predictions(model, loader, device=None):
    """Get model predictions and true labels from a DataLoader."""
    if device is None:
        device = torch.device(Config.DEVICE)
    model.eval()

    all_probs = []
    all_labels = []
    all_attn = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            # Check if model supports attention output
            if hasattr(model, 'attention') and hasattr(model, 'forward'):
                try:
                    output, attn = model(X_batch, return_attention=True)
                    all_attn.append(attn.cpu().numpy())
                except TypeError:
                    output = model(X_batch)
            else:
                output = model(X_batch)

            probs = torch.sigmoid(output).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy().flatten())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    attn_weights = np.concatenate(all_attn, axis=0) if all_attn else None

    return probs, labels, attn_weights


def compute_metrics(probs, labels, threshold=0.5):
    """Compute all classification metrics."""
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    return metrics


def evaluate_model(model, test_loader, model_name="Model"):
    """Full evaluation of a model on the test set."""
    device = torch.device(Config.DEVICE)
    probs, labels, attn = get_predictions(model, test_loader, device)
    metrics = compute_metrics(probs, labels)

    print(f"\n--- {model_name} Test Results ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    return metrics, probs, labels, attn


def plot_confusion_matrix(labels, probs, model_name, save_dir=None, threshold=0.5):
    """Plot and save confusion matrix."""
    if save_dir is None:
        save_dir = Config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Stampede'],
                yticklabels=['Normal', 'Stampede'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name}.png'), dpi=150)
    plt.close()


def plot_roc_curves(results_dict, save_dir=None):
    """Plot ROC curves for all models on one figure."""
    if save_dir is None:
        save_dir = Config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (metrics, probs, labels) in results_dict.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = metrics["roc_auc"]
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves_comparison.png'), dpi=150)
    plt.close()


def plot_training_history(history, model_name, save_dir=None):
    """Plot training/validation loss and accuracy curves."""
    if save_dir is None:
        save_dir = Config.FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label='Train', linewidth=2)
    ax1.plot(epochs, history["val_loss"], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label='Train', linewidth=2)
    ax2.plot(epochs, history["val_acc"], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_{model_name}.png'), dpi=150)
    plt.close()
