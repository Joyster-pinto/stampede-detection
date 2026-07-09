"""
train.py - Step 8: Training Pipeline
======================================
Trains all 4 models: LSTM, GRU, BiLSTM, BiLSTM+Attention
Includes: early stopping, learning rate scheduling, label smoothing.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from .config import Config
from .models import LSTMClassifier, GRUClassifier, BiLSTMClassifier, BiLSTMAttentionClassifier


def get_model(model_name, input_size):
    """Create a model by name."""
    models = {
        "LSTM": LSTMClassifier,
        "GRU": GRUClassifier,
        "BiLSTM": BiLSTMClassifier,
        "BiLSTM_Attention": BiLSTMAttentionClassifier,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    kwargs = dict(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    )
    if model_name == "BiLSTM_Attention":
        kwargs["attention_units"] = Config.NUM_ATTENTION_UNITS
    return models[model_name](**kwargs)


def get_scheduler(optimizer, scheduler_name):
    """Create learning rate scheduler."""
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    elif scheduler_name == "step":
        return StepLR(optimizer, step_size=20, gamma=0.5)
    elif scheduler_name == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(output) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            output = model(X_batch)
            loss = criterion(output, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_loss / total, correct / total


def train_model(model_name, train_loader, val_loader, input_size, save_path=None):
    """Full training pipeline for a single model.

    Args:
        model_name: "LSTM", "GRU", "BiLSTM", or "BiLSTM_Attention"
        train_loader, val_loader: PyTorch DataLoaders
        input_size: number of input features
        save_path: path to save the best model

    Returns:
        model: trained model
        history: dict with training/validation loss and accuracy per epoch
    """
    device = torch.device(Config.DEVICE)
    model = get_model(model_name, input_size).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training {model_name} | Parameters: {n_params:,} | Device: {device}")
    print(f"{'='*60}")

    # Loss with label smoothing
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0]).to(device)  # adjust if imbalanced
    )

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                      weight_decay=Config.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, Config.SCHEDULER)

    # Training loop with early stopping
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {lr:.6f}")

        if patience_counter >= Config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s | Best val loss: {best_val_loss:.4f}")

    # Load best model
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))

    return model, history


def train_all_models(train_loader, val_loader, input_size):
    """Train all 4 models for comparison (Step 8).

    Returns:
        results: dict mapping model_name -> (model, history)
    """
    Config.ensure_dirs()
    results = {}

    for name in ["LSTM", "GRU", "BiLSTM", "BiLSTM_Attention"]:
        save_path = os.path.join(Config.MODEL_DIR, f"{name}_best.pt")
        model, history = train_model(name, train_loader, val_loader,
                                     input_size, save_path)
        results[name] = (model, history)

    return results
