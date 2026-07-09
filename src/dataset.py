"""
dataset.py - Step 5: Create Temporal Sequences + PyTorch Dataset
=================================================================
Groups feature vectors into sequences for the temporal model.
Each sequence = SEQUENCE_LENGTH consecutive feature windows.
Labels: 0 = normal, 1 = stampede/anomaly.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .config import Config


class StampedeDataset(Dataset):
    """PyTorch dataset for stampede detection sequences."""

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: (N, seq_len, num_features) numpy array
            labels: (N,) numpy array of 0/1 labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_sequences(feature_matrix, labels_per_window, seq_len=None, stride=None):
    """Create overlapping sequences from feature windows.

    Args:
        feature_matrix: (num_windows, num_features) array
        labels_per_window: (num_windows,) array - 0=normal, 1=anomaly
        seq_len: number of windows per sequence
        stride: step between consecutive sequences

    Returns:
        X: (num_sequences, seq_len, num_features) array
        y: (num_sequences,) array - label is 1 if ANY window in sequence is anomaly
    """
    if seq_len is None:
        seq_len = Config.SEQUENCE_LENGTH
    if stride is None:
        stride = Config.STRIDE

    num_windows = len(feature_matrix)
    if num_windows < seq_len:
        return np.array([]), np.array([])

    X, y = [], []
    for start in range(0, num_windows - seq_len + 1, stride):
        seq = feature_matrix[start:start + seq_len]
        # Label: 1 if any window in the sequence is anomalous
        label = 1.0 if np.any(labels_per_window[start:start + seq_len] == 1) else 0.0
        X.append(seq)
        y.append(label)

    return np.array(X), np.array(y)


def generate_umn_labels(num_windows, anomaly_ratio=0.3):
    """Generate frame-level labels for UMN videos.

    UMN videos start with normal behavior and transition to panic.
    The last ~30% of each video is typically the anomalous portion.

    Args:
        num_windows: total number of feature windows
        anomaly_ratio: fraction of windows that are anomalous (at the end)

    Returns:
        labels: (num_windows,) array of 0/1
    """
    labels = np.zeros(num_windows, dtype=np.float32)
    anomaly_start = int(num_windows * (1 - anomaly_ratio))
    labels[anomaly_start:] = 1.0
    return labels


def normalize_features(X_train, X_val=None, X_test=None):
    """Normalize features using StandardScaler (fit on train only).

    Args:
        X_train: (N, seq_len, features) training sequences
        X_val: optional validation sequences
        X_test: optional test sequences

    Returns:
        Normalized arrays and the fitted scaler
    """
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()

    # Fit on training data (reshape to 2D, fit, reshape back)
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    X_train_norm = scaler.transform(X_train_2d).reshape(n_train, seq_len, n_feat)

    results = [X_train_norm, scaler]

    if X_val is not None:
        n_val = X_val.shape[0]
        X_val_norm = scaler.transform(X_val.reshape(-1, n_feat)).reshape(n_val, seq_len, n_feat)
        results.insert(1, X_val_norm)

    if X_test is not None:
        n_test = X_test.shape[0]
        X_test_norm = scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, seq_len, n_feat)
        results.insert(-1, X_test_norm)

    return tuple(results)


def prepare_dataloaders(X, y, batch_size=None):
    """Split data and create PyTorch DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE

    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        random_state=Config.SEED, stratify=y
    )
    relative_test = Config.TEST_RATIO / (Config.VAL_RATIO + Config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=Config.SEED, stratify=y_temp
    )

    # Normalize
    X_train_n, X_val_n, X_test_n, scaler = normalize_features(X_train, X_val, X_test)

    # Create datasets
    train_ds = StampedeDataset(X_train_n, y_train)
    val_ds = StampedeDataset(X_val_n, y_val)
    test_ds = StampedeDataset(X_test_n, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Data split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Train class balance: {y_train.mean():.2%} anomaly")

    return train_loader, val_loader, test_loader, scaler
