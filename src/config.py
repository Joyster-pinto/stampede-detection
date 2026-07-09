"""
config.py - Central configuration for Stampede Detection Pipeline
=================================================================
All hyperparameters are here. Modify these to tune performance.
Based on: "Stampede detector based on deep learning models using dense optical flow"
"""

import os
import torch


class Config:
    """All tunable parameters in one place."""

    # ── Paths ──────────────────────────────────────────────────────────
    BASE_DIR = r"D:\Research"
    DATASET_DIR = os.path.join(BASE_DIR, "datasets")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
    FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

    UMN_DIR = os.path.join(DATASET_DIR, "UMN")
    UCSD_DIR = os.path.join(DATASET_DIR, "UCSD")
    AVENUE_DIR = os.path.join(DATASET_DIR, "Avenue")

    # ── Preprocessing (Step 2) ─────────────────────────────────────────
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    GAUSSIAN_BLUR_KERNEL = (5, 5)

    # ── Optical Flow - Farneback (Step 3) ──────────────────────────────
    OF_PYR_SCALE = 0.5       # pyramid scale (<1 to build pyramids)
    OF_LEVELS = 3             # number of pyramid layers
    OF_WINSIZE = 15           # averaging window size
    OF_ITERATIONS = 3         # iterations at each pyramid level
    OF_POLY_N = 5             # polynomial expansion neighborhood size
    OF_POLY_SIGMA = 1.2       # Gaussian std for polynomial expansion

    # ── Feature Extraction (Step 4) ────────────────────────────────────
    TOV_THRESHOLD = 0.3       # activity map threshold for TOV
    KDE_BANDWIDTH = 0.5       # bandwidth for angle KDE
    ACTIVITY_MAP_NORM = True  # normalize activity map by max pixel value

    # ── Temporal Windows (Step 5) ──────────────────────────────────────
    # Window size L = FPS / 2 (set dynamically per video)
    SEQUENCE_LENGTH = 10      # number of windows per input sequence
    STRIDE = 1                # sliding window stride for sequences

    # ── Feature dimensions ─────────────────────────────────────────────
    BASELINE_FEATURES = 3     # Entropy, TOV, KDE
    NOVEL_FEATURES = 6        # mean_mag, std_mag, skewness, coherence, dir_var, density
    TOTAL_FEATURES = 9        # baseline + novel

    # ── Model Architecture (Steps 6-7) ─────────────────────────────────
    HIDDEN_SIZE = 128         # LSTM/GRU hidden dimension
    NUM_LAYERS = 2            # number of recurrent layers
    DROPOUT = 0.3             # dropout rate
    NUM_ATTENTION_UNITS = 64  # attention layer internal size

    # ── Training (Step 8) ──────────────────────────────────────────────
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4       # AdamW weight decay
    EPOCHS = 100
    PATIENCE = 15             # early stopping patience
    SCHEDULER = "cosine"      # "cosine" | "step" | "plateau"
    LABEL_SMOOTHING = 0.1     # label smoothing factor
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ── Device ─────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        for d in [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.FIGURE_DIR,
                  cls.RESULTS_DIR, cls.FEATURES_DIR]:
            os.makedirs(d, exist_ok=True)
