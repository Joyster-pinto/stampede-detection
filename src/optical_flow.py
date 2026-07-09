"""
optical_flow.py - Step 3: Compute Dense Optical Flow
=====================================================
Uses Farneback method (OpenCV) to compute:
  - Flow magnitude (how much each pixel moved)
  - Flow angle (direction of movement)
"""

import cv2
import numpy as np
from .config import Config


def compute_optical_flow(prev_frame, curr_frame):
    """Compute dense optical flow between two consecutive frames.

    Uses Gunnar-Farneback method (analyzes every pixel, not just corners).

    Args:
        prev_frame: grayscale frame (H, W), uint8
        curr_frame: grayscale frame (H, W), uint8

    Returns:
        magnitude: (H, W) array of flow magnitudes
        angle: (H, W) array of flow angles in radians [-pi, pi]
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame, curr_frame,
        flow=None,
        pyr_scale=Config.OF_PYR_SCALE,
        levels=Config.OF_LEVELS,
        winsize=Config.OF_WINSIZE,
        iterations=Config.OF_ITERATIONS,
        poly_n=Config.OF_POLY_N,
        poly_sigma=Config.OF_POLY_SIGMA,
        flags=0
    )
    # flow shape: (H, W, 2) - horizontal and vertical components
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Convert angle from [0, 2*pi] to [-pi, pi]
    angle = angle - np.pi
    return magnitude, angle


def compute_flow_sequence(frames):
    """Compute optical flow for a sequence of frames.

    Args:
        frames: list of preprocessed grayscale frames

    Returns:
        magnitudes: list of (H, W) magnitude arrays (len = N-1)
        angles: list of (H, W) angle arrays (len = N-1)
    """
    magnitudes = []
    angles = []
    for i in range(len(frames) - 1):
        mag, ang = compute_optical_flow(frames[i], frames[i + 1])
        magnitudes.append(mag)
        angles.append(ang)
    return magnitudes, angles


def compute_activity_map(magnitudes_window):
    """Compute activity map from a window of magnitude frames.

    Activity map = average magnitude over L frames, normalized.
    The base paper normalizes by the max pixel value (not 255)
    to adapt to different environments.

    Args:
        magnitudes_window: list of L magnitude arrays

    Returns:
        activity_map: (H, W) normalized activity map in [0, 1]
    """
    stacked = np.stack(magnitudes_window, axis=0)  # (L, H, W)
    activity_map = np.mean(stacked, axis=0)          # (H, W)

    # Normalize by max value (as per base paper)
    max_val = activity_map.max()
    if max_val > 0:
        activity_map = activity_map / max_val
    return activity_map
