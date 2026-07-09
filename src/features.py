"""
features.py - Step 4: Feature Extraction
==========================================
Baseline features (from base paper): Entropy, TOV, KDE
Novel features (our contribution): mean_mag, std_mag, skewness,
                                    motion_coherence, direction_variance, crowd_density
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
from .config import Config
from .optical_flow import compute_activity_map


# ── Baseline Features (Base Paper) ──────────────────────────────────────

def compute_entropy(activity_map):
    """Shannon entropy of activity map pixel distribution.

    Higher entropy = more chaotic/uniformly distributed movement.
    Stampede events tend to show different entropy patterns.
    """
    flat = activity_map.flatten()
    # Create histogram and normalize to probabilities (must sum to 1)
    hist, _ = np.histogram(flat, bins=256, range=(0, 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total  # proper probability distribution
    probs = probs[probs > 0]  # remove zero bins
    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_tov(activity_map, threshold=None):
    """Temporal Occupancy of Vision (TOV).

    Fraction of pixels in the activity map above a threshold.
    High TOV = lots of movement across the scene.
    """
    if threshold is None:
        threshold = Config.TOV_THRESHOLD
    return float(np.mean(activity_map > threshold))


def compute_kde_peak(angles_window, magnitudes_window=None):
    """KDE of optical flow angles - returns peak density value.

    Measures directional distribution of movement.
    High peak = movement in consistent direction.
    Low peak = chaotic multi-directional movement (panic).
    """
    # Collect all angle values from the window
    all_angles = np.concatenate([a.flatten() for a in angles_window])

    # If magnitudes available, filter out near-zero flow (noise)
    if magnitudes_window is not None:
        all_mags = np.concatenate([m.flatten() for m in magnitudes_window])
        mask = all_mags > 0.5  # only consider meaningful movement
        all_angles = all_angles[mask]

    if len(all_angles) < 10:
        return 0.0

    # Subsample for speed (KDE on millions of pixels is very slow)
    if len(all_angles) > 5000:
        idx = np.random.choice(len(all_angles), 5000, replace=False)
        all_angles = all_angles[idx]

    try:
        kde = gaussian_kde(all_angles, bw_method=Config.KDE_BANDWIDTH)
        x = np.linspace(-np.pi, np.pi, 360)
        density = kde(x)
        return float(np.max(density))
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


# ── Novel Features (Our Contribution) ──────────────────────────────────

def compute_mean_magnitude(magnitudes_window):
    """Mean optical flow magnitude across the window.
    Higher = more overall movement."""
    all_mags = np.concatenate([m.flatten() for m in magnitudes_window])
    return float(np.mean(all_mags))


def compute_std_magnitude(magnitudes_window):
    """Standard deviation of optical flow magnitude.
    Higher = more variable/erratic movement."""
    all_mags = np.concatenate([m.flatten() for m in magnitudes_window])
    return float(np.std(all_mags))


def compute_skewness(magnitudes_window):
    """Skewness of optical flow magnitude distribution.
    Positive skew = few areas with very high movement (panic onset)."""
    all_mags = np.concatenate([m.flatten() for m in magnitudes_window])
    return float(scipy_stats.skew(all_mags))


def compute_motion_coherence(angles_window):
    """Motion coherence score - detects panic directional inconsistency.

    Uses circular statistics: R = |mean resultant vector| / N.
    High coherence (near 1) = everyone moving same direction (normal crowd flow).
    Low coherence (near 0) = chaotic directions (panic/stampede).
    """
    all_angles = np.concatenate([a.flatten() for a in angles_window])
    cos_sum = np.sum(np.cos(all_angles))
    sin_sum = np.sum(np.sin(all_angles))
    n = len(all_angles)
    if n == 0:
        return 0.0
    coherence = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
    return float(coherence)


def compute_direction_variance(angles_window):
    """Circular variance of flow directions.
    High variance = inconsistent movement (stampede indicator).
    direction_variance = 1 - coherence."""
    return 1.0 - compute_motion_coherence(angles_window)


def compute_crowd_density(magnitudes_window):
    """Crowd density estimate using foreground occupancy.

    Pixels with flow above (mean + 0.5*std) are considered 'active'.
    High density = crowded scene with lots of moving people.
    """
    all_mags = np.concatenate([m.flatten() for m in magnitudes_window])
    threshold = np.mean(all_mags) + 0.5 * np.std(all_mags)
    # Compute per-frame density and average
    densities = []
    for m in magnitudes_window:
        densities.append(np.mean(m > threshold))
    return float(np.mean(densities))


# ── Feature Extraction Pipeline ────────────────────────────────────────

def extract_window_features(magnitudes_window, angles_window, use_novel=True):
    """Extract all features for one temporal window.

    Args:
        magnitudes_window: list of L magnitude arrays
        angles_window: list of L angle arrays
        use_novel: if True, include novel features (default True)

    Returns:
        feature_vector: numpy array of shape (num_features,)
        feature_names: list of feature name strings
    """
    # Compute activity map for this window
    activity_map = compute_activity_map(magnitudes_window)

    # Baseline features (base paper)
    entropy = compute_entropy(activity_map)
    tov = compute_tov(activity_map)
    kde = compute_kde_peak(angles_window, magnitudes_window)

    features = [entropy, tov, kde]
    names = ["entropy", "tov", "kde"]

    if use_novel:
        # Novel features (our contribution)
        mean_mag = compute_mean_magnitude(magnitudes_window)
        std_mag = compute_std_magnitude(magnitudes_window)
        skewness = compute_skewness(magnitudes_window)
        coherence = compute_motion_coherence(angles_window)
        dir_var = compute_direction_variance(angles_window)
        density = compute_crowd_density(magnitudes_window)

        features.extend([mean_mag, std_mag, skewness, coherence, dir_var, density])
        names.extend(["mean_mag", "std_mag", "skewness",
                       "motion_coherence", "direction_variance", "crowd_density"])

    return np.array(features, dtype=np.float32), names


def extract_all_features(magnitudes, angles, fps, use_novel=True):
    """Extract features for all temporal windows from a video.

    Each window is L = FPS/2 frames (0.5 seconds).

    Args:
        magnitudes: list of (H,W) arrays from optical flow
        angles: list of (H,W) arrays from optical flow
        fps: frames per second
        use_novel: include novel features

    Returns:
        feature_matrix: (num_windows, num_features) numpy array
        feature_names: list of feature names
    """
    L = max(int(fps / 2), 2)  # window size in frames
    num_flows = len(magnitudes)
    feature_list = []
    names = None

    # Sliding window with step 1 (frame-by-frame)
    for start in range(0, num_flows - L + 1):
        mag_window = magnitudes[start:start + L]
        ang_window = angles[start:start + L]

        feat_vec, names = extract_window_features(mag_window, ang_window, use_novel)
        feature_list.append(feat_vec)

    if not feature_list:
        return np.array([]), names

    return np.stack(feature_list, axis=0), names
