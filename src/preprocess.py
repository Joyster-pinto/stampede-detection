"""
preprocess.py - Step 2: Video Preprocessing
============================================
For every video:
  1. Extract frames
  2. Resize to 320x240
  3. Convert to grayscale
  4. Apply Gaussian blur
  5. Return/save processed frames
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
from .config import Config


def preprocess_frame(frame):
    """Preprocess a single frame: resize → grayscale → blur."""
    # Resize to 320x240
    frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    # Convert to grayscale
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to treat crowd as a whole unit
    frame = cv2.GaussianBlur(frame, Config.GAUSSIAN_BLUR_KERNEL, 0)
    return frame


def extract_frames_from_video(video_path):
    """Extract and preprocess all frames from a video file.

    Returns:
        frames: list of preprocessed grayscale frames (H, W)
        fps: frames per second of the video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # default fallback
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))

    cap.release()
    print(f"  Extracted {len(frames)} frames at {fps:.1f} FPS from {os.path.basename(video_path)}")
    return frames, fps


def extract_frames_from_folder(folder_path):
    """Extract and preprocess frames from a folder of images (e.g. UCSD, Avenue).

    Returns:
        frames: list of preprocessed grayscale frames
        fps: estimated FPS (default 30 if unknown)
    """
    valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if not files:
        raise FileNotFoundError(f"No image files in {folder_path}")

    frames = []
    for f in files:
        img = cv2.imread(os.path.join(folder_path, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
            img = cv2.GaussianBlur(img, Config.GAUSSIAN_BLUR_KERNEL, 0)
            frames.append(img)

    fps = 30  # UCSD and Avenue are typically ~30fps
    print(f"  Loaded {len(frames)} frames from {folder_path}")
    return frames, fps


def load_umn_videos(umn_dir=None):
    """Load all UMN dataset videos.

    Expected structure:
        UMN/
          ├── Crowd-Activity-All.avi   (or individual scene files)
          ├── scene1.avi
          ├── scene2.avi
          └── scene3.avi

    Returns:
        list of (video_name, frames, fps, ground_truth_labels)
    """
    if umn_dir is None:
        umn_dir = Config.UMN_DIR

    videos = []
    video_files = sorted([
        f for f in os.listdir(umn_dir)
        if f.endswith(('.avi', '.mp4', '.mkv'))
    ])

    if not video_files:
        raise FileNotFoundError(f"No video files found in {umn_dir}")

    for vf in video_files:
        path = os.path.join(umn_dir, vf)
        frames, fps = extract_frames_from_video(path)
        videos.append((vf, frames, fps))
        
    print(f"Loaded {len(videos)} UMN videos")
    return videos


def load_ucsd_clips(ucsd_dir=None, subset="UCSDped2"):
    """Load UCSD dataset clips (frame folders).

    Expected structure:
        UCSD/
          ├── UCSDped1/
          │   ├── Train/ (Train001/, Train002/, ...)
          │   └── Test/  (Test001/, Test002/, ...)
          └── UCSDped2/
              ├── Train/
              └── Test/
    """
    if ucsd_dir is None:
        ucsd_dir = Config.UCSD_DIR

    subset_dir = os.path.join(ucsd_dir, subset)
    clips = []

    for split in ["Train", "Test"]:
        split_dir = os.path.join(subset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        clip_dirs = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        for cd in clip_dirs:
            clip_path = os.path.join(split_dir, cd)
            frames, fps = extract_frames_from_folder(clip_path)
            is_test = (split == "Test")
            clips.append((f"{subset}/{split}/{cd}", frames, fps, is_test))

    print(f"Loaded {len(clips)} UCSD clips from {subset}")
    return clips


def load_avenue_clips(avenue_dir=None):
    """Load Avenue dataset.

    Expected structure:
        Avenue/
          ├── training_videos/ (01.avi, 02.avi, ...)
          └── testing_videos/  (01.avi, 02.avi, ...)
    """
    if avenue_dir is None:
        avenue_dir = Config.AVENUE_DIR

    clips = []
    for split in ["training_videos", "testing_videos"]:
        split_dir = os.path.join(avenue_dir, split)
        if not os.path.isdir(split_dir):
            continue
        video_files = sorted([
            f for f in os.listdir(split_dir)
            if f.endswith(('.avi', '.mp4'))
        ])
        for vf in video_files:
            path = os.path.join(split_dir, vf)
            frames, fps = extract_frames_from_video(path)
            is_test = ("testing" in split)
            clips.append((f"Avenue/{split}/{vf}", frames, fps, is_test))

    print(f"Loaded {len(clips)} Avenue clips")
    return clips
