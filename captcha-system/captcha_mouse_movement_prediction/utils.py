from typing import Any, Dict, List

import numpy as np
import yaml

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# Time gap > 450ms implies the user lifted their hand to write the next letter
THRESHOLD_BETWEEN_CHARS_MS = 450 

FEATURE_NAMES_KINEMATIC = [
    "vel_mean", "vel_std", "vel_max",
    "acc_mean", "acc_std",
    "jerk_mean", "jerk_std",
    "pause_mean_ms", "pause_std_ms", 
    "secret_tremor_energy" # Crucial for Human vs Bot
]

FEATURE_NAMES_SHAPE = [
    "num_strokes", "total_length", "aspect_ratio",
    "duration_ms", "points_per_stroke_mean",
    "curvature_mean"
]

ALL_FEATURE_NAMES = FEATURE_NAMES_KINEMATIC + FEATURE_NAMES_SHAPE

# ------------------------------------------------------------------
# Data Parsing & Normalization
# ------------------------------------------------------------------
def parse_strokes(yaml_str: str) -> List[List[Dict]]:
    """Parses the HWRT YAML string format into Python lists."""
    try:
        data = yaml.safe_load(yaml_str)
        strokes = []
        for stroke in data:
            # Standardize to list of dicts
            points = [{'x': float(p['x']), 'y': float(p['y']), 'time': int(p['time'])} for p in stroke]
            strokes.append(points)
        return strokes
    except Exception as e:
        print(f"[Error] parsing strokes: {e}")
        return []

def normalize_strokes(strokes_raw: List[List[Dict]]) -> np.ndarray:
    """
    Normalizes raw strokes into a unit square (0-1) and relative time (ms).
    Returns Nx3 numpy array [x, y, time].
    """
    flat = [p for s in strokes_raw for p in s]
    if len(flat) < 2:
        return np.array([])

    arr = np.array([[p['x'], p['y'], p['time']] for p in flat])

    # 1. Relative Time (start at 0)
    arr[:, 2] -= arr[0, 2]

    # 2. Spatial Normalization (Unit Square)
    min_xy = arr[:, :2].min(axis=0)
    max_xy = arr[:, :2].max(axis=0)
    dims = max_xy - min_xy
    # Prevent divide by zero if user draws a dot
    scale = np.maximum(dims, 1.0) 
    
    arr[:, :2] = (arr[:, :2] - min_xy) / scale
    return arr

def extract_features(arr: np.ndarray, strokes_raw: List[List[Dict]]) -> Dict[str, float]:
    """
    Extracts biometric features.
    Input: Normalized Nx3 array, Raw strokes list.
    """
    if arr is None or len(arr) < 3:
        return {k: 0.0 for k in ALL_FEATURE_NAMES}

    pos = arr[:, :2]
    t = arr[:, 2]

    # Derivatives
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-5, dt) # Safety: replace 0 with epsilon

    dxdy = np.diff(pos, axis=0)
    dist_segments = np.sqrt(np.sum(dxdy**2, axis=1))
    
    # Velocity
    speed = dist_segments / dt
    vel_mean = np.mean(speed)
    vel_std  = np.std(speed)
    vel_max  = np.max(speed)

    # Acceleration
    acc = np.diff(speed) / dt[1:]
    acc_mean = np.mean(np.abs(acc)) if len(acc) > 0 else 0
    acc_std  = np.std(acc) if len(acc) > 0 else 0

    # Jerk (Change in acceleration - Humans have high jerk, Bots are smooth)
    jerk = np.diff(acc) / dt[2:] if len(acc) > 1 else np.array([0])
    jerk_mean = np.mean(np.abs(jerk))
    jerk_std  = np.std(jerk)

    # Pauses
    pauses = []
    if len(strokes_raw) > 1:
        for i in range(len(strokes_raw) - 1):
            # Time between end of stroke A and start of stroke B
            p = strokes_raw[i+1][0]['time'] - strokes_raw[i][-1]['time']
            pauses.append(p)
    pause_mean = np.mean(pauses) if pauses else 0
    pause_std  = np.std(pauses) if pauses else 0

    # Secret Tremor (FFT Energy)
    # Humans have physiological tremor (8-12Hz). Bots following a curve do not.
    deviations = np.linalg.norm(dxdy, axis=1)
    if len(deviations) > 10:
        fft_vals = np.abs(np.fft.rfft(deviations - np.mean(deviations)))
        # Sum energy in high-frequency bands
        secret_tremor_energy = np.sum(fft_vals[5:]) 
    else:
        secret_tremor_energy = 0

    # Shape / Curvature
    # Calculate angle changes
    angles = np.arctan2(dxdy[:, 1], dxdy[:, 0])
    angle_changes = np.diff(angles)
    # Normalize angles to -pi to pi
    angle_changes = np.arctan2(np.sin(angle_changes), np.cos(angle_changes))
    curvature_mean = np.mean(np.abs(angle_changes)) if len(angle_changes) > 0 else 0

    width = pos[:, 0].max() - pos[:, 0].min()
    height = pos[:, 1].max() - pos[:, 1].min()
    
    feats = {
        "vel_mean": float(vel_mean), "vel_std": float(vel_std), "vel_max": float(vel_max),
        "acc_mean": float(acc_mean), "acc_std": float(acc_std),
        "jerk_mean": float(jerk_mean), "jerk_std": float(jerk_std),
        "pause_mean_ms": float(pause_mean), "pause_std_ms": float(pause_std),
        "secret_tremor_energy": float(secret_tremor_energy),
        "num_strokes": float(len(strokes_raw)), 
        "total_length": float(np.sum(dist_segments)),
        "aspect_ratio": float(width / (height + 1e-6)), 
        "duration_ms": float(t[-1]),
        "points_per_stroke_mean": float(np.mean([len(s) for s in strokes_raw])),
        "curvature_mean": float(curvature_mean)
    }
        
    return feats

# ------------------------------------------------------------------
# Segmentation Logic
# ------------------------------------------------------------------
def segment_into_characters(strokes_raw: List[List[Dict]]) -> List[List[List[Dict]]]:
    """
    Splits a long list of strokes into characters based on Time Gaps.
    Then sorts the characters left-to-right.
    """
    if not strokes_raw:
        return []

    # 1. Sort raw strokes by time just to be safe
    strokes_raw.sort(key=lambda s: s[0]['time'])

    char_groups = []
    current_group = [strokes_raw[0]]
    
    for i in range(1, len(strokes_raw)):
        prev_end = strokes_raw[i-1][-1]['time']
        curr_start = strokes_raw[i][0]['time']
        gap = curr_start - prev_end
        
        if gap > THRESHOLD_BETWEEN_CHARS_MS:
            char_groups.append(current_group)
            current_group = [strokes_raw[i]]
        else:
            current_group.append(strokes_raw[i])
    
    if current_group:
        char_groups.append(current_group)

    # 2. Sort groups spatially (Left to Right)
    # We use the average X position of the character to sort
    def get_avg_x(group):
        xs = [p['x'] for s in group for p in s]
        return sum(xs) / len(xs)

    char_groups.sort(key=get_avg_x)
    return char_groups
