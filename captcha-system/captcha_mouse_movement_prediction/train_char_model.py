# captcha_mouse_movement_prediction/train_char_model.py

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle

# Import your existing stroke utilities
from .utils import (
    normalize_strokes,
    extract_features,
    FEATURE_NAMES_KINEMATIC,
    SYMBOLS,
)


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    """
    Load labeled stroke data from a JSONL file (one JSON object per line).

    Each line should look like:
      {"label": "A", "strokes": [...]}
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    samples: List[Dict[str, Any]] = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(obj)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping invalid JSON line: {line[:80]}...")

    if not samples:
        raise RuntimeError(f"No valid samples found in dataset: {json_path}")

    print(f"[INFO] Loaded {len(samples)} samples from {json_path}")
    return samples


def strokes_to_features(sample: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Convert a single labeled sample into:
      - feature vector (np.array of length len(FEATURE_NAMES_KINEMATIC))
      - label index (int, index into SYMBOLS)
    """
    label = sample.get("label")
    strokes = sample.get("strokes")

    if label is None or strokes is None:
        raise ValueError("Sample missing 'label' or 'strokes' field.")

    if label not in SYMBOLS:
        raise ValueError(
            f"Label '{label}' not found in SYMBOLS. "
            f"Valid symbols are: {SYMBOLS}"
        )

    # Normalize and extract features using your existing pipeline
    arr = normalize_strokes(strokes)
    feats = extract_features(arr, strokes)

    # Build feature vector in the same order used in main.py
    vec = np.array([feats[k] for k in FEATURE_NAMES_KINEMATIC], dtype=np.float32)

    # Map label → index
    label_idx = SYMBOLS.index(label)

    return vec, label_idx


def build_feature_matrix(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of samples into:
      X: (n_samples, n_features)
      y: (n_samples,)
    """
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for i, sample in enumerate(samples):
        try:
            vec, label_idx = strokes_to_features(sample)
            X_list.append(vec)
            y_list.append(label_idx)
        except Exception as e:
            print(f"[WARN] Skipping sample {i} due to error: {e}")

    if not X_list:
        raise RuntimeError("No valid samples after feature extraction.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)

    print(f"[INFO] Feature matrix shape: {X.shape}, labels shape: {y.shape}")
    return X, y


def train_char_model(
    dataset_path: str,
    model_out_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Train an XGBoost-based character recognition model on stroke features.

    Args:
        dataset_path: Path to JSONL dataset file.
        model_out_path: Where to save the trained model (.pkl).
        test_size: Fraction of data used for test set.
    """
    print(f"[INFO] Loading dataset from: {dataset_path}")
    samples = load_dataset(dataset_path)

    print("[INFO] Building feature matrix...")
    X, y = build_feature_matrix(samples)

    # Simple stratified train/test split so you get metrics per class
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    num_classes = len(SYMBOLS)
    print(f"[INFO] Training model for {num_classes} classes.")

    # XGBoost multi-class classifier
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=4,
        tree_method="hist",  # fast and CPU-friendly
    )

    print("[INFO] Fitting model...")
    model.fit(X_train, y_train)

    print("[INFO] Evaluating on test set...")
    y_pred = model.predict(X_test)

    # Print metrics
    label_map = {i: ch for i, ch in enumerate(SYMBOLS)}
    print("\n[INFO] Label index → symbol mapping:")
    print(label_map)

    print("\n[INFO] Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\n[INFO] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)

    with open(model_out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n[INFO] Saved character recognition model to: {model_out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stroke-based character recognition model for the CAPTCHA system."
    )
    parser.add_argument(
        "--data",
        default="captcha_mouse_movement_prediction/data/char_strokes_dataset.json",
        help="Path to JSONL dataset file (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="captcha_mouse_movement_prediction/models/char_recognition_model.pkl",
        help="Path to save trained model (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_char_model(
        dataset_path=args.data,
        model_out_path=args.out,
        test_size=args.test_size,
    )
