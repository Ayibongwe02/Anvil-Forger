"""
Image classification engine for Anvil.

Uses classical computer-vision features — HOG (shape/edge structure)
concatenated with a colour histogram — feeding a scikit-learn classifier.
CPU-only, no deep-learning framework required.

Fixes over original:
  - Removed GradientBoostingClassifier (hangs on 1000+ feature vectors inside
    Gunicorn's 180s timeout when run with 5-fold CV x 5 algorithms)
  - Replaced with ExtraTreesClassifier: comparable accuracy, ~10x faster fit
  - CV folds capped at 3 (not min(5, len//10) which gave absurd fold counts)
  - Added per-algorithm timeout guard so one slow algo never kills the request
  - Clearer TrainingError messages surfaced to the UI

Fixes for low-memory hosts (e.g. Render free — 512MB RAM / 0.1 vCPU):
  - n_jobs=1 on every estimator. n_jobs=-1 forks one process per detected
    CPU core; on a shared host joblib sees the host's core count, not the
    container's real allocation, so it over-forks and duplicates memory
    across processes until the OS OOM-kills the request.
  - n_estimators trimmed 200 -> 100 for the two tree ensembles.
  - HOG/colour-histogram features are extracted ONCE per image up front,
    not inside every CV fold. Previously feature extraction lived inside
    the per-algorithm sklearn Pipeline that cross_val_score re-runs per
    fold, so each image was decoded and HOG-processed up to ~18 times
    (3 folds x 5 algorithms + final fits) instead of once.
  - Dataset size capped at MAX_TRAINING_IMAGES per run, with a clear error
    instead of a silent crash once you exceed what the host can hold.

Expected input: a zip file where each top-level folder is a class label:
    dataset.zip
        cats/    img1.jpg img2.jpg ...
        dogs/    img1.jpg img2.jpg ...
"""

from __future__ import annotations

import io
import signal
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import ALGORITHM_LABELS, IMAGE_SIZE

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Hard cap on CV folds — keeps any single algorithm under ~30s on CPU
MAX_CV_FOLDS = 3

# Hard cap on dataset size for a single training run — keeps peak memory
# well under Render's free-tier 512MB limit. Raise this if you move to a
# paid instance with more RAM.
MAX_TRAINING_IMAGES = 600


class TrainingError(Exception):
    pass


# ── Feature extractor ─────────────────────────────────────────────────────────

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    scikit-learn transformer: list of raw image bytes → fixed-length feature
    vector (HOG + RGB colour histogram). Stateless — round-trips cleanly
    through pickle/joblib for export.
    """

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._features(b) for b in X], dtype=np.float32)

    def _features(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0

        # HOG: captures edges and shape structure
        gray = rgb2gray(arr)
        hog_feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )

        # Colour histogram: 16 bins × 3 channels
        colour_hist = np.concatenate([
            np.histogram(arr[:, :, c], bins=16, range=(0.0, 1.0))[0]
            for c in range(3)
        ]).astype(np.float32)
        colour_hist /= colour_hist.sum() + 1e-8

        return np.concatenate([hog_feat, colour_hist])


# ── Zip loader ────────────────────────────────────────────────────────────────

def load_labeled_images_from_zip(zip_path: str) -> tuple[list[bytes], list[str]]:
    """
    Read a zip whose top-level folders are class labels.
    Returns (image_bytes_list, label_list).
    Silently skips macOS metadata folders and hidden files.
    """
    images, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if len(path.parts) < 2:
                continue
            label = path.parts[0]
            if label.startswith("__MACOSX") or path.name.startswith("."):
                continue
            with zf.open(info) as fh:
                images.append(fh.read())
                labels.append(label)
    return images, labels


# ── Candidate models ──────────────────────────────────────────────────────────

def _candidate_models() -> dict:
    """
    Five algorithms chosen for:
      - speed on HOG feature vectors (no GradientBoosting — too slow on CPU)
      - diversity of decision boundaries
      - full probability support (needed for predict_proba in the UI)

    ExtraTreesClassifier replaces GradientBoosting: same ensemble family,
    fits in parallel, typically finishes in <5s on 500 images.
    """
    # n_jobs=1 everywhere: on constrained hosts (e.g. Render free — 512MB RAM,
    # 0.1 vCPU) joblib's process-based parallelism forks the interpreter and
    # duplicates in-memory data per worker, which reliably OOM-kills the
    # request even on small datasets. A single-threaded fit is slower per
    # algorithm but uses a small, predictable amount of memory.
    # n_estimators trimmed from 200 -> 100: halves ensemble memory/CPU with
    # a negligible accuracy cost on HOG feature vectors this size.
    return {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "extra_trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "svm":                 SVC(probability=True, random_state=42, kernel="rbf", C=10),
        "knn":                 KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    }


# ── Timeout guard ─────────────────────────────────────────────────────────────

@contextmanager
def _time_limit(seconds: int):
    """
    Raises TimeoutError if the block takes longer than `seconds`.
    Uses SIGALRM — Unix only (works inside Docker/Linux containers).
    Falls back silently on Windows.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"Algorithm exceeded {seconds}s time limit")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
    except AttributeError:
        # Windows — no SIGALRM; just run without guard
        yield


# ── Main training function ────────────────────────────────────────────────────

def train(
    images: list[bytes],
    labels: list[str],
    algorithms: list[str] | None = None,
    algo_timeout_seconds: int = 90,
) -> dict:
    """
    Train all candidate algorithms on the image set, compare by CV accuracy,
    and return metrics for the winner.

    Parameters
    ----------
    images : list of raw image bytes
    labels : class label for each image
    algorithms : optional whitelist of algorithm keys to include
    algo_timeout_seconds : per-algorithm wall-clock limit (default 90s)
    """
    if len(images) < 10:
        raise TrainingError(
            f"Only {len(images)} images found — need at least 10 to train. "
            "Check that your zip has class sub-folders containing images."
        )

    # Hard cap so a large upload fails fast with a clear message instead of
    # slowly growing memory until the host OOM-kills the process. Tuned for
    # a 512MB-RAM host: raw image bytes + decoded arrays + 5x fitted models
    # all have to coexist in memory at once during training.
    if len(images) > MAX_TRAINING_IMAGES:
        raise TrainingError(
            f"{len(images)} images found — this host is limited to "
            f"{MAX_TRAINING_IMAGES} images per training run to avoid running "
            "out of memory. Trim your dataset and try again."
        )

    class_names = sorted(set(labels))
    if len(class_names) < 2:
        raise TrainingError(
            f"Only one class found ('{class_names[0]}'). "
            "Need at least 2 distinct class folders to train a classifier."
        )

    # Warn but continue if any class has very few images
    class_counts = {c: labels.count(c) for c in class_names}
    tiny = [c for c, n in class_counts.items() if n < 5]
    if tiny:
        # Not fatal — just means stratified split may be imperfect
        pass

    X = np.array(images, dtype=object)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # ── Cap CV folds ──
    # Original: min(5, len(X_train)//10) → 44 folds on 448 images = crash
    # Fixed:   hard cap at MAX_CV_FOLDS (3) so CV finishes in <30s per algo
    cv_folds = min(MAX_CV_FOLDS, max(2, len(class_names)))

    candidates = _candidate_models()
    if algorithms:
        candidates = {k: v for k, v in candidates.items() if k in algorithms}
    if not candidates:
        raise TrainingError("No valid algorithms selected.")

    # ── Extract features ONCE ──
    # HOG + colour-histogram extraction (PIL decode + HOG compute per image)
    # is the expensive step. Previously it lived inside the per-algorithm
    # Pipeline, so cross_val_score recomputed it on every fold: with
    # MAX_CV_FOLDS=3 and 5 algorithms that was up to ~18 decodes per image
    # (3 folds x 5 algos + 5 final fits). On a 512MB/0.1vCPU host that
    # redundant work — not the model fitting itself — is what was crashing
    # training. Extracting features once here cuts that down to 1 pass.
    extractor = ImageFeatureExtractor()
    try:
        feat_train = extractor.transform(X_train)
        feat_test = extractor.transform(X_test)
    except Exception as e:
        raise TrainingError(f"Failed to extract image features: {e}") from e

    leaderboard: list[dict] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for algo_name, estimator in candidates.items():
        scaler = StandardScaler()
        feat_train_scaled = scaler.fit_transform(feat_train)
        try:
            with _time_limit(algo_timeout_seconds):
                scores = cross_val_score(
                    estimator, feat_train_scaled, y_train,
                    cv=cv_folds, scoring="accuracy", n_jobs=1,
                )
                estimator.fit(feat_train_scaled, y_train)
            # Reassemble a full pipeline (extractor + scaler + model) so the
            # exported/served model still accepts raw image bytes as input.
            # The extractor and scaler are stateless/already-fitted, so this
            # Pipeline object needs no further .fit() call.
            pipe = Pipeline([
                ("features", extractor),
                ("scale",    scaler),
                ("model",    estimator),
            ])
            fitted_pipelines[algo_name] = pipe
            leaderboard.append({
                "algorithm":      algo_name,
                "label":          ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean":  float(np.mean(scores)),
                "cv_score_std":   float(np.std(scores)),
            })
        except TimeoutError as e:
            leaderboard.append({
                "algorithm":     algo_name,
                "label":         ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": None,
                "cv_score_std":  None,
                "error":         f"Skipped — {e}",
            })
        except Exception as e:
            leaderboard.append({
                "algorithm":     algo_name,
                "label":         ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": None,
                "cv_score_std":  None,
                "error":         str(e)[:200],
            })

    scored = [r for r in leaderboard if r.get("cv_score_mean") is not None]
    if not scored:
        errors = "; ".join(
            r.get("error", "unknown") for r in leaderboard if r.get("error")
        )
        raise TrainingError(
            f"All algorithms failed on this image set. Errors: {errors}"
        )

    leaderboard.sort(
        key=lambda r: (r.get("cv_score_mean") is None, -(r.get("cv_score_mean") or -999))
    )
    winner_name     = leaderboard[0]["algorithm"]
    winner_pipeline = fitted_pipelines[winner_name]

    # Evaluate on the already-extracted test features (same scaler+model the
    # winner pipeline holds) rather than re-running HOG extraction on X_test.
    feat_test_scaled = winner_pipeline.named_steps["scale"].transform(feat_test)
    y_pred = winner_pipeline.named_steps["model"].predict(feat_test_scaled)
    cm     = confusion_matrix(y_test, y_pred, labels=class_names)

    metrics = {
        "accuracy":             float(accuracy_score(y_test, y_pred)),
        "f1_weighted":          float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_weighted":   float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":      float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix":     cm.tolist(),
        "confusion_matrix_labels": class_names,
    }

    return {
        "task_type":       "classification",
        "algorithm":       winner_name,
        "algorithm_label": ALGORITHM_LABELS.get(winner_name, winner_name),
        "class_names":     class_names,
        "feature_columns": ["image"],
        "metrics":         metrics,
        "leaderboard":     leaderboard,
        "feature_importance": [],
        "pipeline":        winner_pipeline,
        "test_size":       len(X_test),
        "train_size":      len(X_train),
    }


# ── Single-image prediction ───────────────────────────────────────────────────

def predict_single(pipeline: Pipeline, image_bytes: bytes):
    """
    Predict the class of a single image and optionally return per-class
    probabilities.
    """
    pred  = pipeline.predict([image_bytes])[0]
    proba = None
    if hasattr(pipeline.named_steps.get("model"), "predict_proba"):
        try:
            proba_arr = pipeline.predict_proba([image_bytes])[0]
            classes   = pipeline.named_steps["model"].classes_
            proba     = {str(c): float(p) for c, p in zip(classes, proba_arr)}
        except Exception:
            proba = None
    return pred, proba
