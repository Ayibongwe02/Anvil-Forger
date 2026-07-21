"""
Image classification engine for Anvil.

No deep learning framework is available in this environment, so images are
classified using classical computer-vision features — HOG (shape/edge
structure) concatenated with a color histogram (RGB) — feeding a
scikit-learn classifier. This is a legitimate, fast, CPU-only approach for
small internal-tool use cases (visual QC, tagging, simple sorting) and is
honestly disclosed as non-deep-learning in the UI: it will not match a CNN
on complex natural images, but it's real, it works, and it exports cleanly.

Expected input: a zip file where each top-level folder name is a class
label and contains images of that class, e.g.:
    dataset.zip
      good/   img1.jpg img2.jpg ...
      defective/ img1.jpg img2.jpg ...
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import IMAGE_SIZE, ALGORITHM_LABELS

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TrainingError(Exception):
    pass


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """scikit-learn-compatible transformer: PIL-loadable image bytes ->
    fixed-length feature vector (HOG + color histogram). Stateless, so it
    round-trips cleanly through pickle/joblib for export."""

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._features_from_bytes(b) for b in X])

    def _features_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(self.image_size)
        arr = np.asarray(img) / 255.0

        gray = rgb2gray(arr)
        hog_features = hog(
            gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            feature_vector=True,
        )

        color_hist = np.concatenate([
            np.histogram(arr[:, :, c], bins=16, range=(0, 1))[0] for c in range(3)
        ]).astype(float)
        color_hist = color_hist / (color_hist.sum() + 1e-8)

        return np.concatenate([hog_features, color_hist])


def load_labeled_images_from_zip(zip_path: str) -> tuple[list[bytes], list[str]]:
    images, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if len(path.parts) < 2:
                continue  # must be inside a class-named folder
            label = path.parts[0]
            if label.startswith("__MACOSX") or path.name.startswith("."):
                continue
            with zf.open(info) as f:
                images.append(f.read())
                labels.append(label)
    return images, labels


def _candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(),
    }


def train(images: list[bytes], labels: list[str], algorithms: list[str] = None) -> dict:
    if len(images) < 10:
        raise TrainingError("Need at least 10 labeled images to train.")
    class_names = sorted(set(labels))
    if len(class_names) < 2:
        raise TrainingError("Need at least 2 distinct class folders to classify.")

    X = np.array(images, dtype=object)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = _candidate_models()
    if algorithms:
        candidates = {k: v for k, v in candidates.items() if k in algorithms}

    leaderboard = []
    fitted_pipelines = {}
    for algo_name, estimator in candidates.items():
        try:
            pipe = Pipeline([
                ("features", ImageFeatureExtractor()),
                ("scale", StandardScaler()),
                ("model", estimator),
            ])
            cv = min(5, max(2, len(X_train) // 10))
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            pipe.fit(X_train, y_train)
            fitted_pipelines[algo_name] = pipe
            leaderboard.append({
                "algorithm": algo_name,
                "label": ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": float(np.mean(scores)),
                "cv_score_std": float(np.std(scores)),
            })
        except Exception as e:
            leaderboard.append({
                "algorithm": algo_name,
                "label": ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": None,
                "cv_score_std": None,
                "error": str(e)[:200],
            })

    scored = [r for r in leaderboard if r.get("cv_score_mean") is not None]
    if not scored:
        raise TrainingError("All candidate algorithms failed on this image set.")

    scored.sort(key=lambda r: r["cv_score_mean"], reverse=True)
    leaderboard.sort(key=lambda r: (r.get("cv_score_mean") is None, -(r.get("cv_score_mean") or -999)))
    winner_name = scored[0]["algorithm"]
    winner_pipeline = fitted_pipelines[winner_name]

    y_pred = winner_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
    }

    return {
        "task_type": "classification",
        "algorithm": winner_name,
        "algorithm_label": ALGORITHM_LABELS.get(winner_name, winner_name),
        "class_names": class_names,
        "feature_columns": ["image"],
        "metrics": metrics,
        "leaderboard": leaderboard,
        "feature_importance": [],
        "pipeline": winner_pipeline,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }


def predict_single(pipeline: Pipeline, image_bytes: bytes):
    pred = pipeline.predict([image_bytes])[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba_arr = pipeline.predict_proba([image_bytes])[0]
            classes = pipeline.named_steps["model"].classes_
            proba = {str(c): float(p) for c, p in zip(classes, proba_arr)}
        except Exception:
            proba = None
    return pred, proba
