"""
Tabular AutoML engine for Anvil.

Given a CSV and a target column, this:
  1. Infers task type (classification vs regression) from the target.
  2. Builds a preprocessing pipeline (impute + scale numeric, impute +
     one-hot categorical) shared by every candidate algorithm.
  3. Cross-validates a shortlist of scikit-learn algorithms.
  4. Refits the winner on the full training split, evaluates on a held-out
     test split, and returns a leaderboard + metrics + the fitted pipeline.

No Streamlit/Flask dependency — pure functions over pandas/sklearn so this
is unit-testable in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC, SVR

from src.constants import TABULAR_ALGORITHMS_CLASSIFICATION, TABULAR_ALGORITHMS_REGRESSION, \
    ALGORITHM_LABELS


class TrainingError(Exception):
    pass


def infer_task_type(y: pd.Series) -> str:
    """Classification if target is non-numeric or has few unique values
    relative to sample size; regression otherwise."""
    if y.dtype == object or str(y.dtype) == "category" or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    if n_unique <= max(20, int(0.05 * len(y))) and n_unique < len(y) * 0.5:
        # heuristic: small integer target likely a class label, e.g. 0/1, star ratings
        if pd.api.types.is_integer_dtype(y) or set(y.dropna().unique()) <= {0, 1}:
            return "classification"
    return "regression"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers)


def _candidate_models(task_type: str) -> dict:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "knn": KNeighborsClassifier(),
        }
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "svm": SVR(),
        "knn": KNeighborsRegressor(),
    }


def train(df: pd.DataFrame, target_column: str, algorithms: list[str] = None) -> dict:
    if target_column not in df.columns:
        raise TrainingError(f"Target column '{target_column}' not found in dataset.")

    df = df.dropna(subset=[target_column]).reset_index(drop=True)
    if len(df) < 20:
        raise TrainingError("Need at least 20 rows with a non-empty target to train reliably.")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    task_type = infer_task_type(y)

    class_names = []
    if task_type == "classification":
        y = y.astype(str)
        class_names = sorted(y.unique().tolist())
        if len(class_names) < 2:
            raise TrainingError("Target column needs at least 2 distinct classes to classify.")
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    candidates = _candidate_models(task_type)
    if algorithms:
        candidates = {k: v for k, v in candidates.items() if k in algorithms}

    scoring = "accuracy" if task_type == "classification" else "r2"
    leaderboard = []
    fitted_pipelines = {}

    for algo_name, estimator in candidates.items():
        try:
            preprocessor = _build_preprocessor(X_train)
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            cv = min(5, max(2, len(X_train) // 20))
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
            pipe.fit(X_train, y_train)
            fitted_pipelines[algo_name] = pipe
            leaderboard.append({
                "algorithm": algo_name,
                "label": ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": float(np.mean(scores)),
                "cv_score_std": float(np.std(scores)),
            })
        except Exception as e:  # keep going even if one algorithm fails on this data
            leaderboard.append({
                "algorithm": algo_name,
                "label": ALGORITHM_LABELS.get(algo_name, algo_name),
                "cv_score_mean": None,
                "cv_score_std": None,
                "error": str(e)[:200],
            })

    scored = [r for r in leaderboard if r.get("cv_score_mean") is not None]
    if not scored:
        raise TrainingError("All candidate algorithms failed on this dataset. Check column types.")

    scored.sort(key=lambda r: r["cv_score_mean"], reverse=True)
    leaderboard.sort(key=lambda r: (r.get("cv_score_mean") is None, -(r.get("cv_score_mean") or -999)))
    winner_name = scored[0]["algorithm"]
    winner_pipeline = fitted_pipelines[winner_name]

    y_pred = winner_pipeline.predict(X_test)
    metrics = _compute_metrics(task_type, y_test, y_pred, class_names)

    feature_importance = _extract_feature_importance(winner_pipeline, X_train.columns.tolist())

    return {
        "task_type": task_type,
        "algorithm": winner_name,
        "algorithm_label": ALGORITHM_LABELS.get(winner_name, winner_name),
        "class_names": class_names,
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "pipeline": winner_pipeline,
        "test_size": len(X_test),
        "train_size": len(X_train),
    }


def _compute_metrics(task_type: str, y_test, y_pred, class_names: list) -> dict:
    if task_type == "classification":
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": class_names,
        }
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def _extract_feature_importance(pipeline: Pipeline, original_columns: list) -> list:
    """Best-effort feature importance, mapped back through one-hot expansion
    where possible. Returns [] if the model type doesn't expose one."""
    model = pipeline.named_steps["model"]
    prep = pipeline.named_steps["prep"]
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(original_columns))]

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
    return [{"feature": str(n), "importance": float(v)} for n, v in pairs[:15]]


def predict_single(pipeline: Pipeline, feature_columns: list, input_dict: dict):
    row = {col: input_dict.get(col) for col in feature_columns}
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba_arr = pipeline.predict_proba(X)[0]
            classes = pipeline.named_steps["model"].classes_
            proba = {str(c): float(p) for c, p in zip(classes, proba_arr)}
        except Exception:
            proba = None
    return pred, proba
