"""Shared constants for Anvil."""

TASK_KINDS = {
    "tabular": "Tabular data (CSV)",
    "image": "Image classification",
}

TABULAR_ALGORITHMS_CLASSIFICATION = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm",
    "knn",
]

TABULAR_ALGORITHMS_REGRESSION = [
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "svm",
    "knn",
]

ALGORITHM_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbors",
}

# Algorithms with a fully implemented dependency-free ("universal") export:
# linear models export as raw coefficients, random forest exports as a
# serialized tree structure (feature/threshold/children/leaf-values), with
# generated pure-Python and pure-JS inference code that needs no ML library
# at all on the receiving end. Gradient boosting, SVM, and KNN are excellent
# models but don't have as clean a closed-form export, so universal export
# is disabled for them — pickle bundle and hosted API export still work.
UNIVERSAL_EXPORT_SUPPORTED = {
    "logistic_regression", "linear_regression", "random_forest",
}

MAX_CSV_ROWS_PREVIEW = 10
IMAGE_SIZE = (64, 64)  # normalized size for classical feature extraction

# "Bring your own model" import is ONNX-only and tabular-only — see
# src/onnx_import.py for why pickle uploads aren't accepted.
IMPORT_TASK_TYPES = {"classification": "Classification", "regression": "Regression"}
