"""Shared constants for Anvil."""

TASK_KINDS = {
    "tabular": "Tabular data (CSV)",
    "image":   "Image classification",
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

IMAGE_ALGORITHMS = [
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "svm",
    "knn",
]

ALGORITHM_LABELS = {
    "logistic_regression": "Logistic Regression",
    "linear_regression":   "Linear Regression",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "extra_trees":         "Extra Trees",
    "svm":                 "Support Vector Machine",
    "knn":                 "K-Nearest Neighbors",
}

UNIVERSAL_EXPORT_SUPPORTED = {
    "logistic_regression", "linear_regression", "random_forest",
}

MAX_CSV_ROWS_PREVIEW = 10
IMAGE_SIZE = (64, 64)

IMPORT_TASK_TYPES = {"classification": "Classification", "regression": "Regression"}
