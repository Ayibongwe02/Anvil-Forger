"""
Export bundling for Anvil.

Wraps the two "portable file" export paths into single downloadable zips:
  - Pickle bundle: joblib-serialized pipeline + a tiny load_and_predict.py
    + requirements.txt, for teams staying inside the Python/scikit-learn
    ecosystem.
  - Universal bundle: model.json + infer.py + infer.js + README, zero
    dependencies (see src.export_universal).

The third export path — the hosted API — doesn't produce a file at all;
it's served live by src.api_serving from the artifact already on disk.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from src import export_universal


def _typed_example_row(pipeline, feature_columns: list) -> dict:
    """Best-effort realistic example row: numeric columns get their training
    median, categorical columns get their most-frequent category. Falls back
    to a placeholder string if the pipeline shape is unexpected (e.g. image
    models, which don't have a ColumnTransformer to introspect)."""
    if pipeline is None or "prep" not in getattr(pipeline, "named_steps", {}):
        return {col: "REPLACE_ME" for col in feature_columns}
    try:
        columns = export_universal.serialize_preprocessing(pipeline.named_steps["prep"])
        by_name = {c["name"]: c for c in columns}
        row = {}
        for col in feature_columns:
            spec = by_name.get(col)
            if spec is None:
                row[col] = "REPLACE_ME"
            elif spec["type"] == "numeric":
                row[col] = round(spec["impute_value"], 2)
            else:
                row[col] = spec["impute_value"]
        return row
    except Exception:
        return {col: "REPLACE_ME" for col in feature_columns}


def build_pickle_bundle_zip(artifact_path: str, model_name: str, task_type: str,
                             algorithm: str, feature_columns: list, class_names: list,
                             pipeline=None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(artifact_path, arcname="model.pkl")

        example_row = _typed_example_row(pipeline, feature_columns)
        load_script = f'''"""
{model_name} — pickle bundle export from Anvil.
Algorithm: {algorithm} · Task: {task_type}

Requires: scikit-learn, pandas, joblib (see requirements.txt — pin to the
same major versions you trained with to avoid pickle incompatibilities).
"""

import joblib
import pandas as pd

pipeline = joblib.load("model.pkl")

# Example feature row — replace these values with real ones. Types shown
# reflect what each column looked like at training time.
example_row = {json.dumps(example_row, indent=4)}

def predict(row: dict):
    X = pd.DataFrame([row])
    pred = pipeline.predict(X)[0]
    result = {{"prediction": pred}}
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        classes = pipeline.named_steps["model"].classes_
        result["probabilities"] = {{str(c): float(p) for c, p in zip(classes, proba)}}
    return result

if __name__ == "__main__":
    print(predict(example_row))
'''
        zf.writestr("load_and_predict.py", load_script)
        zf.writestr("requirements.txt", "scikit-learn>=1.5\npandas>=2.0\njoblib>=1.3\n")
        zf.writestr("README.md", f"""# {model_name} — Pickle Bundle

Algorithm: **{algorithm}** · Task: **{task_type}** · Class names: {class_names or "N/A (regression)"}

1. `pip install -r requirements.txt`
2. `python load_and_predict.py` to see an example prediction
3. Import `predict()` from `load_and_predict.py` into your own app, or load
   `model.pkl` directly with `joblib.load()`.

This bundle requires scikit-learn on the receiving end. For a
dependency-free option, use the "Universal Export" from the model page
instead.
""")
    return buf.getvalue()


def build_universal_zip(pipeline, algorithm: str, task_type: str, class_names: list,
                         model_name: str, tmp_dir: Path) -> bytes:
    files = export_universal.write_universal_bundle(
        pipeline, algorithm, task_type, class_names, model_name, tmp_dir
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return buf.getvalue()
