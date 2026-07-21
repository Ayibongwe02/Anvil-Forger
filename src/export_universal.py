"""
Universal export for Anvil.

The headline feature: take a trained scikit-learn pipeline (preprocessing +
model) and serialize it into a single JSON weights file, paired with a
generic ~120-line inference engine (one version in pure Python, one in
pure JavaScript) that has ZERO dependencies — no scikit-learn, no numpy,
no ONNX runtime. Drop model.json + infer.py (or infer.js) into any project
and call predict(row_dict).

Supported model families (see constants.UNIVERSAL_EXPORT_SUPPORTED):
  - Linear/logistic regression -> raw coefficients, closed-form.
  - Random forest -> serialized tree structure (feature/threshold/children/
    leaf-values), averaged across trees at inference time.

Preprocessing (numeric impute+scale, categorical impute+one-hot) is baked
into the same JSON so the exported model is a true drop-in: raw feature
dict in, prediction out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from src.constants import UNIVERSAL_EXPORT_SUPPORTED


class UnsupportedExportError(Exception):
    pass


# ---------------------------------------------------------------------------
# Serialization: fitted sklearn pipeline -> JSON-able dict
# ---------------------------------------------------------------------------
def serialize_preprocessing(prep) -> list:
    """Walks a fitted ColumnTransformer (num pipeline + cat pipeline) and
    returns an ordered list of column specs matching the transformer's
    output feature order."""
    columns = []
    for name, transformer, cols in prep.transformers_:
        if name == "num" and cols:
            imputer = transformer.named_steps["impute"]
            scaler = transformer.named_steps["scale"]
            for i, col in enumerate(cols):
                columns.append({
                    "name": col,
                    "type": "numeric",
                    "impute_value": float(imputer.statistics_[i]),
                    "mean": float(scaler.mean_[i]),
                    "scale": float(scaler.scale_[i]) if scaler.scale_[i] != 0 else 1.0,
                })
        elif name == "cat" and cols:
            imputer = transformer.named_steps["impute"]
            onehot = transformer.named_steps["onehot"]
            for i, col in enumerate(cols):
                categories = onehot.categories_[i].tolist()
                columns.append({
                    "name": col,
                    "type": "categorical",
                    "impute_value": str(imputer.statistics_[i]),
                    "categories": [str(c) for c in categories],
                })
    return columns


def serialize_linear_model(model, algorithm: str, class_names: list) -> dict:
    coef = np.atleast_2d(model.coef_)
    intercept = np.atleast_1d(model.intercept_)
    return {
        "kind": "linear",
        "coef": coef.tolist(),          # shape (n_outputs, n_features)
        "intercept": intercept.tolist(),  # shape (n_outputs,)
        "classes": [str(c) for c in getattr(model, "classes_", [])] or class_names,
        "binary": coef.shape[0] == 1,
    }


def _serialize_tree(tree) -> list:
    """Flattens a single sklearn DecisionTree's internal tree_ structure
    into a list of plain-dict nodes indexed by position."""
    nodes = []
    n_nodes = tree.node_count
    for i in range(n_nodes):
        is_leaf = tree.children_left[i] == tree.children_right[i] == -1
        if is_leaf:
            value = tree.value[i][0]
            total = value.sum()
            leaf_value = (value / total).tolist() if total > 0 else value.tolist()
            nodes.append({"leaf": True, "value": leaf_value})
        else:
            nodes.append({
                "leaf": False,
                "feature": int(tree.feature[i]),
                "threshold": float(tree.threshold[i]),
                "left": int(tree.children_left[i]),
                "right": int(tree.children_right[i]),
            })
    return nodes


def serialize_forest_model(model, algorithm: str, task_type: str, class_names: list) -> dict:
    trees = [_serialize_tree(est.tree_) for est in model.estimators_]
    return {
        "kind": "forest",
        "task_type": task_type,
        "trees": trees,
        "classes": [str(c) for c in getattr(model, "classes_", [])] or class_names,
    }


def serialize_model(pipeline: Pipeline, algorithm: str, task_type: str, class_names: list) -> dict:
    if algorithm not in UNIVERSAL_EXPORT_SUPPORTED:
        raise UnsupportedExportError(
            f"Universal export isn't available for '{algorithm}' yet — download the "
            f"pickle bundle or use the hosted API endpoint for this model instead."
        )
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    columns = serialize_preprocessing(prep)

    if algorithm in ("logistic_regression", "linear_regression"):
        model_spec = serialize_linear_model(model, algorithm, class_names)
    else:  # random_forest
        model_spec = serialize_forest_model(model, algorithm, task_type, class_names)

    return {
        "anvil_export_version": 1,
        "algorithm": algorithm,
        "task_type": task_type,
        "class_names": class_names,
        "columns": columns,
        "model": model_spec,
    }


# ---------------------------------------------------------------------------
# Generic inference engines (written once, work for ANY exported model)
# ---------------------------------------------------------------------------
PYTHON_ENGINE = '''\
"""
Anvil universal inference engine — zero dependencies (stdlib only).
Usage:
    import json, infer
    model = json.load(open("model.json"))
    prediction = infer.predict(model, {"feature_a": 1.2, "feature_b": "red"})
"""

import math
import struct


def _to_f32(x):
    """Round-trip through float32, matching scikit-learn's internal tree
    representation (trees are fit against float32-cast features, so
    thresholds are only meaningful at float32 precision)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def _encode_row(model, row):
    vec = []
    for col in model["columns"]:
        raw = row.get(col["name"])
        if col["type"] == "numeric":
            val = col["impute_value"] if raw is None or raw == "" else float(raw)
            vec.append((val - col["mean"]) / col["scale"])
        else:
            val = col["impute_value"] if raw is None or raw == "" else str(raw)
            for cat in col["categories"]:
                vec.append(1.0 if val == cat else 0.0)
    return vec


def _predict_linear(spec, vec):
    outputs = []
    for coefs, intercept in zip(spec["coef"], spec["intercept"]):
        outputs.append(sum(c * v for c, v in zip(coefs, vec)) + intercept)
    return outputs


def _softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _tree_predict(nodes, vec):
    i = 0
    while not nodes[i]["leaf"]:
        node = nodes[i]
        i = node["left"] if _to_f32(vec[node["feature"]]) <= node["threshold"] else node["right"]
    return nodes[i]["value"]


def _predict_forest(spec, vec):
    n_trees = len(spec["trees"])
    first = _tree_predict(spec["trees"][0], vec)
    if spec["task_type"] == "regression":
        total = sum(_tree_predict(t, vec)[0] for t in spec["trees"])
        return total / n_trees
    else:
        n_classes = len(first)
        acc = [0.0] * n_classes
        for t in spec["trees"]:
            leaf = _tree_predict(t, vec)
            for k in range(n_classes):
                acc[k] += leaf[k]
        return [a / n_trees for a in acc]


def predict(model, row):
    """Returns dict with 'prediction' and, for classifiers, 'probabilities'."""
    vec = _encode_row(model, row)
    spec = model["model"]

    if spec["kind"] == "linear":
        raw = _predict_linear(spec, vec)
        if model["task_type"] == "regression":
            return {"prediction": raw[0]}
        if spec["binary"]:
            p1 = _sigmoid(raw[0])
            probs = {spec["classes"][0]: 1 - p1, spec["classes"][1]: p1} \\
                if len(spec["classes"]) == 2 else {"class_1": p1, "class_0": 1 - p1}
            pred_class = max(probs, key=probs.get)
            return {"prediction": pred_class, "probabilities": probs}
        else:
            probs_list = _softmax(raw)
            probs = {c: p for c, p in zip(spec["classes"], probs_list)}
            pred_class = max(probs, key=probs.get)
            return {"prediction": pred_class, "probabilities": probs}

    else:  # forest
        raw = _predict_forest(spec, vec)
        if model["task_type"] == "regression":
            return {"prediction": raw}
        probs = {c: p for c, p in zip(spec["classes"], raw)}
        pred_class = max(probs, key=probs.get)
        return {"prediction": pred_class, "probabilities": probs}
'''

JS_ENGINE = '''\
/**
 * Anvil universal inference engine — zero dependencies.
 * Usage:
 *   const model = require("./model.json");
 *   const { predict } = require("./infer.js");
 *   const result = predict(model, { feature_a: 1.2, feature_b: "red" });
 */

function encodeRow(model, row) {
  const vec = [];
  for (const col of model.columns) {
    const raw = row[col.name];
    if (col.type === "numeric") {
      const val = (raw === undefined || raw === null || raw === "") ? col.impute_value : Number(raw);
      vec.push((val - col.mean) / col.scale);
    } else {
      const val = (raw === undefined || raw === null || raw === "") ? col.impute_value : String(raw);
      for (const cat of col.categories) {
        vec.push(val === cat ? 1.0 : 0.0);
      }
    }
  }
  return vec;
}

function predictLinear(spec, vec) {
  return spec.coef.map((coefs, idx) => {
    const intercept = spec.intercept[idx];
    return coefs.reduce((sum, c, i) => sum + c * vec[i], intercept);
  });
}

function softmax(xs) {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function treePredict(nodes, vec) {
  let i = 0;
  while (!nodes[i].leaf) {
    const node = nodes[i];
    // scikit-learn fits tree thresholds against float32-cast features;
    // Math.fround replicates that precision so boundary cases match exactly.
    i = Math.fround(vec[node.feature]) <= node.threshold ? node.left : node.right;
  }
  return nodes[i].value;
}

function predictForest(spec, vec) {
  const nTrees = spec.trees.length;
  const first = treePredict(spec.trees[0], vec);
  if (spec.task_type === "regression") {
    const total = spec.trees.reduce((sum, t) => sum + treePredict(t, vec)[0], 0);
    return total / nTrees;
  } else {
    const nClasses = first.length;
    const acc = new Array(nClasses).fill(0);
    for (const t of spec.trees) {
      const leaf = treePredict(t, vec);
      for (let k = 0; k < nClasses; k++) acc[k] += leaf[k];
    }
    return acc.map((a) => a / nTrees);
  }
}

function argmaxEntry(obj) {
  let bestKey = null, bestVal = -Infinity;
  for (const [k, v] of Object.entries(obj)) {
    if (v > bestVal) { bestVal = v; bestKey = k; }
  }
  return bestKey;
}

function predict(model, row) {
  const vec = encodeRow(model, row);
  const spec = model.model;

  if (spec.kind === "linear") {
    const raw = predictLinear(spec, vec);
    if (model.task_type === "regression") {
      return { prediction: raw[0] };
    }
    if (spec.binary) {
      const p1 = sigmoid(raw[0]);
      const probs = spec.classes.length === 2
        ? { [spec.classes[0]]: 1 - p1, [spec.classes[1]]: p1 }
        : { class_0: 1 - p1, class_1: p1 };
      return { prediction: argmaxEntry(probs), probabilities: probs };
    } else {
      const probsList = softmax(raw);
      const probs = {};
      spec.classes.forEach((c, i) => (probs[c] = probsList[i]));
      return { prediction: argmaxEntry(probs), probabilities: probs };
    }
  } else {
    const raw = predictForest(spec, vec);
    if (model.task_type === "regression") {
      return { prediction: raw };
    }
    const probs = {};
    spec.classes.forEach((c, i) => (probs[c] = raw[i]));
    return { prediction: argmaxEntry(probs), probabilities: probs };
  }
}

module.exports = { predict };
'''


def write_universal_bundle(pipeline: Pipeline, algorithm: str, task_type: str,
                            class_names: list, model_name: str, output_dir: Path) -> list:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = serialize_model(pipeline, algorithm, task_type, class_names)
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(spec, indent=2))

    py_path = output_dir / "infer.py"
    py_path.write_text(PYTHON_ENGINE)

    js_path = output_dir / "infer.js"
    js_path.write_text(JS_ENGINE)

    readme_path = output_dir / "README.md"
    example_row = {c["name"]: (0 if c["type"] == "numeric" else c["categories"][0])
                   for c in spec["columns"]}
    readme_path.write_text(f"""# {model_name} — Universal Export

Zero-dependency model export from Anvil. No scikit-learn, no numpy, no ONNX
runtime required on the receiving end — just `model.json` + one inference
file.

Algorithm: **{spec['algorithm']}** · Task: **{spec['task_type']}**

## Python

```python
import json
from infer import predict

model = json.load(open("model.json"))
result = predict(model, {json.dumps(example_row)})
print(result)
```

## JavaScript / Node

```javascript
const model = require("./model.json");
const {{ predict }} = require("./infer.js");

const result = predict(model, {json.dumps(example_row)});
console.log(result);
```

## Input format

Pass a plain object/dict keyed by the original column names:
{json.dumps([c["name"] for c in spec["columns"]], indent=2)}

Missing or empty values are automatically imputed using values learned at
training time (median for numeric columns, most-frequent for categorical).
""")

    return [model_path, py_path, js_path, readme_path]
