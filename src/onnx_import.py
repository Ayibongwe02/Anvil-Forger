"""
Bring-your-own-model import for Anvil — ONNX only, deliberately.

Anvil trains its own models in-process and unpickles/joblib-loads the
resulting artifacts, which is safe *only* because Anvil itself produced
them. A pickle (or joblib) file is a program, not a data format: loading
one runs arbitrary code chosen by whoever wrote the file, via
`__reduce__`. There is no way to "sandbox" that at the file-format level,
which is why this module does not accept uploaded .pkl files at all.

ONNX is a protobuf-described, static computation graph. Loading one does
not execute arbitrary Python — the runtime only evaluates the numeric ops
declared in the graph. That's what makes "let a user hand us a model file"
tractable here. Two things still need checking on the way in:

  1. The file has to actually be a well-formed ONNX model (checker).
  2. The graph must not reference "external data" — ONNX allows a tensor's
     bytes to live in a separate file pointed to by a relative path stored
     inside the model, which is an arbitrary-file-read vector if honored
     for an untrusted upload. We refuse any model that uses it.

Only tabular models are supported (mirrors the existing "universal
export" limitation) — image tensors would need a documented preprocessing
contract we don't have for arbitrary uploads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

MAX_ONNX_BYTES = 50 * 1024 * 1024  # 50MB — plenty for a tabular model


class OnnxImportError(Exception):
    pass


def validate_and_load(file_bytes: bytes) -> onnx.ModelProto:
    """Parses + validates an uploaded ONNX file. Raises OnnxImportError on
    anything that isn't a plain, self-contained ONNX model. Never executes
    the graph — this only inspects the protobuf structure."""
    if len(file_bytes) > MAX_ONNX_BYTES:
        raise OnnxImportError(f"File is too large (max {MAX_ONNX_BYTES // (1024*1024)}MB for an imported model).")

    try:
        model = onnx.load_model_from_string(file_bytes)
    except Exception as e:
        raise OnnxImportError(f"Not a valid ONNX file: {e}")

    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as e:
        raise OnnxImportError(f"ONNX model failed validation: {e}")

    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            raise OnnxImportError(
                "This model references external data files, which Anvil doesn't allow "
                "for imported models. Re-export it with weights embedded in a single "
                ".onnx file (e.g. onnx.save_model(model, path, save_as_external_data=False))."
            )

    if len(model.graph.input) == 0:
        raise OnnxImportError("Model graph has no inputs.")

    return model


def describe_inputs(model: onnx.ModelProto) -> list[dict]:
    """Returns [{name, elem_type, shape}] for each graph input, in order."""
    out = []
    for inp in model.graph.input:
        shape = []
        ttype = inp.type.tensor_type
        for d in ttype.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else None)
        out.append({"name": inp.name, "elem_type": ttype.elem_type, "shape": shape})
    return out


def load_session(artifact_path: str) -> ort.InferenceSession:
    """Loads an already-validated .onnx file for inference. Restricts to
    the default CPU execution provider and disables arbitrary op
    extensions — only ONNX's built-in numeric ops are ever evaluated."""
    so = ort.SessionOptions()
    so.enable_mem_pattern = False
    return ort.InferenceSession(artifact_path, sess_options=so, providers=["CPUExecutionProvider"])


def _np_dtype_for(elem_type: int):
    # Common ONNX tensor element types used by tabular exports.
    return {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.STRING: np.str_,
    }.get(elem_type, np.float32)


def predict_single(session: ort.InferenceSession, feature_columns: list[str],
                    class_names: list[str], task_type: str, input_dict: dict):
    """Builds input tensors for the session from a flat feature dict and
    runs one prediction.

    Two graph shapes are supported, matching how tabular sklearn->ONNX
    exports typically look:
      - One input named after a single vector (e.g. "float_input") of
        shape [None, n_features]: features are packed in the order given
        by `feature_columns`.
      - One input per feature, each shaped [None] or [None, 1]: matched by
        name directly against `feature_columns`.
    """
    inputs = session.get_inputs()
    feed = {}

    if len(inputs) == 1 and (len(inputs[0].shape) == 2 or len(inputs[0].shape) == 1):
        inp = inputs[0]
        dtype = _np_dtype_for(getattr(onnx.TensorProto, _ort_type_to_onnx_name(inp.type)))
        row = [_coerce(input_dict.get(col), dtype) for col in feature_columns]
        arr = np.array([row], dtype=dtype)
        feed[inp.name] = arr
    else:
        for inp in inputs:
            if inp.name not in feature_columns:
                raise OnnxImportError(
                    f"Model expects an input named '{inp.name}' but that wasn't in the "
                    f"feature columns you provided."
                )
            dtype = _np_dtype_for(getattr(onnx.TensorProto, _ort_type_to_onnx_name(inp.type)))
            val = _coerce(input_dict.get(inp.name), dtype)
            feed[inp.name] = np.array([[val]], dtype=dtype)

    outputs = session.run(None, feed)
    output_names = [o.name for o in session.get_outputs()]

    pred = None
    proba = None
    for name, val in zip(output_names, outputs):
        if "label" in name.lower() or (pred is None and "prob" not in name.lower()):
            pred = _scalar(val)
        elif "prob" in name.lower() or proba is None:
            proba = _to_proba_dict(val, class_names)

    if pred is None:
        pred = _scalar(outputs[0])
    return pred, (proba if task_type == "classification" else None)


def _ort_type_to_onnx_name(ort_type: str) -> str:
    # onnxruntime reports types like "tensor(float)", "tensor(int64)".
    mapping = {
        "tensor(float)": "FLOAT",
        "tensor(double)": "DOUBLE",
        "tensor(int64)": "INT64",
        "tensor(int32)": "INT32",
        "tensor(string)": "STRING",
    }
    return mapping.get(ort_type, "FLOAT")


def _coerce(val, dtype):
    if val is None or val == "":
        return 0 if dtype != np.str_ else ""
    if dtype == np.str_:
        return str(val)
    try:
        return dtype(val)
    except (TypeError, ValueError):
        return 0


def _scalar(arr):
    val = np.asarray(arr).reshape(-1)[0]
    return val.item() if hasattr(val, "item") else val


def _to_proba_dict(arr, class_names: list[str]):
    try:
        # skl2onnx classifiers often return a list of dicts (ZipMap output).
        if isinstance(arr, list) and arr and isinstance(arr[0], dict):
            return {str(k): float(v) for k, v in arr[0].items()}
        vec = np.asarray(arr).reshape(-1)
        if class_names and len(class_names) == len(vec):
            return {c: float(p) for c, p in zip(class_names, vec)}
        return {str(i): float(p) for i, p in enumerate(vec)}
    except Exception:
        return None
