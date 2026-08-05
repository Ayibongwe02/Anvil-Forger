"""
Hosted prediction API for Anvil — the "hosted API endpoint" export path.

Every trained model gets a live prediction endpoint for free, gated by a
per-team API key:

    POST /api/v1/predict/<model_id>
    Header: X-API-Key: <key>
    Body (tabular): {"feature_a": 1.2, "feature_b": "red"}
    Body (image):   multipart/form-data, field name "image"

Every call is logged to prediction_log for a lightweight audit trail,
visible on the model's detail page.
"""

from datetime import datetime, timezone

import joblib
from flask import Blueprint, request, jsonify

from src import db
from src import tabular_training, image_training, onnx_import

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

_pipeline_cache = {}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_pipeline(model_row):
    path = model_row["artifact_path"]
    if path not in _pipeline_cache:
        if model_row["runtime"] == "onnx":
            _pipeline_cache[path] = onnx_import.load_session(path)
        else:
            _pipeline_cache[path] = joblib.load(path)
    return _pipeline_cache[path]


def _authenticate_request(team_id_expected):
    key_value = request.headers.get("X-API-Key")
    if not key_value:
        return None, ({"error": "Missing X-API-Key header."}, 401)
    api_key = db.get_api_key(key_value)
    if not api_key:
        return None, ({"error": "Invalid or revoked API key."}, 401)
    if api_key["team_id"] != team_id_expected:
        return None, ({"error": "API key does not have access to this model."}, 403)
    return api_key, None


@api_bp.route("/predict/<int:model_id>", methods=["POST"])
def predict(model_id):
    model_row = db.get_model(model_id)
    if not model_row:
        return jsonify({"error": "Model not found."}), 404
    if model_row["status"] != "ready":
        return jsonify({"error": f"Model status is '{model_row['status']}', not ready to serve."}), 409

    with db.get_conn() as conn:
        project_row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (model_row["project_id"],)
        ).fetchone()
    team_id = project_row["team_id"]

    api_key, err = _authenticate_request(team_id)
    if err:
        body, status = err
        return jsonify(body), status

    pipeline = _load_pipeline(model_row)
    project_task_kind = project_row["task_kind"]

    try:
        if model_row["runtime"] == "onnx":
            payload = request.get_json(force=True, silent=True) or {}
            import json as _json
            feature_columns = _json.loads(model_row["feature_columns_json"])
            class_names = _json.loads(model_row["class_names_json"])
            pred, proba = onnx_import.predict_single(
                pipeline, feature_columns, class_names, model_row["task_type"], payload
            )
            input_summary = payload
        elif project_task_kind == "image":
            if "image" not in request.files:
                return jsonify({"error": "Send the image as multipart/form-data field 'image'."}), 400
            image_bytes = request.files["image"].read()
            pred, proba = image_training.predict_single(pipeline, image_bytes)
            input_summary = {"image": request.files["image"].filename}
        else:
            payload = request.get_json(force=True, silent=True) or {}
            import json as _json
            feature_columns = _json.loads(model_row["feature_columns_json"])
            pred, proba = tabular_training.predict_single(pipeline, feature_columns, payload)
            input_summary = payload

        result = {"prediction": pred if not hasattr(pred, "item") else pred.item()}
        if proba is not None:
            result["probabilities"] = proba

        db.log_prediction(model_id, api_key["id"], input_summary, result, _now())
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)[:300]}"}), 400


@api_bp.route("/models/<int:model_id>/schema", methods=["GET"])
def schema(model_id):
    """Lets a consuming app introspect what fields to send, without an API key
    (metadata only, no data exposure)."""
    model_row = db.get_model(model_id)
    if not model_row:
        return jsonify({"error": "Model not found."}), 404
    import json as _json
    return jsonify({
        "model_id": model_row["id"],
        "name": model_row["name"],
        "task_type": model_row["task_type"],
        "algorithm": model_row["algorithm"],
        "feature_columns": _json.loads(model_row["feature_columns_json"]),
        "class_names": _json.loads(model_row["class_names_json"]),
        "endpoint": f"/api/v1/predict/{model_row['id']}",
    })
