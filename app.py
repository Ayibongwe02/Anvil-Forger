"""
Anvil — train ML models as a team, export them anywhere.
Main Flask entry point.
"""

import io
import json
import secrets
import tempfile
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, \
    send_file, session, jsonify
from werkzeug.utils import secure_filename

from src import db, auth
from src.auth import login_required
from src.constants import TASK_KINDS, TABULAR_ALGORITHMS_CLASSIFICATION, \
    TABULAR_ALGORITHMS_REGRESSION, ALGORITHM_LABELS, UNIVERSAL_EXPORT_SUPPORTED, \
    IMPORT_TASK_TYPES
from src import tabular_training, image_training, export_bundle, onnx_import
from src.api_serving import api_bp

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
import os
app.secret_key = os.environ.get("ANVIL_SECRET_KEY", "anvil-dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB uploads (image zips)
app.register_blueprint(api_bp)

with app.app_context():
    db.init_db()


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


@app.template_filter("from_json")
def from_json_filter(value):
    return json.loads(value) if value else {}


@app.context_processor
def inject_globals():
    return {"current_user": auth.current_user(), "current_team": auth.current_team()}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup/team", methods=["GET", "POST"])
def signup_team():
    if request.method == "POST":
        team_name = request.form["team_name"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        display_name = request.form["display_name"].strip()
        if not all([team_name, email, password, display_name]):
            flash("All fields are required.", "error")
            return render_template("signup_team.html")
        if db.get_user_by_email(email.lower()):
            flash("That email is already registered.", "error")
            return render_template("signup_team.html")
        team_id, user_id, invite_code = auth.create_team_with_admin(
            team_name, email, password, display_name
        )
        user = db.get_user(user_id)
        auth.login_user(user)
        flash(f"Team created! Your invite code is {invite_code} — share it with teammates.", "success")
        return redirect(url_for("dashboard"))
    return render_template("signup_team.html")


@app.route("/signup/join", methods=["GET", "POST"])
def signup_join():
    if request.method == "POST":
        invite_code = request.form["invite_code"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]
        display_name = request.form["display_name"].strip()
        user_id, error = auth.join_team_with_code(invite_code, email, password, display_name)
        if error:
            flash(error, "error")
            return render_template("signup_join.html")
        user = db.get_user(user_id)
        auth.login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("signup_join.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = auth.authenticate(request.form["email"], request.form["password"])
        if not user:
            flash("Incorrect email or password.", "error")
            return render_template("login.html")
        auth.login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    auth.logout_user()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboard / Projects
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    team = auth.current_team()
    if not team:
        auth.logout_user()
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("login"))
    projects = db.list_projects(team["id"])
    return render_template("dashboard.html", projects=projects, task_kinds=TASK_KINDS)


@app.route("/projects/new", methods=["GET", "POST"])
@login_required
def new_project():
    if request.method == "POST":
        team = auth.current_team()
        if not team:
            flash("Session expired. Please log in again.", "warning")
            return redirect(url_for("login"))
        user = auth.current_user()
        name = request.form["name"].strip()
        task_kind = request.form["task_kind"]
        description = request.form.get("description", "").strip()
        if not name or task_kind not in TASK_KINDS:
            flash("Please provide a project name and choose a task type.", "error")
            return render_template("new_project.html", task_kinds=TASK_KINDS)
        project_id = db.create_project(team["id"], name, task_kind, description, user["id"], _now())
        return redirect(url_for("project_detail", project_id=project_id))
    return render_template("new_project.html", task_kinds=TASK_KINDS)


@app.route("/projects/<int:project_id>")
@login_required
def project_detail(project_id):
    team = auth.current_team()
    if not team:
        flash("Session expired. Please log in again.", "warning")
        return redirect(url_for("login"))
    project = db.get_project(project_id, team["id"])
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("dashboard"))
    datasets = [d for d in db.list_datasets(project_id) if d["kind"] != "external"]
    models = db.list_models(project_id)
    algo_choices = (TABULAR_ALGORITHMS_CLASSIFICATION + TABULAR_ALGORITHMS_REGRESSION) \
        if project["task_kind"] == "tabular" else list(ALGORITHM_LABELS.keys())
    algo_choices = sorted(set(algo_choices))
    return render_template(
        "project_detail.html", project=project, datasets=datasets, models=models,
        algorithm_labels=ALGORITHM_LABELS, algo_choices=algo_choices,
        import_task_types=IMPORT_TASK_TYPES,
    )


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
@login_required
def delete_project(project_id):
    team = auth.current_team()
    project = db.get_project(project_id, team["id"])
    if project:
        db.delete_project(project_id)
        flash("Project deleted.", "success")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@app.route("/projects/<int:project_id>/datasets/upload", methods=["POST"])
@login_required
def upload_dataset(project_id):
    team = auth.current_team()
    user = auth.current_user()
    project = db.get_project(project_id, team["id"])
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("dashboard"))

    file = request.files.get("file")
    if not file or not file.filename:
        flash("Choose a file to upload.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = DATASETS_DIR / unique_name
    file.save(save_path)

    try:
        if project["task_kind"] == "tabular":
            df = pd.read_csv(save_path)
            meta = {
                "columns": df.columns.tolist(),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "row_count": len(df),
                "preview": df.head(5).to_dict(orient="records"),
            }
            kind = "csv"
        else:
            images, labels = image_training.load_labeled_images_from_zip(str(save_path))
            class_counts = pd.Series(labels).value_counts().to_dict() if labels else {}
            meta = {"class_counts": class_counts, "image_count": len(images)}
            kind = "image_folder_zip"
            if len(images) == 0:
                raise ValueError(
                    "No labeled images found. Zip must contain top-level folders "
                    "named after each class, with images inside."
                )
    except Exception as e:
        save_path.unlink(missing_ok=True)
        flash(f"Couldn't read that file: {e}", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    db.create_dataset(project_id, filename, str(save_path), kind, meta, user["id"], _now())
    flash("Dataset uploaded.", "success")
    return redirect(url_for("project_detail", project_id=project_id))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@app.route("/projects/<int:project_id>/train", methods=["POST"])
@login_required
def train_model(project_id):
    team = auth.current_team()
    user = auth.current_user()
    project = db.get_project(project_id, team["id"])
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("dashboard"))

    dataset_id = int(request.form["dataset_id"])
    model_name = request.form.get("model_name", "").strip() or f"Model {_now()[:19]}"
    selected_algorithms = request.form.getlist("algorithms") or None
    dataset = db.get_dataset(dataset_id)

    try:
        if project["task_kind"] == "tabular":
            target_column = request.form["target_column"]
            df = pd.read_csv(dataset["file_path"])
            result = tabular_training.train(df, target_column, selected_algorithms)
        else:
            target_column = ""
            images, labels = image_training.load_labeled_images_from_zip(dataset["file_path"])
            result = image_training.train(images, labels, selected_algorithms)

        artifact_path = MODELS_DIR / f"{uuid.uuid4().hex}.pkl"
        joblib.dump(result["pipeline"], artifact_path)

        db.create_model(
            project_id, dataset_id, model_name, result["task_type"], result["algorithm"],
            target_column, result["feature_columns"], result["class_names"],
            {**result["metrics"], "train_size": result["train_size"], "test_size": result["test_size"],
             "feature_importance": result["feature_importance"]},
            result["leaderboard"], str(artifact_path), "ready", user["id"], _now(),
        )
        flash(f"Trained '{model_name}' — winner: {result['algorithm_label']}.", "success")

    except (tabular_training.TrainingError, image_training.TrainingError) as e:
        flash(f"Training failed: {e}", "error")
    except Exception as e:
        flash(f"Training failed unexpectedly: {str(e)[:300]}", "error")

    return redirect(url_for("project_detail", project_id=project_id))


# ---------------------------------------------------------------------------
# Bring your own model (ONNX import)
# ---------------------------------------------------------------------------
@app.route("/projects/<int:project_id>/models/import", methods=["POST"])
@login_required
def import_model(project_id):
    team = auth.current_team()
    user = auth.current_user()
    project = db.get_project(project_id, team["id"])
    if not project:
        flash("Project not found.", "error")
        return redirect(url_for("dashboard"))
    if project["task_kind"] != "tabular":
        flash("Model import currently supports tabular projects only.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    file = request.files.get("onnx_file")
    if not file or not file.filename:
        flash("Choose an .onnx file to import.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    if not file.filename.lower().endswith(".onnx"):
        flash("Only .onnx files can be imported (not pickle/.pkl — see the note above).", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    model_name = request.form.get("model_name", "").strip() or f"Imported model {_now()[:19]}"
    task_type = request.form.get("task_type", "")
    if task_type not in IMPORT_TASK_TYPES:
        flash("Choose a task type for the imported model.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    feature_columns = [c.strip() for c in request.form.get("feature_columns", "").split(",") if c.strip()]
    class_names = [c.strip() for c in request.form.get("class_names", "").split(",") if c.strip()]
    if not feature_columns:
        flash("List the feature columns (comma-separated) your model expects, in order.", "error")
        return redirect(url_for("project_detail", project_id=project_id))
    if task_type == "classification" and not class_names:
        flash("List the class names (comma-separated), in the order your model outputs them.", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    file_bytes = file.read()
    try:
        model = onnx_import.validate_and_load(file_bytes)
    except onnx_import.OnnxImportError as e:
        flash(f"Couldn't import that model: {e}", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    artifact_path = MODELS_DIR / f"{uuid.uuid4().hex}.onnx"
    artifact_path.write_bytes(file_bytes)

    # Sanity-check inference works before we save the model row.
    try:
        session = onnx_import.load_session(str(artifact_path))
        sample = {col: 0 for col in feature_columns}
        onnx_import.predict_single(session, feature_columns, class_names, task_type, sample)
    except Exception as e:
        artifact_path.unlink(missing_ok=True)
        flash(f"Model loaded but a test prediction failed — double check feature columns/order: {str(e)[:300]}", "error")
        return redirect(url_for("project_detail", project_id=project_id))

    dataset_id = db.get_or_create_external_dataset_id(project_id, user["id"], _now())
    db.create_model(
        project_id, dataset_id, model_name, task_type, "imported_onnx", "",
        feature_columns, class_names,
        {"note": "Imported model — Anvil did not train or evaluate this model."},
        [], str(artifact_path), "ready", user["id"], _now(),
        source="imported_onnx", runtime="onnx",
    )
    flash(f"Imported '{model_name}' as an ONNX model.", "success")
    return redirect(url_for("project_detail", project_id=project_id))


# ---------------------------------------------------------------------------
# Model detail + export + try-it
# ---------------------------------------------------------------------------
def _get_model_scoped(model_id):
    team = auth.current_team()
    if not team:
        return None, None
    model_row = db.get_model(model_id)
    if not model_row:
        return None, None
    project = db.get_project(model_row["project_id"], team["id"])
    if not project:
        return None, None
    return model_row, project


@app.route("/models/<int:model_id>")
@login_required
def model_detail(model_id):
    model_row, project = _get_model_scoped(model_id)
    if not model_row:
        flash("Model not found.", "error")
        return redirect(url_for("dashboard"))

    metrics = json.loads(model_row["metrics_json"])
    leaderboard = json.loads(model_row["leaderboard_json"])
    feature_columns = json.loads(model_row["feature_columns_json"])
    class_names = json.loads(model_row["class_names_json"])
    team = auth.current_team()
    api_keys = db.list_api_keys(team["id"])
    recent = db.recent_predictions(model_id, limit=10)
    is_imported = model_row["source"] == "imported_onnx"
    universal_available = (not is_imported) and model_row["algorithm"] in UNIVERSAL_EXPORT_SUPPORTED \
        and project["task_kind"] == "tabular"

    return render_template(
        "model_detail.html", model=model_row, project=project, metrics=metrics,
        leaderboard=leaderboard, feature_columns=feature_columns, class_names=class_names,
        algorithm_labels=ALGORITHM_LABELS, api_keys=api_keys, recent=recent,
        universal_available=universal_available, is_imported=is_imported,
    )


@app.route("/models/<int:model_id>/export/onnx")
@login_required
def export_onnx(model_id):
    model_row, project = _get_model_scoped(model_id)
    if not model_row:
        flash("Model not found.", "error")
        return redirect(url_for("dashboard"))
    if model_row["source"] != "imported_onnx":
        flash("This model wasn't imported as ONNX.", "error")
        return redirect(url_for("model_detail", model_id=model_id))
    return send_file(
        model_row["artifact_path"], mimetype="application/octet-stream", as_attachment=True,
        download_name=f"{secure_filename(model_row['name'])}.onnx",
    )


@app.route("/models/<int:model_id>/export/pickle")
@login_required
def export_pickle(model_id):
    model_row, project = _get_model_scoped(model_id)
    if not model_row:
        flash("Model not found.", "error")
        return redirect(url_for("dashboard"))
    if model_row["source"] == "imported_onnx":
        flash("Imported ONNX models don't have a pickle bundle — download the .onnx file instead.", "error")
        return redirect(url_for("model_detail", model_id=model_id))
    data = export_bundle.build_pickle_bundle_zip(
        model_row["artifact_path"], model_row["name"], model_row["task_type"],
        model_row["algorithm"], json.loads(model_row["feature_columns_json"]),
        json.loads(model_row["class_names_json"]),
        pipeline=joblib.load(model_row["artifact_path"]) if project["task_kind"] == "tabular" else None,
    )
    return send_file(
        io.BytesIO(data), mimetype="application/zip", as_attachment=True,
        download_name=f"{secure_filename(model_row['name'])}_pickle_bundle.zip",
    )


@app.route("/models/<int:model_id>/export/universal")
@login_required
def export_universal_route(model_id):
    model_row, project = _get_model_scoped(model_id)
    if not model_row:
        flash("Model not found.", "error")
        return redirect(url_for("dashboard"))
    if model_row["source"] == "imported_onnx":
        flash("Imported ONNX models already are the universal, dependency-free format.", "error")
        return redirect(url_for("model_detail", model_id=model_id))
    pipeline = joblib.load(model_row["artifact_path"])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            data = export_bundle.build_universal_zip(
                pipeline, model_row["algorithm"], model_row["task_type"],
                json.loads(model_row["class_names_json"]), model_row["name"], Path(tmp),
            )
        except Exception as e:
            flash(str(e), "error")
            return redirect(url_for("model_detail", model_id=model_id))
    return send_file(
        io.BytesIO(data), mimetype="application/zip", as_attachment=True,
        download_name=f"{secure_filename(model_row['name'])}_universal_export.zip",
    )


@app.route("/models/<int:model_id>/try", methods=["POST"])
@login_required
def try_model(model_id):
    model_row, project = _get_model_scoped(model_id)
    if not model_row:
        return jsonify({"error": "Model not found."}), 404

    try:
        if model_row["runtime"] == "onnx":
            feature_columns = json.loads(model_row["feature_columns_json"])
            class_names = json.loads(model_row["class_names_json"])
            session = onnx_import.load_session(model_row["artifact_path"])
            payload = {k: v for k, v in request.form.items()}
            pred, proba = onnx_import.predict_single(
                session, feature_columns, class_names, model_row["task_type"], payload
            )
        elif project["task_kind"] == "image":
            pipeline = joblib.load(model_row["artifact_path"])
            file = request.files.get("image")
            if not file:
                return jsonify({"error": "Upload an image to test."}), 400
            pred, proba = image_training.predict_single(pipeline, file.read())
        else:
            pipeline = joblib.load(model_row["artifact_path"])
            feature_columns = json.loads(model_row["feature_columns_json"])
            payload = {k: v for k, v in request.form.items()}
            pred, proba = tabular_training.predict_single(pipeline, feature_columns, payload)

        result = {"prediction": pred if not hasattr(pred, "item") else pred.item()}
        if proba is not None:
            result["probabilities"] = proba
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 400


@app.route("/models/<int:model_id>/delete", methods=["POST"])
@login_required
def delete_model(model_id):
    model_row, project = _get_model_scoped(model_id)
    if model_row:
        Path(model_row["artifact_path"]).unlink(missing_ok=True)
        db.delete_model(model_id)
        flash("Model deleted.", "success")
        return redirect(url_for("project_detail", project_id=project["id"]))
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
@app.route("/settings/api-keys", methods=["GET", "POST"])
@login_required
def api_keys():
    team = auth.current_team()
    user = auth.current_user()
    if request.method == "POST":
        label = request.form.get("label", "").strip() or "Unlabeled key"
        key_value = "anvil_" + secrets.token_urlsafe(24)
        db.create_api_key(team["id"], key_value, label, user["id"], _now())
        flash(f"New API key created: {key_value} — copy it now, it won't be shown again in full.", "success")
    keys = db.list_api_keys(team["id"])
    members = db.list_team_members(team["id"])
    return render_template("api_keys.html", keys=keys, team=team, members=members)


@app.route("/settings/api-keys/<int:key_id>/revoke", methods=["POST"])
@login_required
def revoke_key(key_id):
    db.revoke_api_key(key_id)
    flash("API key revoked.", "success")
    return redirect(url_for("api_keys"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
