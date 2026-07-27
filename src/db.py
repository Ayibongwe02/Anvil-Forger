"""
Persistence layer for Anvil.

SQLite, stdlib only. Schema covers: teams (the "shared internal tool" unit),
users (scoped to a team), projects, datasets, training runs, models
(a trained artifact + metrics + export state), and API keys for the hosted
prediction endpoint.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "anvil.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    invite_code TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    task_kind TEXT NOT NULL,           -- 'tabular' | 'image'
    description TEXT DEFAULT '',
    created_by INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    kind TEXT NOT NULL,                -- 'csv' | 'image_folder_zip'
    meta_json TEXT DEFAULT '{}',       -- columns, row count, class names, etc.
    uploaded_by INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    dataset_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,           -- 'classification' | 'regression'
    algorithm TEXT NOT NULL,           -- winning algorithm name
    target_column TEXT DEFAULT '',
    feature_columns_json TEXT DEFAULT '[]',
    class_names_json TEXT DEFAULT '[]',
    metrics_json TEXT NOT NULL,
    leaderboard_json TEXT NOT NULL,
    artifact_path TEXT NOT NULL,       -- pickled bundle (or .onnx file) on disk
    status TEXT NOT NULL DEFAULT 'ready',  -- 'training' | 'ready' | 'failed'
    error_message TEXT DEFAULT '',
    trained_by INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'trained',   -- 'trained' | 'imported_onnx'
    runtime TEXT NOT NULL DEFAULT 'sklearn',  -- 'sklearn' | 'onnx'
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    key_value TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    created_by INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (team_id) REFERENCES teams (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    api_key_id INTEGER,
    input_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
);
"""


@contextmanager
def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Migrate older databases created before 'source'/'runtime' existed.
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(models)")}
        if "source" not in existing_cols:
            conn.execute("ALTER TABLE models ADD COLUMN source TEXT NOT NULL DEFAULT 'trained'")
        if "runtime" not in existing_cols:
            conn.execute("ALTER TABLE models ADD COLUMN runtime TEXT NOT NULL DEFAULT 'sklearn'")


# ---------------------------------------------------------------------------
# Teams / users
# ---------------------------------------------------------------------------
def create_team(name: str, invite_code: str, created_at: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO teams (name, invite_code, created_at) VALUES (?, ?, ?)",
            (name, invite_code, created_at),
        )
        return cur.lastrowid


def get_team_by_invite_code(invite_code: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM teams WHERE invite_code = ?", (invite_code,)).fetchone()


def get_team(team_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,)).fetchone()


def create_user(team_id, email, password_hash, display_name, role, created_at) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO users (team_id, email, password_hash, display_name, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (team_id, email, password_hash, display_name, role, created_at),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()


def get_user(user_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def list_team_members(team_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE team_id = ? ORDER BY created_at", (team_id,)
        ).fetchall()


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------
def create_project(team_id, name, task_kind, description, created_by, created_at) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO projects (team_id, name, task_kind, description, created_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (team_id, name, task_kind, description, created_by, created_at),
        )
        return cur.lastrowid


def list_projects(team_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM projects WHERE team_id = ? ORDER BY created_at DESC", (team_id,)
        ).fetchall()


def get_project(project_id: int, team_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM projects WHERE id = ? AND team_id = ?", (project_id, team_id)
        ).fetchone()


def delete_project(project_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM models WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM datasets WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def create_dataset(project_id, name, file_path, kind, meta: dict, uploaded_by, created_at) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO datasets (project_id, name, file_path, kind, meta_json, uploaded_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project_id, name, file_path, kind, json.dumps(meta), uploaded_by, created_at),
        )
        return cur.lastrowid


def list_datasets(project_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
        ).fetchall()


def get_dataset(dataset_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()


def get_or_create_external_dataset_id(project_id: int, uploaded_by, created_at) -> int:
    """Imported (bring-your-own) models still need a dataset_id to satisfy the
    models table's foreign key, since they don't come from a CSV/zip trained
    in Anvil. Reuse one placeholder 'external' dataset row per project."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM datasets WHERE project_id = ? AND kind = 'external'", (project_id,)
        ).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            """INSERT INTO datasets (project_id, name, file_path, kind, meta_json, uploaded_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project_id, "(externally trained — no dataset)", "", "external",
             json.dumps({}), uploaded_by, created_at),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def create_model(project_id, dataset_id, name, task_type, algorithm, target_column,
                  feature_columns, class_names, metrics: dict, leaderboard: list,
                  artifact_path, status, trained_by, created_at, error_message="",
                  source="trained", runtime="sklearn") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO models
               (project_id, dataset_id, name, task_type, algorithm, target_column,
                feature_columns_json, class_names_json, metrics_json, leaderboard_json,
                artifact_path, status, error_message, trained_by, created_at, source, runtime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, dataset_id, name, task_type, algorithm, target_column,
             json.dumps(feature_columns), json.dumps(class_names), json.dumps(metrics),
             json.dumps(leaderboard), artifact_path, status, error_message, trained_by, created_at,
             source, runtime),
        )
        return cur.lastrowid


def list_models(project_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM models WHERE project_id = ? ORDER BY created_at DESC", (project_id,)
        ).fetchall()


def get_model(model_id: int):
    with get_conn() as conn:
        return conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()


def delete_model(model_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
def create_api_key(team_id, key_value, label, created_by, created_at) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO api_keys (team_id, key_value, label, created_by, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (team_id, key_value, label, created_by, created_at),
        )
        return cur.lastrowid


def list_api_keys(team_id: int):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM api_keys WHERE team_id = ? ORDER BY created_at DESC", (team_id,)
        ).fetchall()


def get_api_key(key_value: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM api_keys WHERE key_value = ? AND revoked = 0", (key_value,)
        ).fetchone()


def revoke_api_key(key_id: int):
    with get_conn() as conn:
        conn.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))


# ---------------------------------------------------------------------------
# Prediction log (lightweight audit trail for the hosted API)
# ---------------------------------------------------------------------------
def log_prediction(model_id, api_key_id, input_data, output_data, created_at) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO prediction_log (model_id, api_key_id, input_json, output_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (model_id, api_key_id, json.dumps(input_data), json.dumps(output_data), created_at),
        )
        return cur.lastrowid


def recent_predictions(model_id: int, limit: int = 20):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM prediction_log WHERE model_id = ? ORDER BY created_at DESC LIMIT ?",
            (model_id, limit),
        ).fetchall()
