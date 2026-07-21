"""
Authentication for Anvil.

Team-scoped, not global: a "team" is the shared-internal-tool unit. The
first user to create a team becomes its admin and gets an invite code;
everyone else joins with that code. Sessions are Flask's signed cookie
session — fine for an internal tool behind normal HTTPS/VPN, not meant to
replace SSO for a public-facing product.
"""

import secrets
import string
from functools import wraps
from datetime import datetime, timezone

from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

from src import db


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_invite_code(length: int = 8) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_team_with_admin(team_name: str, email: str, password: str, display_name: str):
    invite_code = _gen_invite_code()
    team_id = db.create_team(team_name, invite_code, _now())
    user_id = db.create_user(
        team_id, email.lower().strip(), generate_password_hash(password),
        display_name, "admin", _now(),
    )
    return team_id, user_id, invite_code


def join_team_with_code(invite_code: str, email: str, password: str, display_name: str):
    team = db.get_team_by_invite_code(invite_code.strip().upper())
    if not team:
        return None, "Invalid invite code."
    if db.get_user_by_email(email.lower().strip()):
        return None, "That email is already registered."
    user_id = db.create_user(
        team["id"], email.lower().strip(), generate_password_hash(password),
        display_name, "member", _now(),
    )
    return user_id, None


def authenticate(email: str, password: str):
    user = db.get_user_by_email(email.lower().strip())
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def login_user(user):
    session["user_id"] = user["id"]
    session["team_id"] = user["team_id"]


def logout_user():
    session.clear()


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.get_user(uid)


def current_team():
    tid = session.get("team_id")
    if not tid:
        return None
    return db.get_team(tid)
