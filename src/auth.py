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


def request_password_reset(email: str):
    """Create a one-time reset token for the user if the email exists.
    Returns (token, None) on success or (None, error_message).
    Token is returned so the caller can surface a reset link when no
    outbound email is configured (typical for internal deployments).
    """
    from datetime import timedelta
    user = db.get_user_by_email(email.lower().strip())
    if not user:
        # Don't leak whether the email exists
        return None, None
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    db.create_password_reset_token(user["id"], token, expires, _now())
    return token, None


def reset_password_with_token(token: str, new_password: str):
    """Validate token and set a new password. Returns error string or None."""
    if not new_password or len(new_password) < 8:
        return "Password must be at least 8 characters."
    row = db.get_password_reset_token(token)
    if not row:
        return "This reset link is invalid or has already been used."
    expires = row["expires_at"]
    try:
        exp_dt = datetime.fromisoformat(expires)
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > exp_dt:
            return "This reset link has expired. Please request a new one."
    except ValueError:
        return "This reset link is invalid."
    db.update_user_password(row["user_id"], generate_password_hash(new_password))
    db.mark_password_reset_token_used(token)
    return None


def change_password(user_id: int, current_password: str, new_password: str):
    """Change password for a logged-in user. Returns error string or None."""
    user = db.get_user(user_id)
    if not user:
        return "User not found."
    if not check_password_hash(user["password_hash"], current_password):
        return "Current password is incorrect."
    if not new_password or len(new_password) < 8:
        return "New password must be at least 8 characters."
    if current_password == new_password:
        return "New password must be different from the current password."
    db.update_user_password(user_id, generate_password_hash(new_password))
    return None

