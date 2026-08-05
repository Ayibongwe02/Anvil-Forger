"""
Optional outbound email for Anvil (password reset, etc.).

Uses stdlib smtplib only. If SMTP is not configured, callers should fall
back to showing the reset link in the UI (internal-tool mode).
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage


class EmailNotConfigured(Exception):
    pass


class EmailSendError(Exception):
    pass


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_FROM"))


def send_email(to_address: str, subject: str, body_text: str) -> None:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", "").strip()
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")

    if not host or not from_addr:
        raise EmailNotConfigured(
            "SMTP is not configured. Set SMTP_HOST and SMTP_FROM (and optionally "
            "SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_USE_TLS)."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_address
    msg.set_content(body_text)

    try:
        if use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                if user:
                    server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                if user:
                    server.login(user, password)
                server.send_message(msg)
    except Exception as e:
        raise EmailSendError(str(e)[:300]) from e


def send_password_reset_email(to_address: str, reset_url: str) -> None:
    body = (
        "You requested a password reset for your Anvil account.\n\n"
        f"Open this link to choose a new password (valid for 1 hour):\n\n"
        f"{reset_url}\n\n"
        "If you did not request this, you can ignore this email.\n"
    )
    send_email(to_address, "Reset your Anvil password", body)
