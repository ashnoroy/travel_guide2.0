# auth.py
# Simple username/password authentication for the Wanderly Streamlit app.
# Storage: SQLite (users.db, created next to this file on first run)
# Hashing: bcrypt (industry-standard, salted, slow-by-design)
#
# ---------------------------------------------------------------------------
# IMPORTANT - if you deploy this on Streamlit Community Cloud:
# The filesystem there is EPHEMERAL. Every time the app redeploys (new git
# push) or wakes up after being asleep, the container is rebuilt from your
# repo and users.db is wiped, so all registered accounts are lost.
# This is fine for a demo / course project. For real persistence, swap the
# sqlite3 calls below for a hosted DB (Postgres, Supabase, Neon, etc.) - the
# function signatures (register_user, verify_login) can stay the same, so
# the rest of the app doesn't need to change.
# ---------------------------------------------------------------------------

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import bcrypt

DB_PATH = Path(__file__).parent / "users.db"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the users table if it doesn't exist yet. Safe to call every run."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _validate_registration(username: str, email: str, password: str, confirm: str) -> Optional[str]:
    """Return a human-readable error string, or None if the input is valid."""
    if not username or not email or not password or not confirm:
        return "Please fill in every field."
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if not USERNAME_RE.match(username):
        return "Username can only contain letters, numbers, and underscores."
    if not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if password != confirm:
        return "Passwords do not match."
    return None


def register_user(username: str, email: str, password: str, confirm: str) -> Tuple[bool, str]:
    """Attempt to create a new account. Returns (success, message)."""
    username = username.strip()
    email = email.strip().lower()

    error = _validate_registration(username, email, password, confirm)
    if error:
        return False, error

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash),
            )
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def verify_login(identifier: str, password: str) -> Tuple[bool, str]:
    """
    Check a username-or-email + password combo.
    Returns (True, username) on success, or (False, error_message) on failure.
    """
    identifier = identifier.strip().lower()
    if not identifier or not password:
        return False, "Please enter your username/email and password."

    with get_conn() as conn:
        row = conn.execute(
            "SELECT username, password_hash FROM users WHERE lower(username) = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()

    if row is None:
        return False, "No account found with that username or email."

    if bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return True, row["username"]
    return False, "Incorrect password."
