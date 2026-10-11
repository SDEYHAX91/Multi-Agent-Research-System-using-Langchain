"""
Authentication for ResearchScribe (SQLite + PBKDF2, Streamlit UI).

Usage in app.py (right after st.set_page_config + CSS):
    from auth import require_login, render_account_sidebar
    user = require_login()          # shows login / sign-up page and stops the script if needed
    ...
    render_account_sidebar()        # inside `with st.sidebar:` -> account info + Log out
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from pathlib import Path
import streamlit as st

# Absolute path, so the DB is found no matter which folder `streamlit run` is started from.
# Override with the AUTH_DB_PATH environment variable if you want it elsewhere.
DB = Path(os.getenv("AUTH_DB_PATH") or Path(__file__).with_name("users.db"))

PBKDF2_ROUNDS = 200_000  # must match the hashes already stored in users.db
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
MIN_PASSWORD = 8
MAX_PASSWORD = 128  # PBKDF2 cost grows with input size; cap it
MAX_NAME = 80
MAX_EMAIL = 254

# ------------------------------------------------------------------ #
# Database
# ------------------------------------------------------------------ #
def init_db() -> None:
    with sqlite3.connect(DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ROUNDS
    ).hex()
    return salt, hashed

def create_user(name: str, email: str, password: str) -> bool:
    salt, hashed = hash_password(password)
    try:
        with sqlite3.connect(DB) as conn:
            conn.execute(
                "INSERT INTO users (name, email, salt, password_hash) VALUES (?, ?, ?, ?)",
                (name, email, salt, hashed))
        return True
    except sqlite3.IntegrityError:  # email already registered
        return False

_DUMMY_SALT = secrets.token_hex(16)

def authenticate(email: str, password: str) -> dict | None:
    with sqlite3.connect(DB) as conn:
        row = conn.execute(
            "SELECT id, name, email, salt, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    if not row:
        hash_password(password, _DUMMY_SALT)  # keep timing similar for unknown emails
        return None

    uid, name, mail, salt, stored = row
    _, computed = hash_password(password, salt)
    if hmac.compare_digest(computed, stored):
        return {"id": uid, "name": name, "email": mail}
    return None

# ------------------------------------------------------------------ #
# Session helpers
# ------------------------------------------------------------------ #
def current_user() -> dict | None:
    return st.session_state.get("user")

def logout() -> None:
    """Wipe everything (user, history, results...) so nothing leaks to the next login."""
    st.session_state.clear()

_dialog = getattr(st, "dialog", None) or st.experimental_dialog

@_dialog("Log out?")
def confirm_logout_dialog() -> None:
    st.markdown(
        '<div class="dlg-note" style="margin-top:0">You will be signed out and returned '
        "to the login page.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(key="dlg_cancel_logout"):
            if st.button("Cancel", use_container_width=True):
                st.rerun()  # closes the dialog
    with c2:
        with st.container(key="dlg_confirm_logout"):
            if st.button("Log out", type="primary", use_container_width=True):
                logout()
                st.rerun()  # back to the login page

# ------------------------------------------------------------------ #
# Styling (re-uses the CSS variables defined in app.py)
# ------------------------------------------------------------------ #
_AUTH_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {display: none !important;}

.st-key-auth_card {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.4rem 1.6rem 1.2rem 1.6rem;
}
.auth-title {
  font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.35rem;
  color: var(--text); margin: .2rem 0 .2rem 0;
}
.auth-sub {color: var(--muted) !important; font-size: .9rem; margin-bottom: .6rem;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: 1px solid var(--border);}
.stTabs [data-baseweb="tab"] p {
  font-family: 'JetBrains Mono', monospace; font-size: .78rem; letter-spacing: .08em;
  color: var(--muted);
}
.stTabs [aria-selected="true"] p {color: var(--accent) !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: var(--accent) !important;}
.stTabs [data-baseweb="tab-border"] {background-color: transparent !important;}
.stTabs [data-baseweb="tab-panel"] {padding-top: 1.1rem;}
</style>
"""

_SIDEBAR_CSS = """
<style>
.acct-label {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; letter-spacing: .18em;
  color: var(--accent); margin: 1.6rem 0 .5rem 0;
}
.acct-box {
  background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
  padding: .6rem .8rem; margin-bottom: .6rem;
}
.acct-name {font-weight: 600; font-size: .9rem; color: var(--text); word-break: break-word;}
.acct-email {
  font-family: 'JetBrains Mono', monospace; font-size: .68rem; color: var(--muted) !important;
  word-break: break-all; margin-top: .15rem;
}
.st-key-logoutbtn button {
  background: transparent !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
.st-key-logoutbtn button p {color: var(--muted) !important; font-weight: 600;}
.st-key-logoutbtn button:hover {border-color: #ff5c6c !important;}
.st-key-logoutbtn button:hover p {color: #ff5c6c !important;}
</style>
"""

# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #
def _validate_registration(name, email, password, confirm) -> str | None:
    if not name or not email or not password or not confirm:
        return "Please fill in all fields."
    if len(name) > MAX_NAME or len(email) > MAX_EMAIL:
        return "Name or email is too long."
    if not EMAIL_RE.fullmatch(email):
        return "Please enter a valid email address."
    if len(password) < MIN_PASSWORD:
        return f"Password must contain at least {MIN_PASSWORD} characters."
    if len(password) > MAX_PASSWORD:
        return f"Password must be at most {MAX_PASSWORD} characters."
    if password != confirm:
        return "Passwords do not match."
    return None
  

def _render_auth_page(app_name: str) -> None:
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    head, tail = (app_name[:8], app_name[8:]) if app_name == "ResearchScribe" else (app_name, "")
    st.markdown(
        f"""<div class="eyebrow">Sign in to continue</div>
<div class="brand"><span class="b1">{head}</span><span class="b2">{tail}</span></div>
<div class="fade-line"></div>""",
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.container(key="auth_card"):
            login_tab, register_tab = st.tabs(["🔑 Login", "✨ Create Account"])

            # ---------------- Login ----------------
            with login_tab:
                st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="auth-sub">Enter your credentials to access your account.</div>',
                    unsafe_allow_html=True,
                )
                with st.form("login_form", border=False):
                    email = st.text_input(
                        "Email address", key="login_email", placeholder="you@example.com"
                    ).strip().lower()
                    password = st.text_input(
                        "Password", key="login_password", type="password",
                        placeholder="Enter your password",
                    )
                    submitted = st.form_submit_button("Login  →", type="primary")

                if submitted:
                    user = authenticate(email, password) if email and password else None
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

            # ---------------- Register ----------------
            with register_tab:
                st.markdown('<div class="auth-title">Create your account</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="auth-sub">Join by filling in the details below.</div>',
                    unsafe_allow_html=True,
                )
                with st.form("register_form", border=False):
                    name = st.text_input("Full name", key="reg_name", placeholder="Your name").strip()
                    email = st.text_input(
                        "Email address", key="reg_email", placeholder="you@example.com"
                    ).strip().lower()
                    password = st.text_input(
                        "Create password", key="reg_password", type="password",
                        placeholder=f"At least {MIN_PASSWORD} characters",
                    )
                    confirm = st.text_input(
                        "Confirm password", key="reg_confirm", type="password",
                        placeholder="Re-enter your password",
                    )
                    submitted = st.form_submit_button("Create Account  →", type="primary")

                if submitted:
                    error = _validate_registration(name, email, password, confirm)
                    if error:
                        st.error(error)
                    elif create_user(name, email, password):
                        st.success("Account created! Switch to the Login tab to sign in.")
                    else:
                        st.error("An account with this email already exists.")


def require_login(app_name: str = "ResearchScribe") -> dict:
    """Gate for the whole app. Returns the logged-in user, or shows the auth page and halts the script."""
    init_db()
    user = current_user()
    if user:
        return user
    _render_auth_page(app_name)
    st.stop()


def render_account_sidebar() -> None:
    """Call inside `with st.sidebar:` — shows who is signed in and a Log out button."""
    user = current_user()
    if not user:
        return
    from html import escape

    st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown('<div class="acct-label">ACCOUNT</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="acct-box"><div class="acct-name">{escape(user["name"])}</div>'
        f'<div class="acct-email">{escape(user["email"])}</div></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="logoutbtn"):
        if st.button("⎋  Log out", use_container_width=True):
            confirm_logout_dialog()
