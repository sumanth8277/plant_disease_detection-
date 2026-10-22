"""
Authentication & Authorization Utilities (utils/auth.py)
Provides login-required decorators, open-redirect prevention,
and credential validation helpers for user authentication.
"""

import functools
import re
from urllib.parse import urljoin, urlparse
from typing import Optional, Tuple

from flask import current_app, flash, redirect, request, session, url_for


def is_safe_url(target: Optional[str]) -> bool:
    """
    Validates that a redirect target URL is safe and points to the same host.
    Prevents open redirect attacks (e.g., https://evil.com or //evil.com).
    """
    if not target:
        return False
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ("http", "https") and ref_url.netloc == test_url.netloc


def login_required(view_func):
    """
    Decorator protecting routes that require authenticated user access.
    Redirects unauthenticated visitors to login page with safe 'next' parameter.
    """
    @functools.wraps(view_func)
    def decorated_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "warning")
            next_url = request.full_path if request.method == "GET" else None
            if next_url and next_url.endswith("?"):
                next_url = next_url[:-1]
            return redirect(url_for("auth.login", next=next_url if is_safe_url(next_url) else None))
        return view_func(*args, **kwargs)

    return decorated_view


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """
    Validates username format: 3-50 chars, letters, numbers, and underscores only.
    """
    if not username or not isinstance(username, str):
        return False, "Username is required."
    uname = username.strip()
    if len(uname) < 3 or len(uname) > 50:
        return False, "Username must be between 3 and 50 characters."
    if not re.match(r"^[A-Za-z0-9_]+$", uname):
        return False, "Username may only contain letters, numbers, and underscores."
    return True, None


def validate_email_format(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validates email format syntax.
    """
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    clean_email = email.strip()
    if len(clean_email) > 190:
        return False, "Email address exceeds maximum length of 190 characters."
    # Standard RFC-compliant pattern
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(pattern, clean_email):
        return False, "Please enter a valid email address (e.g., user@example.com)."
    return True, None


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    """
    Enforces password complexity:
    - Minimum 8 characters
    - At least one letter (a-z or A-Z)
    - At least one digit (0-9)
    """
    if not password or not isinstance(password, str):
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters in length."
    if len(password) > 128:
        return False, "Password exceeds maximum allowable length (128 characters)."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, None
