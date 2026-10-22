"""
Authentication & User Dashboard Routes Module (routes/auth_routes.py)
Provides endpoints for user registration, secure login with Werkzeug password verification,
logout with CSRF/POST validation, and authenticated prediction history dashboard.
"""

import logging
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from services.auth_service import (
    authenticate_user,
    get_user_prediction_history,
    register_user,
)
from utils.auth import is_safe_url, login_required

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    User registration view.
    GET: Displays registration form.
    POST: Validates inputs, checks for uniqueness, hashes password, inserts into MySQL.
    """
    if session.get("user_id"):
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    success, user_id, message = register_user(
        username=username,
        email=email,
        password=password,
        confirm_password=confirm_password,
    )

    if success:
        flash(message, "success")
        return redirect(url_for("auth.login"))

    flash(message, "danger")
    return render_template(
        "register.html",
        username=username,
        email=email,
    ), 400


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    User login view.
    GET: Displays login form.
    POST: Validates credentials, checks password hash, establishes session,
          and redirects safely with open-redirect protection.
    """
    if session.get("user_id"):
        return redirect(url_for("index"))

    next_page = request.args.get("next") or request.form.get("next")

    if request.method == "GET":
        return render_template("login.html", next=next_page)

    identifier = request.form.get("username_or_email", "").strip()
    password = request.form.get("password", "")

    success, user_info, message = authenticate_user(identifier, password)

    if success and user_info:
        # Establish protected session
        session.clear()
        session["user_id"] = user_info["id"]
        session["user_name"] = user_info["name"]
        session["user_email"] = user_info["email"]
        session.permanent = True

        flash(f"Welcome back, {user_info['name']}!", "success")

        # Validate redirect target to prevent open redirects
        if next_page and is_safe_url(next_page):
            return redirect(next_page)
        return redirect(url_for("index"))

    flash(message, "danger")
    return render_template("login.html", next=next_page, identifier=identifier), 401


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Logs out the authenticated user.
    Requires POST method to prevent CSRF logout via GET navigation or image tags.
    """
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("index"))


@auth_bp.route("/history", methods=["GET"])
@login_required
def history():
    """
    User Prediction History Dashboard.
    Displays authenticated user's own diagnostic records with pagination.
    """
    user_id = session.get("user_id")
    try:
        page = int(request.args.get("page", 1))
    except (ValueError, TypeError):
        page = 1

    per_page = 10
    history_data = get_user_prediction_history(user_id=user_id, page=page, per_page=per_page)

    return render_template(
        "history.html",
        predictions=history_data["predictions"],
        total=history_data["total"],
        page=history_data["page"],
        per_page=history_data["per_page"],
        total_pages=history_data["total_pages"],
        db_error=history_data.get("db_error"),
    )
