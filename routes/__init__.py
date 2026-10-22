"""
Application routes package.
Exports endpoint registration for web interface, API, and upload pipelines.
"""

from routes.auth_routes import auth_bp
from routes.main_routes import register_routes

__all__ = ["register_routes", "auth_bp"]
