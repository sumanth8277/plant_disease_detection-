"""
Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System
Flask Application Entry Point and Application Factory (Phases 1 & 2)
"""

import os
from typing import Optional, Type

from flask import Flask
from flask_wtf.csrf import CSRFProtect

from config import Config, get_config
from routes.auth_routes import auth_bp
from routes.main_routes import register_routes

# Global CSRF protection instance
csrf = CSRFProtect()


def create_app(config_class: Optional[Type[Config]] = None) -> Flask:
    """
    Application factory pattern for Flask.
    Initializes Flask app, registers configurations, attaches CSRF protection,
    and registers application routes.
    """
    app = Flask(__name__)

    # Load configuration
    if config_class is None:
        config_class = get_config()
    app.config.from_object(config_class)

    # Initialize CSRF protection
    csrf.init_app(app)

    # Ensure upload directory exists
    upload_folder = app.config.get("UPLOAD_FOLDER", Config.UPLOAD_FOLDER)
    upload_folder.mkdir(parents=True, exist_ok=True)

    # Register application blueprints and routes
    app.register_blueprint(auth_bp)
    register_routes(app)

    return app


# Application instance for standard WSGI servers or flask CLI
app = create_app()

if __name__ == "__main__":
    # Run development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=app.config.get("DEBUG", False))
