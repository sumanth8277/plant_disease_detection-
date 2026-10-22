"""
Unit tests for configuration loading and validation (tests/test_config.py).
"""

import os
from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig, get_config


def test_default_config_values():
    """Verify default values meet project requirements."""
    assert Config.DB_NAME == "plant_disease_db"
    assert Config.DB_PORT == 3306
    assert Config.DEBUG is False
    assert Config.TESTING is False


def test_get_config_mappings():
    """Verify environment mappings return appropriate configuration classes."""
    assert get_config("development") == DevelopmentConfig
    assert get_config("testing") == TestingConfig
    assert get_config("production") == ProductionConfig
    assert get_config("unknown") == DevelopmentConfig


def test_validate_db_config_detects_placeholder_password(monkeypatch):
    """Verify validation flags default placeholder password safely."""
    monkeypatch.setattr(Config, "DB_PASSWORD", "replace_with_your_password")
    errors = Config.validate_db_config()
    assert any("DB_PASSWORD" in err for err in errors)
    # Ensure error message does not expose raw password values
    assert not any("password=" in err for err in errors)


def test_validate_db_config_missing_host(monkeypatch):
    """Verify validation detects missing host."""
    monkeypatch.setattr(Config, "DB_HOST", "")
    errors = Config.validate_db_config()
    assert any("DB_HOST" in err for err in errors)


def test_validate_db_config_valid(monkeypatch):
    """Verify clean validation when all settings are set."""
    monkeypatch.setattr(Config, "DB_HOST", "127.0.0.1")
    monkeypatch.setattr(Config, "DB_PORT", 3306)
    monkeypatch.setattr(Config, "DB_NAME", "plant_disease_db")
    monkeypatch.setattr(Config, "DB_USER", "plant_app")
    monkeypatch.setattr(Config, "DB_PASSWORD", "StrongPassword123!")
    errors = Config.validate_db_config()
    assert len(errors) == 0


def test_db_config_dictionary():
    """Verify get_db_config returns required keys for connection pooling."""
    db_dict = Config.get_db_config()
    required_keys = {"host", "port", "database", "user", "password", "pool_name", "pool_size"}
    assert required_keys.issubset(db_dict.keys())
    assert db_dict["charset"] == "utf8mb4"
