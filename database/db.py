"""
MySQL Database Connection Module
Provides connection pooling, safe context managers, parameterized execution,
and health check diagnostics using mysql-connector-python.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

import mysql.connector
from mysql.connector import Error, errorcode
from mysql.connector.pooling import MySQLConnectionPool, PooledMySQLConnection

from config import Config

logger = logging.getLogger(__name__)

# Module-level connection pool (initialized lazily)
_pool: Optional[MySQLConnectionPool] = None


def sanitize_db_error(err: Exception) -> str:
    """
    Strips potential credentials or sensitive internal data from database errors.
    Returns a safe, user-friendly error message.
    """
    if isinstance(err, Error):
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            return "Database authentication failed: invalid username or password."
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            return f"Database '{Config.DB_NAME}' does not exist on the server."
        if err.errno == 2003:  # CR_CONN_HOST_ERROR
            return f"Cannot connect to MySQL server at {Config.DB_HOST}:{Config.DB_PORT}. Is MySQL running?"
        return f"Database error [{err.errno}]: {err.msg}"
    return "An unexpected database connection error occurred."


def get_pool() -> Optional[MySQLConnectionPool]:
    """
    Returns the current MySQLConnectionPool or creates one if not already initialized.
    Does not raise an exception if MySQL is unavailable; returns None instead.
    """
    global _pool
    if _pool is not None:
        return _pool

    # Validate that configuration is present before attempting pool creation
    config_errors = Config.validate_db_config()
    if config_errors:
        logger.warning(
            "MySQL pool creation deferred due to configuration: %s",
            "; ".join(config_errors),
        )
        return None

    try:
        db_config = Config.get_db_config()
        _pool = MySQLConnectionPool(
            pool_name=db_config["pool_name"],
            pool_size=db_config["pool_size"],
            pool_reset_session=True,
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            charset=db_config["charset"],
            collation=db_config["collation"],
            autocommit=db_config["autocommit"],
        )
        logger.info(
            "MySQL connection pool '%s' initialized (size=%d).",
            db_config["pool_name"],
            db_config["pool_size"],
        )
        return _pool
    except Error as err:
        safe_msg = sanitize_db_error(err)
        logger.error("Failed to initialize MySQL connection pool: %s", safe_msg)
        return None


def get_raw_connection() -> PooledMySQLConnection:
    """
    Retrieves a connection from the pool.
    Raises ConnectionError with safe message if pool cannot provide a connection.
    """
    pool = get_pool()
    if pool is None:
        raise ConnectionError(
            "MySQL connection pool is not available. Please verify MySQL service and credentials in .env."
        )
    try:
        return pool.get_connection()
    except Error as err:
        raise ConnectionError(sanitize_db_error(err)) from None


@contextmanager
def get_db_connection() -> Generator[PooledMySQLConnection, None, None]:
    """
    Context manager for borrowing a connection from the pool.
    Ensures rollback on error and proper return to the pool upon exit.
    """
    conn = get_raw_connection()
    try:
        yield conn
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.error("Transaction rolled back due to error: %s", exc)
        raise
    finally:
        try:
            conn.close()  # Returns connection back to the pool
        except Exception:
            pass


@contextmanager
def get_db_cursor(
    commit: bool = False, dictionary: bool = True
) -> Generator[Any, None, None]:
    """
    Context manager providing a cursor and handling transactions cleanly.

    Args:
        commit: If True, commits transaction upon successful block execution.
        dictionary: If True, cursor returns rows as dicts; otherwise as tuples.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor(dictionary=dictionary)
        try:
            yield cursor
            if commit:
                conn.commit()
        finally:
            try:
                cursor.close()
            except Exception:
                pass


def execute_query(
    query: str,
    params: Optional[Tuple[Any, ...] | dict] = None,
    fetchall: bool = False,
    fetchone: bool = False,
    commit: bool = False,
) -> Any:
    """
    Helper to execute a parameterized query safely.

    Args:
        query: SQL string with %s parameter placeholders.
        params: Tuple or dict of parameters.
        fetchall: Return all matching rows.
        fetchone: Return a single matching row.
        commit: Commit transaction (for INSERT/UPDATE/DELETE).
    """
    with get_db_cursor(commit=commit, dictionary=True) as cursor:
        cursor.execute(query, params or ())
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        if commit:
            return cursor.lastrowid or cursor.rowcount
        return None


def check_db_connection() -> Tuple[bool, str]:
    """
    Executes a simple 'SELECT 1;' diagnostic check to test MySQL connectivity.
    Returns (True, "MySQL connection healthy") or (False, "<safe error message>").
    Never exposes passwords.
    """
    config_errors = Config.validate_db_config()
    if config_errors:
        return False, f"Configuration incomplete: {'; '.join(config_errors)}"

    try:
        with get_db_cursor(commit=False, dictionary=False) as cursor:
            cursor.execute("SELECT 1 AS ping;")
            result = cursor.fetchone()
            if result and result[0] == 1:
                return True, "MySQL connection healthy."
            return False, "Unexpected response from MySQL ping query."
    except Exception as err:
        return False, sanitize_db_error(err)
