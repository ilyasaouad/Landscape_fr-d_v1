import logging
import os
import urllib.parse
from contextlib import contextmanager
from typing import Optional

import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)
load_dotenv()

# Module-level engine (singleton pattern)
_engine = None


def _validate_env_vars() -> None:
    """Validate required environment variables are set"""
    required_vars = ["db_host", "db_database", "db_username", "db_password"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")


def _build_connection_string() -> str:
    """Build ODBC connection string from environment variables"""
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f'SERVER={os.getenv("db_host")};'
        f'DATABASE={os.getenv("db_database")};'
        f'UID={os.getenv("db_username")};'
        f'PWD={os.getenv("db_password")}'
    )


def _get_engine():
    """Get or create SQLAlchemy engine (singleton)"""
    global _engine

    if _engine is not None:
        return _engine

    try:
        _validate_env_vars()

        conn_str = _build_connection_string()
        encoded_conn_str = urllib.parse.quote_plus(conn_str)
        connection_url = f"mssql+pyodbc:///?odbc_connect={encoded_conn_str}"

        _engine = create_engine(
            connection_url,
            echo=os.getenv("DB_ECHO", "False").lower() == "true",
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"timeout": 30},
        )

        # Test connection
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        logger.info("Database engine created and tested successfully")
        return _engine

    except Exception as e:
        logger.error(f"Failed to create database engine: {e}", exc_info=True)
        raise


def cleanup_engine() -> None:
    """Clean up engine resources (call at application shutdown)"""
    global _engine
    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("Database engine cleaned up")


@contextmanager
def get_session():
    """Context manager for SQLAlchemy sessions with transaction handling

    Usage:
        with get_session() as session:
            result = session.query(Model).all()
    """
    engine = _get_engine()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Session error, rolling back: {e}")
        raise
    finally:
        session.close()


def get_raw_connection() -> Optional[pyodbc.Connection]:
    """Get raw PyODBC connection (use only if SQLAlchemy is insufficient)

    Returns:
        pyodbc.Connection or None if connection fails
    """
    try:
        conn_str = _build_connection_string()
        conn = pyodbc.connect(conn_str)
        logger.info("PyODBC connection successful")
        return conn
    except pyodbc.Error as err:
        logger.error(f"PyODBC connection failed: {err}")
        return None
