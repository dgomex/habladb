"""
Metadata harvesting and persistence for HablaDB.

Discovers database structure (schemas, tables, columns) via SQLAlchemy
reflection, optionally including column/table descriptions where supported
(PostgreSQL, Redshift). Persists and loads metadata as JSON under databases/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

# Default timeout for connection and reflection (seconds).


def _ensure_duckdb_dialect() -> None:
    """Load the DuckDB dialect so SQLAlchemy can resolve duckdb:// URLs."""
    try:
        import duckdb_engine  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "DuckDB support requires the duckdb-engine package. "
            "Install it with: pip install duckdb-engine"
        ) from e
# Used in create_engine(connect_args={"connect_timeout": CONNECT_TIMEOUT}).
CONNECT_TIMEOUT = 10

DATABASES_DIR = Path(__file__).resolve().parent / "databases"


def path_to_duckdb_url(file_path: str) -> str:
    """
    Convert a filesystem path (or :memory:) to a DuckDB SQLAlchemy URL.
    Resolves to absolute path and uses forward slashes (required by duckdb://).
    """
    s = (file_path or "").strip()
    if s.lower() == ":memory:":
        return "duckdb:///:memory:"
    path = Path(s).expanduser().resolve()
    path_str = str(path).replace("\\", "/")
    # duckdb:///relative or duckdb:////absolute (Unix) or duckdb:///C:/... (Windows)
    return f"duckdb:///{path_str}"


def normalize_connection_string(connection_string: str) -> str:
    """
    Normalize common URL forms so SQLAlchemy can load the correct dialect.
    Converts 'postgres://' to 'postgresql+psycopg2://' (the 'postgres' dialect
    was removed in SQLAlchemy 1.4+). DuckDB URLs are left unchanged.
    """
    s = (connection_string or "").strip()
    if s.startswith("duckdb://"):
        return s
    if s.startswith("postgres://"):
        s = "postgresql+psycopg2://" + s[len("postgres://") :]
    elif s.startswith("postgresql://") and "+" not in s.split("://")[0]:
        s = "postgresql+psycopg2://" + s[len("postgresql://") :]
    return s


def is_duckdb_url(url: str) -> bool:
    """True if the connection URL is for DuckDB."""
    return (url or "").strip().lower().startswith("duckdb://")


def engine_connect_args(url: str) -> dict:
    """Connection args for create_engine; DuckDB does not use connect_timeout."""
    if is_duckdb_url(url):
        return {}
    return {"connect_timeout": CONNECT_TIMEOUT}


def get_discovered_connections() -> dict[str, str]:
    """
    Scan os.environ for variables prefixed with HABLADB_CONN_.
    Returns a mapping of connection name (suffix after prefix) -> connection string.
    """
    prefix = "HABLADB_CONN_"
    conns = {}
    for key, value in os.environ.items():
        if key.startswith(prefix) and value and value.strip():
            name = key[len(prefix) :].strip()
            if name:
                conns[name] = value.strip()
    return conns


def validate_connection(connection_string: str) -> tuple[bool, str]:
    """
    Test the connection string by creating an engine and opening a connection.
    Returns (success, message). Normalizes 'postgres://' to 'postgresql+psycopg2://'.
    Accepts DuckDB URLs (duckdb:///path) as-is.
    """
    url = normalize_connection_string(connection_string)
    if is_duckdb_url(url):
        _ensure_duckdb_dialect()
    connect_args = engine_connect_args(url)
    pool_pre_ping = not is_duckdb_url(url)
    try:
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=pool_pre_ping)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connection successful."
    except SQLAlchemyError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _safe_str(value: Any) -> str:
    """Convert a value to string for JSON; handle non-serializable types."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):  # datetime, date
        return value.isoformat()
    return str(value)


def _harvest_metadata_duckdb(engine: Any, conn_name: str) -> None:
    """
    Harvest schemas, tables, and columns from DuckDB using information_schema
    (DuckDB does not support the PostgreSQL catalog used by SQLAlchemy's Inspector).
    """
    schemas_list: list[dict[str, Any]] = []
    tables_list: list[dict[str, Any]] = []
    columns_list: list[dict[str, Any]] = []

    with engine.connect() as conn:
        # Schemas: standard information_schema.schemata
        r = conn.execute(text(
            "SELECT schema_name FROM information_schema.schemata "
            "ORDER BY schema_name"
        ))
        for row in r:
            schemas_list.append({"schema_name": row[0]})

        # Tables: information_schema.tables (BASE TABLE and VIEW)
        r = conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT LIKE 'information_schema%'
              AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_schema, table_name
        """))
        for row in r:
            tables_list.append({"schema_name": row[0], "table_name": row[1]})

        # Columns: information_schema.columns
        r = conn.execute(text("""
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT LIKE 'information_schema%'
            ORDER BY table_schema, table_name, ordinal_position
        """))
        for row in r:
            columns_list.append({
                "schema_name": row[0],
                "table_name": row[1],
                "column_name": row[2],
                "data_type": _safe_str(row[3]),
                "description": "",
            })

    for filename, data in (
        (f"{conn_name}_schemas.json", schemas_list),
        (f"{conn_name}_tables.json", tables_list),
        (f"{conn_name}_columns.json", columns_list),
    ):
        path = DATABASES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def harvest_metadata(connection_string: str, conn_name: str) -> None:
    """
    Extract schemas, tables, and columns from the database and persist them
    under databases/{conn_name}_schemas.json, _tables.json, _columns.json.
    Raises on connection or reflection errors; handles timeouts via connect_args.
    For DuckDB, uses information_schema (not the PostgreSQL-style Inspector).
    """
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    url = normalize_connection_string(connection_string)
    if is_duckdb_url(url):
        _ensure_duckdb_dialect()
    connect_args = engine_connect_args(url)
    pool_pre_ping = not is_duckdb_url(url)

    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=pool_pre_ping)

    if is_duckdb_url(url):
        _harvest_metadata_duckdb(engine, conn_name)
        return

    inspector = inspect(engine)
    schemas_list: list[dict[str, Any]] = []
    tables_list: list[dict[str, Any]] = []
    columns_list: list[dict[str, Any]] = []

    try:
        schema_names = inspector.get_schema_names()
    except (DBAPIError, SQLAlchemyError) as e:
        raise RuntimeError(f"Failed to list schemas: {e}") from e

    for schema in schema_names:
        schemas_list.append({"schema_name": schema})

        try:
            table_names = inspector.get_table_names(schema=schema)
        except (DBAPIError, SQLAlchemyError) as e:
            raise RuntimeError(f"Failed to list tables in schema {schema!r}: {e}") from e

        for table in table_names:
            tables_list.append({
                "schema_name": schema,
                "table_name": table,
            })

            try:
                cols = inspector.get_columns(table, schema=schema)
            except (DBAPIError, SQLAlchemyError) as e:
                raise RuntimeError(
                    f"Failed to get columns for {schema}.{table}: {e}"
                ) from e

            for col in cols:
                col_type = col.get("type")
                type_str = _safe_str(getattr(col_type, "python_type", col_type))
                if hasattr(col_type, "__visit_name__"):
                    type_str = getattr(col_type, "__visit_name__", type_str)
                columns_list.append({
                    "schema_name": schema,
                    "table_name": table,
                    "column_name": col["name"],
                    "data_type": type_str,
                    "description": _safe_str(col.get("comment") or ""),
                })

    for filename, data in (
        (f"{conn_name}_schemas.json", schemas_list),
        (f"{conn_name}_tables.json", tables_list),
        (f"{conn_name}_columns.json", columns_list),
    ):
        path = DATABASES_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_metadata_context(conn_name: str) -> str:
    """
    Load metadata JSON files for the given connection and build a single
    system-prompt-friendly string describing schemas, tables, and columns.
    """
    parts: list[str] = []

    for suffix, label in (
        ("_schemas.json", "Schemas"),
        ("_tables.json", "Tables"),
        ("_columns.json", "Columns"),
    ):
        path = DATABASES_DIR / f"{conn_name}{suffix}"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not data:
            continue
        parts.append(f"## {label}\n")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            parts.append(" | ".join(keys))
            parts.append("-" * 50)
            for row in data[:500]:  # cap size for prompt
                parts.append(" | ".join(_safe_str(row.get(k, "")) for k in keys))
        parts.append("")

    return "\n".join(parts) if parts else "No metadata found for this connection."


def metadata_exists(conn_name: str) -> bool:
    """Return True if at least one metadata file exists for conn_name."""
    return (DATABASES_DIR / f"{conn_name}_columns.json").exists()
