"""
HablaDB — Chat-to-SQL Streamlit application.

Entry point: run with `streamlit run app.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from llm_utils import (
    PROVIDER_MODELS,
    get_litellm_model_id,
    text_to_sql,
)
from metadata_utils import (
    DATABASES_DIR,
    get_discovered_connections,
    harvest_metadata,
    load_metadata_context,
    metadata_exists,
    normalize_connection_string,
    validate_connection,
)

# Load .env so HABLADB_CONN_* and LLM API keys are available.
load_dotenv()

CONNECT_TIMEOUT = 10
ENV_PATH = Path(__file__).resolve().parent / ".env"


def _ensure_databases_dir() -> None:
    DATABASES_DIR.mkdir(parents=True, exist_ok=True)


def _all_connections() -> dict[str, str]:
    """Union of env-discovered and session-added connections."""
    discovered = get_discovered_connections()
    extra = st.session_state.get("extra_connections") or {}
    return {**discovered, **extra}


def _persist_connection_to_env(name: str, connection_string: str) -> bool:
    """Append HABLADB_CONN_{name}=... to .env. Returns True on success."""
    try:
        key = f"HABLADB_CONN_{name}"
        line = f'{key}="{connection_string}"\n'
        with open(ENV_PATH, "a", encoding="utf-8") as f:
            f.write(line)
        os.environ[key] = connection_string
        extra = st.session_state.get("extra_connections") or {}
        extra[name] = connection_string
        st.session_state["extra_connections"] = extra
        return True
    except OSError:
        return False


def _execute_sql(connection_string: str, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """
    Execute the given SQL and return (dataframe, error_message).
    On success error_message is None. Handles timeouts (connect_timeout),
    invalid SQL, and other SQLAlchemy/DBAPI errors without crashing the app.
    """
    url = normalize_connection_string(connection_string)
    try:
        engine = create_engine(
            url,
            connect_args={"connect_timeout": CONNECT_TIMEOUT},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
        return pd.DataFrame(rows, columns=columns), None
    except SQLAlchemyError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


def _dialect_hint(connection_string: str) -> str:
    """Return 'Redshift' if URL looks like Redshift, else 'PostgreSQL'."""
    url = (connection_string or "").lower()
    if "redshift" in url:
        return "Redshift"
    return "PostgreSQL"


def main() -> None:
    st.set_page_config(page_title="HablaDB", page_icon="🗄️", layout="wide")
    st.title("🗄️ HablaDB — Chat to SQL")

    _ensure_databases_dir()
    conns = _all_connections()

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Settings")

        # Active connection
        conn_options = list(conns.keys()) if conns else []
        if not conn_options:
            st.warning("No connections. Add one below or set HABLADB_CONN_* in .env.")
        active_conn = st.selectbox(
            "Active connection",
            options=conn_options,
            index=0 if conn_options else None,
            key="active_connection",
        )

        # LLM provider & model
        provider = st.selectbox(
            "LLM provider",
            options=list(PROVIDER_MODELS.keys()),
            key="llm_provider",
        )
        models = PROVIDER_MODELS.get(provider, [])
        model_name = st.selectbox(
            "Model",
            options=models,
            index=0 if models else None,
            key="llm_model",
        )

        st.divider()
        st.subheader("Connection management")

        with st.form("new_connection_form"):
            new_name = st.text_input("Connection name", placeholder="e.g. mydb")
            new_url = st.text_input(
                "Connection string",
                placeholder="postgresql://user:pass@host:5432/db (postgres:// also accepted)",
                type="password",
            )
            submitted = st.form_submit_button("Add connection")
            if submitted:
                if not new_name or not new_url:
                    st.error("Name and connection string are required.")
                else:
                    ok, msg = validate_connection(new_url)
                    if not ok:
                        st.error(f"Validation failed: {msg}")
                    else:
                        url_to_save = normalize_connection_string(new_url.strip())
                        if _persist_connection_to_env(new_name.strip(), url_to_save):
                            st.success(f"Connection {new_name!r} saved and loaded.")
                            st.rerun()
                        else:
                            st.error("Could not write to .env.")

        if active_conn:
            st.divider()
            if metadata_exists(active_conn):
                st.caption(f"Metadata for {active_conn} is cached.")
            if st.button("Harvest metadata", key="harvest_metadata"):
                url = conns.get(active_conn)
                if url:
                    with st.spinner("Harvesting metadata…"):
                        try:
                            harvest_metadata(url, active_conn)
                            st.success("Metadata harvested.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.error("Connection URL not found.")

    # ----- Chat -----
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("Generated SQL"):
                    st.code(msg["sql"], language="sql")
            if msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)
            if msg.get("error"):
                st.error(msg["error"])

    if prompt := st.chat_input("Ask a question about your data"):
        if not active_conn:
            st.error("Select an active connection in the sidebar.")
            st.stop()
        if not model_name:
            st.error("Select an LLM model in the sidebar.")
            st.stop()

        conn_url = conns.get(active_conn)
        if not conn_url:
            st.error("Active connection URL not found.")
            st.stop()

        # Append user message
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            metadata_context = load_metadata_context(active_conn)
            model_id = get_litellm_model_id(provider, model_name)
            dialect = _dialect_hint(conn_url)

            sql, llm_error = text_to_sql(prompt, metadata_context, model_id, dialect_hint=dialect)

            if llm_error:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "I couldn't generate SQL for that question.",
                    "error": llm_error,
                })
                st.error(llm_error)
                st.stop()

            if not sql:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "I couldn't extract a SQL query from the model response.",
                    "error": None,
                })
                st.warning("No SQL could be extracted from the model response.")
                st.stop()

            st.markdown("Generated SQL:")
            with st.expander("SQL", expanded=True):
                st.code(sql, language="sql")

            df, exec_error = _execute_sql(conn_url, sql)
            if exec_error:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "Here is the generated SQL (execution failed).",
                    "sql": sql,
                    "error": exec_error,
                })
                st.error(f"Execution error: {exec_error}")
                st.stop()

            st.dataframe(df, use_container_width=True)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "Here are the results.",
                "sql": sql,
                "df": df,
                "error": None,
            })

    # Rerender so the new messages show in the loop above (Streamlit will show them after rerun).
    # We've already appended to session_state and rendered in the loop; the next run will display.
    # No explicit rerun needed for chat — Streamlit re-runs on input.
    return None


if __name__ == "__main__":
    main()
