"""
LLM integration for HablaDB text-to-SQL.

Uses LiteLLM for multi-provider support (OpenAI, Anthropic, Groq, Gemini).
Builds a system prompt from database metadata and returns generated SQL.
"""

from __future__ import annotations

import re
from typing import Optional

import litellm


# Provider prefix -> display name and example models for the UI.
PROVIDER_MODELS = {
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
    "gemini": [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
    ],
}


def get_litellm_model_id(provider: str, model: str) -> str:
    """
    Return the model string to pass to litellm (e.g. openai/gpt-4o).
    LiteLLM expects provider/model for multi-provider.
    """
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if not model:
        return ""
    if provider in ("openai", "anthropic", "groq"):
        return f"{provider}/{model}"
    if provider == "gemini":
        if not model.startswith("gemini/"):
            return f"gemini/{model}"
        return model
    return model


def build_system_prompt(metadata_context: str, dialect_hint: str = "PostgreSQL") -> str:
    """
    Build the system prompt that instructs the LLM to generate SQL
    given the provided database metadata. dialect_hint can be "PostgreSQL"
    or "Redshift" for minor dialect differences.
    """
    return f"""You are a SQL expert. Given the following database metadata, generate a single, correct {dialect_hint}-compatible SQL query that answers the user's question.

Rules:
- Output ONLY the SQL query, no explanation before or after.
- If you need to clarify or the question is ambiguous, still output a best-guess query and a one-line comment starting with -- NOTE: ...
- Use only schema, table, and column names that appear in the metadata below.
- Do not use semicolons inside the query unless it is thx§e final statement terminator.
- Do not ever use a query is not select only, it must be a select statement.
- Do not execute DML statements, only select statements.
- Do not execute DDL statements, only select statements.

Database metadata:
{metadata_context}
"""


def extract_sql_from_response(content: str) -> Optional[str]:
    """
    Extract a single SQL query from the LLM response. Handles markdown code blocks
    (```sql ... ``` or ``` ... ```) and raw SQL. Returns None if nothing found.
    """
    if not content or not content.strip():
        return None
    text = content.strip()

    # Prefer ```sql ... ``` or ``` ... ```
    block = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if block:
        return block.group(1).strip() or None

    # Single line that looks like SELECT/WITH/INSERT/UPDATE/DELETE
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("--"):
            continue
        if re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\s", line, re.IGNORECASE):
            return text
    return text


def text_to_sql(
    user_question: str,
    metadata_context: str,
    model_id: str,
    dialect_hint: str = "PostgreSQL",
) -> tuple[str | None, Optional[str]]:
    """
    Call the LLM to generate SQL from the user question and metadata context.
    Returns (generated_sql, error_message). On success error_message is None.
    """
    if not model_id:
        return None, "No model selected."
    if not metadata_context or metadata_context.strip() == "No metadata found for this connection.":
        return None, "No database metadata available. Harvest metadata for the active connection first."

    system = build_system_prompt(metadata_context, dialect_hint=dialect_hint)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question},
    ]

    try:
        response = litellm.completion(
            model=model_id,
            messages=messages,
        )
        content = (response.choices or [{}])[0].get("message", {}).get("content") or ""
        sql = extract_sql_from_response(content)
        return sql, None
    except Exception as e:
        return None, str(e)
