# HablaDB — Chat to SQL

HablaDB lets you query **PostgreSQL**, **Redshift**, and **DuckDB** databases using natural language. A Streamlit chat UI sends your question and database metadata to an LLM (OpenAI, Anthropic, Groq, or Gemini via [LiteLLM](https://docs.litellm.ai/)); the generated SQL is shown and executed, and results are displayed in the app.

## Features

- **Connection management**: Discover connections from `HABLADB_CONN_*` environment variables; add new ones via the UI (PostgreSQL/Redshift connection string or DuckDB file path) with validation and persistence to `.env`.
- **Metadata harvesting (RAG)**: Catalog schemas, tables, columns (and descriptions where supported); store under `databases/` and inject into the LLM context for accurate SQL generation.
- **Conversational UI**: Choose active connection, LLM provider, and model in the sidebar; chat thread with expandable SQL and result dataframes; graceful error handling for invalid SQL and timeouts.

## Setup

1. **Clone and create a virtual environment**

   ```bash
   cd habladb
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure connections and LLM**

   Copy `.env.example` to `.env` and set at least:

   - One database connection, e.g.  
     `HABLADB_CONN_MYDB=postgresql://user:password@host:5432/dbname`  
     or add a **DuckDB** connection in the UI by choosing "DuckDB" and entering the path to your `.duckdb` file.
   - For Redshift use the same URL form with your cluster endpoint and port (e.g. 5439).
   - The API key for your chosen LLM provider:  
     `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY`.

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Sidebar**
   - **Active connection**: Select the database to query (from `.env` or added in the UI).
   - **LLM provider / Model**: Choose provider and model for text-to-SQL.
   - **Add connection**: Name + either a **PostgreSQL/Redshift** connection string or **DuckDB** path to a `.duckdb` file; validated before saving to `.env`.
   - **Harvest metadata**: After selecting a connection, click to refresh schema/table/column metadata into `databases/{conn_name}_*.json`. Do this at least once per connection (and after schema changes) for good SQL generation.

2. **Chat**
   - Type a natural-language question (e.g. “How many users signed up last month?”).
   - The app loads metadata for the active connection, calls the LLM, shows the generated SQL in an expandable block, runs it, and displays the result in a dataframe. Errors (e.g. invalid SQL or timeouts) are shown in the thread.

## Project layout

| File / directory      | Purpose |
|-----------------------|--------|
| `app.py`              | Streamlit UI: sidebar, connection form, chat, execution and display. |
| `metadata_utils.py`   | DB crawling (reflection), validation, and read/write of `databases/*.json`. |
| `llm_utils.py`        | System prompt construction and LiteLLM text-to-SQL. |
| `databases/`          | Generated `{conn_name}_schemas.json`, `_tables.json`, `_columns.json`. |
| `.env`                | Connection strings and API keys (git-ignored). |

## Requirements

- Python 3.10+
- Streamlit, SQLAlchemy, psycopg2-binary, LiteLLM, python-dotenv, pandas (see `requirements.txt`).

## Security and robustness

- Connection strings are stored in `.env` (do not commit). Validation is done with `create_engine().connect()` before saving.
- DB connection and reflection use timeouts; invalid SQL or DB errors are caught and shown in the UI instead of crashing the app.
