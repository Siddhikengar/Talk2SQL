import os
import re
import sqlite3
import subprocess

import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Config
DATA_DIR = os.environ.get("DATA_DIR", "data")
BASE_MODEL = os.environ.get("OLLAMA_BASE_MODEL", "llama3.1")
CUSTOM_MODEL = os.environ.get("OLLAMA_MODEL", "sql-assistant")

SQL_SYSTEM_PROMPT = (
    "You are a SQL generation assistant. Given a table schema and a question "
    "in plain English, respond with ONLY a single-line SQL query that answers "
    "the question. Do not include explanations, markdown formatting, code "
    "fences, or any text other than the SQL query itself."
)

EXPLAIN_SYSTEM_PROMPT = (
    "You are a helpful data analyst. Given a user's original question, the "
    "SQL query that was run, and the resulting data, explain the result in "
    "clear, plain, everyday language a non-technical person could understand. "
    "Summarize what the data shows, call out any notable numbers or patterns, "
    "and keep it concise (a few sentences). Do not mention SQL syntax or "
    "restate the query itself."
)

# ---------------------------------------------------------------------------
# Demo datasets 
DEMO_DATASETS = {
    "Sales": {
        "tables": {
            "customers": "sales_customers.csv",
            "products": "sales_products.csv",
            "orders": "sales_orders.csv",
        },
    },
    "Students": {
        "tables": {
            "students": "students_students.csv",
            "courses": "students_courses.csv",
         
        },
    },
    "Hospital": {
        "tables": {
            "patients": "hospital_patients.csv",
            "doctors": "hospital_doctors.csv",
            "appointments": "hospital_appointments.csv",
        },
    },
}


def mem_db_uri(mem_name):
    return f"file:{mem_name}?mode=memory&cache=shared"


def seed_database(dataset_name):
    config = DEMO_DATASETS[dataset_name]
    mem_name = f"memdb_{dataset_name}"
    seeded_key = f"seeded_{mem_name}"

    st.session_state.setdefault("_mem_keepalive_conns", {})

    if st.session_state.get(seeded_key):
        return mem_name

    conn = sqlite3.connect(mem_db_uri(mem_name), uri=True)
    # Keep this connection open for the life of the session so the
    # shared-cache in-memory database isn't dropped.
    st.session_state["_mem_keepalive_conns"][mem_name] = conn

    missing = []
    for table_name, csv_filename in config["tables"].items():
        csv_path = os.path.join(DATA_DIR, csv_filename)
        if not os.path.exists(csv_path):
            missing.append(csv_path)
            continue
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.commit()

    if missing:
        st.error("Missing CSV file(s): " + ", ".join(missing))

    st.session_state[seeded_key] = True
    return mem_name


# ---------------------------------------------------------------------------
# Ollama model setup

def ensure_ollama_model():
    try:
        existing = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if CUSTOM_MODEL in existing.stdout:
            return True, None
        modelfile_content = f"FROM {BASE_MODEL}\nPARAMETER temperature 0\nSYSTEM \"\"\"{SQL_SYSTEM_PROMPT}\"\"\"\n"
        modelfile_path = "._sql_assistant_modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)
        result = subprocess.run(
            ["ollama", "create", CUSTOM_MODEL, "-f", modelfile_path],
            capture_output=True, text=True, timeout=300,
        )
        os.remove(modelfile_path)
        if result.returncode != 0:
            return False, result.stderr
        return True, None
    except FileNotFoundError:
        return False, "The `ollama` command was not found. Install it from https://ollama.com/download"
    except Exception as e:
        return False, str(e)


def build_chain(db, model_name):
    template = "Table Schema:\n{schema}\n\nQuestion: {question}\nSQL Query:\n"
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_name, temperature=0)
    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )


def build_explain_chain(model_name):
    template = (
        f"{EXPLAIN_SYSTEM_PROMPT}\n\n"
        "Original question: {question}\n"
        "SQL query: {sql}\n"
        "Result data (as rows):\n{data}\n\n"
        "Plain-language explanation:\n"
    )
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model=model_name, temperature=0)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# App

st.set_page_config(page_title="Talk2SQL",layout="wide")

if "model_ready" not in st.session_state:
    with st.spinner("Setting up the local model (first run only, may take a minute)..."):
        ok, err = ensure_ollama_model()
    st.session_state.model_ready = ok
    st.session_state.model_error = err
    st.session_state.active_model = CUSTOM_MODEL if ok else BASE_MODEL

if not st.session_state.model_ready:
    st.warning(
        f"Couldn't set up the custom '{CUSTOM_MODEL}' model ({st.session_state.model_error}). "
        f"Falling back to '{BASE_MODEL}' directly — make sure you've run `ollama pull {BASE_MODEL}`."
    )

for key, default in {
    "databases": None,
    "tables": None,
    "columns": None,
    "generated_sql": "",
    "sql_error": None,
    "last_question": "",
    "query_columns": None,
    "query_rows": None,
    "explanation": None,
    "explain_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — dataset picker + table/column browser 
with st.sidebar:
    if st.button("🗄️ Show Databases", use_container_width=True):
        st.session_state.databases = list(DEMO_DATASETS.keys())

    if st.session_state.databases:
        st.markdown("**Databases:**")
        st.json(st.session_state.databases)

    demo_choice = st.selectbox("Dataset:", list(DEMO_DATASETS.keys()), label_visibility="collapsed")

    if st.session_state.get("loaded_domain") != demo_choice:
        path = seed_database(demo_choice)
        st.session_state.current_db = path
        st.session_state.loaded_domain = demo_choice
        st.session_state.tables = None
        st.session_state.columns = None

    if "current_db" not in st.session_state:
        st.session_state.current_db = None

    st.divider()
    if st.session_state.current_db:
        if st.button("📁 Show Tables", use_container_width=True):
            conn = sqlite3.connect(mem_db_uri(st.session_state.current_db), uri=True)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            st.session_state.tables = [row[0] for row in cur.fetchall()]
            conn.close()

        if st.session_state.tables:
            st.markdown("**Tables:**")
            st.json(st.session_state.tables)

        table_name_lookup = st.text_input("Enter Table Name:", value="")
        if st.button("📋 Show Columns", use_container_width=True):
            try:
                conn = sqlite3.connect(mem_db_uri(st.session_state.current_db), uri=True)
                cur = conn.cursor()
                cur.execute(f"PRAGMA table_info({table_name_lookup})")
                st.session_state.columns = [row[1] for row in cur.fetchall()]
                conn.close()
            except Exception as e:
                st.error(f"Could not list columns: {e}")

        if st.session_state.columns:
            st.markdown("**Columns:**")
            st.json(st.session_state.columns)

# --- Generate + execute SQL --------------------------------------------------
if not st.session_state.current_db:
    st.info("Load a demo dataset or create a table above to get started.")
else:
    st.markdown("## 🤖 AI-Powered SQL Query Generation")
    nl_question = st.text_area(
        "Enter your query in plain English:", height=100,
        placeholder="e.g. Show me all records where age is over 30",
    )

    if st.button("⚡ Generate SQL"):
        if not nl_question.strip():
            st.warning("Type a question first.")
        else:
            try:
                db = SQLDatabase.from_uri(
                    f"sqlite:///{mem_db_uri(st.session_state.current_db)}&uri=true",
                    sample_rows_in_table_info=1,
                )
                chain = build_chain(db, st.session_state.active_model)
                raw_response = chain.invoke({"question": nl_question})
                match = re.search(r"```sql\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE)
                st.session_state.generated_sql = match.group(1).strip() if match else raw_response.strip()
                st.session_state.sql_error = None
                st.session_state.last_question = nl_question.strip()
            except Exception as e:
                st.session_state.sql_error = f"SQL generation failed: {e}"

    if st.session_state.sql_error:
        st.error(st.session_state.sql_error)

    st.markdown("## 🔍 Execute SQL Query")
    sql_query = st.text_area(
        "Enter SQL query to execute:", value=st.session_state.generated_sql, height=100, key="sql_query_box"
    )

    exec_col, explain_col = st.columns(2)

    with exec_col:
        run_clicked = st.button("▶️ Execute Query", use_container_width=True)
    with explain_col:
        explain_clicked = st.button(
            "🗣️ Explain Results", use_container_width=True,
            help="Explains the last executed results in plain language — does not re-run the query.",
        )

    if run_clicked:
        if not sql_query.strip():
            st.warning("Enter or generate a SQL query first.")
        else:
            try:
                conn = sqlite3.connect(mem_db_uri(st.session_state.current_db), uri=True)
                cur = conn.cursor()
                cur.execute(sql_query)
                if cur.description:
                    columns = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                    st.session_state.query_columns = columns
                    st.session_state.query_rows = rows
                else:
                    conn.commit()
                    st.session_state.query_columns = None
                    st.session_state.query_rows = None
                    st.success(f"Query executed. Rows affected: {cur.rowcount}")
                conn.close()
                # A fresh run invalidates any previous explanation.
                st.session_state.explanation = None
                st.session_state.explain_error = None
            except Exception as e:
                st.error(f"Query execution failed: {e}")

    # Always show the most recently fetched results, if any, so they persist
    # across reruns (e.g. when the Explain button is clicked afterwards).
    if st.session_state.query_columns is not None and st.session_state.query_rows is not None:
        st.dataframe(
            [dict(zip(st.session_state.query_columns, row)) for row in st.session_state.query_rows],
            use_container_width=True,
        )

    if explain_clicked:
        if st.session_state.query_columns is None or st.session_state.query_rows is None:
            st.warning("Execute a query first — there are no results to explain yet.")
        else:
            try:
                data_preview = pd.DataFrame(
                    st.session_state.query_rows, columns=st.session_state.query_columns
                ).head(50).to_csv(index=False)
                explain_chain = build_explain_chain(st.session_state.active_model)
                explanation = explain_chain.invoke({
                    "question": st.session_state.last_question or "(no question was recorded)",
                    "sql": sql_query,
                    "data": data_preview,
                })
                st.session_state.explanation = explanation.strip()
                st.session_state.explain_error = None
            except Exception as e:
                st.session_state.explain_error = f"Explanation failed: {e}"

    if st.session_state.explain_error:
        st.error(st.session_state.explain_error)

    if st.session_state.explanation:
        st.markdown("### 💬 In Plain Language")
        st.write(st.session_state.explanation)