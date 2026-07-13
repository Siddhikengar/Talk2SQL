# Talk2SQL 🔧

**Talk2SQL** is a local-first, AI-powered SQL assistant that lets you query databases in plain English. Ask a question in natural language, get back a ready-to-run SQL query, execute it instantly, and browse the results — all from a single Streamlit interface, powered entirely by a locally-hosted LLM (no external API calls, no data leaving your machine).

---

## ✨ Features

- **Natural Language → SQL** — Type a question like *"Show me all orders over $100"* and get a working SQL query back.
- **Fully local inference** — Runs on [Ollama](https://ollama.com/) with a custom fine-tuned model, so your schema and data never leave your machine.
- **Instant query execution** — Review, edit, and run the generated SQL directly against the loaded dataset.
- **Built-in demo datasets** — Sales, Students, and Hospital databases included out of the box for quick testing.
- **Schema explorer** — Browse available databases, tables, and columns from the sidebar without writing a single query.
- **Zero-setup in-memory databases** — Demo data is seeded into SQLite's shared-cache in-memory mode, so nothing touches disk and each session starts clean.

---

## 🧠 How It Works

1. **Pick a dataset** from the sidebar (or point `DATA_DIR` at your own CSVs).
2. Talk2SQL seeds an **in-memory SQLite database** from the dataset's CSV files.
3. On first run, it provisions a **custom Ollama model** (`sql-assistant`) — a fine-tuned variant of a base model (default: `llama3.1`) — with a system prompt tuned specifically for clean, explanation-free SQL generation.
4. Ask your question. Talk2SQL passes your question plus the live table schema to the LLM via a **LangChain** pipeline and returns a single SQL query.
5. Review, tweak if needed, and hit **Execute** to see results rendered as a table.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| LLM orchestration | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-ollama`) |
| Local inference | [Ollama](https://ollama.com/) |
| Database | SQLite (in-memory, shared-cache) |
| Data handling | pandas |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally
- A pulled base model, e.g.:
  ```bash
  ollama pull llama3.1
  ```

### Installation

```bash
git clone https://github.com/<your-username>/talk2sql.git
cd talk2sql
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

On first launch, Talk2SQL will automatically create a custom Ollama model (`sql-assistant`) tuned for SQL generation. This only happens once — subsequent runs reuse the existing model.

---

## ⚙️ Configuration

Talk2SQL is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `data` | Directory containing demo dataset CSVs |
| `OLLAMA_BASE_MODEL` | `llama3.1` | Base model used to build the custom SQL model |
| `OLLAMA_MODEL` | `sql-assistant` | Name of the custom fine-tuned model |

---

## 📁 Demo Datasets

Talk2SQL ships with three ready-to-use demo domains:

- **Sales** — customers, products, orders
- **Students** — students, courses
- **Hospital** — patients, doctors, appointments

Each is loaded from CSVs in `DATA_DIR` and seeded into an isolated in-memory SQLite database per session.

---

## 🛡️ Notes on Local-Only Design

Unlike most NL-to-SQL tools that rely on cloud LLM APIs, Talk2SQL runs its language model **entirely on your machine** through Ollama. This means:

- No API keys, no usage costs, no rate limits
- Your schema and query data are never sent to a third party
- Works offline once the model is pulled

---

## 🗺️ Roadmap Ideas

- [ ] Support for uploading custom CSVs/datasets at runtime
- [ ] Query history and saved queries
- [ ] Multi-table join suggestions
- [ ] Support for additional local model backends (e.g., llama.cpp)

---

## 📄 License

MIT — feel free to use, modify, and build on Talk2SQL.
