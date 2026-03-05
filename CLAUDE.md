# pHera — Medical RAG Platform

RAG platform for women's vaginal health. pH readings + health profiles → evidence-based insights from curated medical research papers with inline citations.

## Tech Stack (Non-Obvious Details)

- **LLM generation**: LangChain `AzureChatOpenAI` (GPT-4o, temperature 0.0 — intentionally deterministic for medical safety)
- **Retrieval**: LlamaIndex retriever over pgvector with hybrid search (NOT CitationQueryEngine — citations are custom-built in `agents/utils.py`)
- **Embeddings**: Azure OpenAI `text-embedding-3-large` (3072 dims)
- **Orchestration**: LangGraph (2-node linear graph: `retrieve_node` → `generate_node`)
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (called inside `retrieve_node`, not a separate graph node)
- **Query generation/rewriting**: Happens pre-graph in the FastAPI route layer (`routes/query.py`), not in the LangGraph workflow

## Key Architectural Decisions

- The actual pgvector table is `data_paper_chunks` (LlamaIndex auto-prefixes `data_` to the `paper_chunks` table name)
- Retrieval uses alpha=0.3 (70% BM25 keyword, 30% vector semantic) — BM25-heavy is intentional for medical terminology precision
- Over-retrieval strategy: 15 candidates → cross-encoder reranks → top 5 kept. The `15` is currently hardcoded in `nodes.py`, will be moved to config's `vector_similarity_top_k`
- Structured output: generation uses Pydantic `MedicalResponse` with `response` + `used_citations` fields
- The Streamlit app is a separate HTTP client — it calls the FastAPI backend at `localhost:8000`, it does NOT invoke the graph directly

## What's NOT Implemented Yet

Do not assume these work — they are planned, not built:

- **Risk assessment** (NORMAL/MONITOR/CONCERNING/URGENT): pH thresholds exist in config but NO classification logic exists in any route or node. `QueryResponse` has NO `risk_level` field
- **Authentication**: Zitadel placeholder exists in the User model but is not integrated
- **User data persistence**: In-memory only. `QueryLog` model exists but is never written to
- **Agentic query rewriting**: Currently pre-graph helper functions in `routes/query.py`. Planned as a LangGraph node for future agentic RAG

## Common Commands

```bash
# API server
uvicorn medical_agent.api.main:app --reload

# Streamlit demo (requires API running)
streamlit run streamlit_app.py

# Tests
pytest

# Lint & format
ruff check src/
ruff format src/

# Type checking
mypy src/medical_agent

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Paper ingestion from GCP
python scripts/ingest_papers.py

# Evaluation
python -m medical_agent.evaluation.generate_testset --size 20 --limit 200
python -m medical_agent.evaluation.run_evaluation --testset <path_to_csv>
```

## Project Layout Gotchas

- `src/` layout: all imports use `medical_agent.*` namespace. Outside venv: `PYTHONPATH=src python scripts/your_script.py`
- Evaluation data lives in `src/medical_agent/evaluation/testsets/` and `results/` — there is no `data/golden_set/` directory (README reference is outdated)
- `core/__init__.py` exports `PaperManager` and deletion utilities from `core/paper_manager.py`
- Async throughout: asyncpg driver, async SQLAlchemy sessions, async FastAPI routes

## Rules

IMPORTANT: Do not make any modification in the code until and unless I tell you to.
