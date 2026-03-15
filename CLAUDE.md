# pHera — Medical RAG Platform

RAG platform for women's vaginal health. pH readings + health profiles → evidence-based insights from curated medical research papers with inline citations.

## Tech Stack (Non-Obvious Details)

- **LLM generation**: LangChain `AzureChatOpenAI` (GPT-4o, temperature 0.0 — intentionally deterministic for medical safety)
- **Retrieval**: LlamaIndex retriever over pgvector with hybrid search (NOT CitationQueryEngine — citations are custom-built in `agents/utils.py`)
- **Embeddings**: Azure OpenAI `text-embedding-3-large` (3072 dims)
- **Orchestration**: LangGraph (2-node linear graph: `retrieve_node` → `generate_node`)
- **Reranking**: ✅ DONE — `cross-encoder/ms-marco-MiniLM-L-6-v2` (called inside `retrieve_node`, over-retrieves 15 → reranks to top 5)
- **Query generation/rewriting**: ✅ DONE — Pre-graph in FastAPI route layer (`routes/query.py`); generates initial queries from health profile + rewrites follow-ups with conversation history
- **Prompt tuning**: ✅ DONE — Comprehensive system prompt with explicit rules for faithfulness, relevance, citations, and patient context integration (see `nodes.py:138-174`)

## Key Architectural Decisions

- The actual pgvector table is `data_paper_chunks` (LlamaIndex auto-prefixes `data_` to the `paper_chunks` table name)
- Retrieval uses alpha=0.3 (70% BM25 keyword, 30% vector semantic) — BM25-heavy is intentional for medical terminology precision
- Over-retrieval strategy: 15 candidates → cross-encoder reranks → top 5 kept. The `15` is currently hardcoded in `nodes.py`, will be moved to config's `vector_similarity_top_k`
- Structured output: generation uses Pydantic `MedicalResponse` with `response` + `used_citations` fields
- The Streamlit app is a separate HTTP client — it calls the FastAPI backend at `localhost:8000`, it does NOT invoke the graph directly

## What's NOT Implemented Yet

Do not assume these work — they are planned, not built:

- **Risk assessment** (NORMAL/MONITOR/CONCERNING/URGENT): NOT NEEDED. pH thresholds exist in config but classification logic was intentionally not built. Deprioritized in favor of agentic reasoning.
- **Authentication**: Zitadel placeholder exists in the User model but is not integrated.
- **User data persistence**: In-memory only. `QueryLog` model exists but is never written to.
- **Agentic loop**: Linear graph without conditional reasoning node (see Roadmap below).

## Roadmap: Three-Phase Agentic RAG Enhancement

Based on 2025-2026 medical RAG best practices, implementing iterative retrieval with confidence-based validation.

### Phase 1: Metadata-Weighted Reranking ✅ DONE
**Goal**: Incorporate extracted medical metadata into relevance scoring.

**Implementation**: Blend cross-encoder (70%) + metadata overlap (30%)
- `final_score = (0.7 × reranker_score) + (0.3 × metadata_overlap_score)`
- Metadata overlap: check if paper's diagnoses/symptoms match user's health profile

**Files modified**:
- `agents/reranker.py` — added `compute_metadata_overlap_score()` and `rerank_nodes_with_metadata()`
- `agents/nodes.py:retrieve_node` — uses metadata weighting

**Outcome**: Better retrieval specificity aligned with patient context.

---

### Phase 2: Reasoning Node with Confidence Validation ✅ DONE
**Goal**: Single-pass validation of retrieval quality using hybrid confidence (score + conditional LLM).

**Implementation**:
- Score-based confidence: average of top-5 relevance scores
- Hybrid confidence: (0.6 × score) + (0.4 × LLM validation) when 0.4 ≤ score < 0.8
- Output: `confidence_score` (0.0-1.0), `retrieval_quality` (high/low), `confidence_method`

**Files created/modified**:
- `agents/reasoning.py` — new file with `reasoning_node()`, `compute_score_based_confidence()`, `validate_with_llm()`
- `agents/graph.py` — added reasoning node: `retrieve → reasoning → generate`
- `agents/state.py` — added 3 state fields

**Outcome**: Confidence-aware retrieval assessment. LLM validates only when uncertain (saves 60-70% of LLM calls).

---

### Phase 3: Agentic Loop with Iterative Retrieval & Adaptive Prompting ← NEXT
**Goal**: Enable multi-round retrieval for low-confidence queries + confidence-adaptive response generation.

**Implementation**:
- **Agentic Loop**: If confidence < 0.7, refine query semantically and retry (max 2 retries)
  - LLM asks: "What's missing from these docs?"
  - Generate refined query targeting gaps
  - Check query diversity with embedding similarity (>10% different)

- **Confidence-Adaptive Prompting**: Three prompt tiers based on confidence
  - **HIGH (≥0.75)**: Direct, confident tone. "Research demonstrates X causes Y"
  - **MEDIUM (0.50-0.75)**: Exploratory, measured tone. "May indicate, suggests, could relate to"
  - **LOW (<0.50)**: Humble, cautious tone. "Limited information available. Please consult provider."

**Files to create/modify**:
- `agents/refine_query.py` — new file with `refine_query_node()`, query refinement + similarity check
- `agents/prompts.py` — new file with 3 system prompts (high/medium/low confidence tiers)
- `agents/graph.py` — add conditional edges for retry routing
- `agents/nodes.py:generate_node` — update with 3-tier prompt selection
- `agents/state.py` — add `retry_count`, `refinement_history`

**Guard rails**:
- Max 2 retries (3 total retrieval attempts)
- Semantic query validation (embedding similarity <90%)
- Fallback: use best-effort results if retries exhausted
- All answers from knowledge base only (curated papers)

**Outcome**: Agentic RAG with self-correcting loops + medical-safe confidence-adaptive generation.

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
