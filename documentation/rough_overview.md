# pHera Medical RAG Agent — Full Technical Breakdown

## 1. ARCHITECTURE & DESIGN

### System Flow (End-to-End)

```
Mobile App / Streamlit UI
        │
        ▼
  FastAPI (/api/v1/query)
        │
        ├─ Initial: LLM generates search query from structured form data (gpt-4o-mini)
        ├─ Follow-up: LLM rewrites query with conversation history (gpt-4o-mini)
        │
        ▼
  LangGraph State Machine (START → retrieve → generate → END)
        │
        ├─ retrieve_node:
        │     ├─ LlamaIndex hybrid retrieval (BM25 + vector, alpha=0.3) → 15 candidates
        │     └─ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top 5
        │
        └─ generate_node:
              ├─ Azure OpenAI gpt-4o with structured output (MedicalResponse)
              ├─ System prompt enforces grounding, citation, and relevance rules
              └─ Returns response + used_citations → FastAPI → Client
```

- **Two-phase query strategy**: Initial queries are generated from structured health form data (pH, symptoms, diagnoses); follow-ups are rewritten with conversation context via `rewrite_followup_query`. Both use gpt-4o-mini to keep latency/cost low on the query formulation step.
- **Conversation memory**: LangGraph `MemorySaver` provides multi-turn memory keyed by `session_id` / `thread_id`. State snapshot is loaded for follow-ups to feed history into the rewriter.
- **Ingestion pipeline** (separate offline process): PDF → Docling (OCR + table extraction) → HybridChunker (1024 tokens, section-aware) → chunk filter → table-to-NL → contextual headers → metadata extraction → embedding → pgvector.

### LLM Integration

| Purpose | Model | Framework | Temperature |
|---|---|---|---|
| Response generation | `gpt-4o` (Azure) | LangChain `AzureChatOpenAI` + `.with_structured_output()` | 0.0 |
| Query generation/rewriting | `gpt-4o-mini` (Azure) | LangChain `AzureChatOpenAI` | 0.0 |
| Contextual headers (ingestion) | `gpt-4o-mini` (Azure) | LangChain `AzureChatOpenAI` | 0.0 |
| Table-to-NL transform (ingestion) | `gpt-4o` (Azure) | LangChain `AzureChatOpenAI` | 0.0 |
| Metadata extraction (ingestion) | `gpt-4o` (Azure) | LlamaIndex `AzureOpenAI` + `astructured_predict` | 0.0 |
| Embeddings | `text-embedding-3-large` (3072-dim, Azure) | LlamaIndex `AzureOpenAIEmbedding` | N/A |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `sentence-transformers` `CrossEncoder` | N/A |

### Vector DB / RAG Setup

- **Database**: PostgreSQL + pgvector extension, async via `asyncpg` + SQLAlchemy 2.0
- **Table**: `data_paper_chunks` — stores text, JSONB metadata, and `VECTOR(3072)` embeddings
- **Hybrid search**: BM25 + vector fusion via LlamaIndex's `PGVectorStore(hybrid_search=True)` with `alpha=0.3` (70% BM25 / 30% semantic) — BM25-heavy is deliberate for medical terminology precision
- **Retrieval**: over-retrieve 15 candidates → cross-encoder rerank → keep top 5
- **Storage**: PDFs in GCP Cloud Storage, metadata/chunks in PostgreSQL

### Agent Orchestration

The agent uses LangGraph with a **linear two-node graph** (no conditional routing):

```python
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
```

This is intentionally simple — no tool-calling loops, no self-reflection. Pure function nodes that accept state and return state updates. The `MedicalAgentState` TypedDict carries messages, pH value, health profile, retrieved docs, and citation tracking through the graph.

---

## 2. KEY TECHNICAL DECISIONS

### Why Azure OpenAI gpt-4o?

- **Enterprise compliance**: Azure's data residency and BAA (Business Associate Agreement) support is critical for medical/health data. Standard OpenAI API would raise HIPAA/GDPR concerns.
- **Structured output**: `gpt-4o` with `.with_structured_output(MedicalResponse)` guarantees a parseable response with `used_citations` — no regex parsing of freeform text.
- **Cost optimization**: gpt-4o-mini handles query generation and contextual headers (high volume, low complexity tasks), while gpt-4o handles response generation and table transformation (high accuracy requirements).

### Why Hybrid Search (BM25-Heavy)?

- Medical terminology is precise — "bacterial vaginosis" should exact-match, not just semantically approximate. The `alpha=0.3` (70% BM25) reflects this.
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) provides a second relevance pass that's model-based but much cheaper than LLM-based reranking.

### Medical Accuracy / Safety

**Grounding rules in the system prompt** (from `nodes.py`):

```python
system_prompt = f"""You are a caring and knowledgeable women's health consultant. Answer the patient's question using the research documents provided below. Be warm and professional — like a thoughtful doctor explaining results.

RULES (follow strictly):

GROUNDING (for faithfulness):
- Base your ENTIRE response on the documents below. Every factual statement you make must come from a specific document — cite it with [1], [2], etc.
- Prefer the documents' own wording when describing findings. Do not paraphrase in ways that change meaning or add nuance not present in the source.
- Do NOT add medical facts, background knowledge, general explanations, or conclusions that are not explicitly stated in the documents.
- Connective phrases for readability (e.g., "According to this research...") are allowed but must NOT introduce new factual claims.

RELEVANCE (for answering the question):
- Answer exactly what the question asks. Start by directly addressing the question in your opening sentence.
- Include ONLY information from the documents that directly helps answer the question. Do not include tangential findings from the same documents.
- If the patient context is provided, you may note how a document's findings relate to it, but ONLY if the document explicitly discusses matching values or conditions.

WHEN NO ANSWER EXISTS:
- If the documents contain genuinely NO information relevant to the question, respond with: "I wasn't able to find information about that in the available research documents." and set used_citations to [].
- But if the documents DO contain relevant information — including study details, authors, locations, or methodology — use it to answer.

CITATIONS:
- Cite every factual claim with [1], [2], etc.
- In used_citations, list ONLY the citation numbers you actually referenced.
```

Additional safety measures:

- **Explicit "NOT diagnostic" disclaimers** on every response and in API documentation
- **`MedicalGuardrailError` exception class** for guardrail violations (400 status code)
- **Structured output** forces the LLM to declare which citations it actually used — enabling auditability
- **Temperature 0.0** across all LLM calls for deterministic, reproducible responses
- **No hallucination fallback**: The prompt explicitly instructs the model to say "I wasn't able to find information" rather than fabricate

### Testing & Evaluation Strategy

**RAGAS evaluation framework** with 4 metrics:

```python
metrics = [
    Faithfulness(),
    LLMContextRecall(),
    LLMContextPrecisionWithReference(),
    ResponseRelevancy(),
    # FACTUAL_CORRECTNESS_METRIC,  # Temporarily disabled
]
```

- **Synthetic test generation**: Uses RAGAS `TestsetGenerator` with 80% single-hop / 20% multi-hop distribution, generating questions from actual ingested chunks
- **Medical-domain context** for test generation: prompts RAGAS to generate questions "from the POV of a user who has provided their health profile and vaginal pH value"
- **LangSmith tracing**: All evaluation LLM calls are traced for debugging with project suffixes (`phera-agent-evaluation`, `phera-agent-testset-generation`)
- **Unit tests**: `tests/test_health.py` covers health/readiness/liveness endpoints

---

## 3. PRODUCTION READINESS

### Error Handling & Edge Cases

**Comprehensive exception hierarchy** in `exceptions.py`:

```python
class AppException(Exception):
    # Base with status_code, error_code, details, to_dict()

class ValidationError(AppException):          # 422
class NotFoundError(AppException):            # 404
class DatabaseException(AppException):        # 500
class ExternalServiceException(AppException): # 502
class RateLimitError(AppException):           # 429
class MedicalGuardrailError(AppException):    # 400
class StorageError(AppException):             # 500
class DocumentParsingError(AppException):     # 500
class LLMError(AppException):                 # 500
```

- **Generic exception handler** in `main.py` catches unhandled exceptions: in production it masks internal error details ("An unexpected error occurred"), in development it exposes them.
- **Graceful degradation**: If conversation history load fails during follow-up, falls back to raw query. If Azure OpenAI isn't configured, returns 503.
- **Duplicate detection**: Ingestion pipeline checks SHA-256 file hash and GCP path before processing.

### Reliability Measures

- **GCP Storage retries**: `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))` on upload/download operations via `tenacity`
- **Singleton caching**: Retriever and cross-encoder model are module-level singletons — loaded once, reused across requests
- **Connection pooling**: SQLAlchemy `pool_size=5, max_overflow=10, pool_pre_ping=True, pool_recycle=3600`
- **Idempotent deletion**: `PaperManager.delete_paper()` returns success if paper already deleted; GCP deletion is last (irreversible) with orphan-file tolerance
- **Ingestion error isolation**: Each paper processes independently; failure of one doesn't block others

### Scaling Considerations

- **Dockerfile** with Cloud Run deployment pattern (`ENV PORT=8000`, Cloud Run overrides)
- **Non-root user** in container for security
- **Async throughout**: FastAPI + asyncpg + async LLM calls
- **Alembic migrations**: Database schema versioned with migration scripts (5 migration files in `alembic/versions/`)
- **Docs disabled in production**: `docs_url=None, redoc_url=None` when `environment != "development"`
- **CORS locked down in production**: `allow_origins=[]` (empty) when not in development

### Monitoring / Logging

- **LangSmith observability**: Full trace of every LLM call in the RAG pipeline (retrieval, generation, query rewriting)
- **Health endpoints**: `/health` (basic), `/health/detailed` (component-level), `/health/ready` (Kubernetes readiness probe with actual DB connectivity check), `/health/live` (liveness), `/health/cloud-services` (GCP + Azure + LangSmith)
- **Structured logging**: Python `logging` throughout with `%(asctime)s - %(name)s - %(levelname)s - %(message)s` format
- **Processing time tracking**: `processing_time_ms` returned in every query response + tracked in `QueryLog` model
- **Pipeline timing**: Ingestion tracks `parse_time_ms`, `pipeline_time_ms`, `store_time_ms`, `total_time_ms` per paper

---

## 4. CODE QUALITY

### Libraries Used

| Category | Libraries |
|---|---|
| Web | FastAPI, Uvicorn, Streamlit |
| LLM Orchestration | LangGraph (state machine), LangChain (LLM wrappers) |
| Retrieval | LlamaIndex (vector store, embeddings, node parsing) |
| PDF Processing | Docling (OCR, table extraction, structured JSON) |
| Database | SQLAlchemy 2.0 (async), asyncpg, pgvector, Alembic |
| Embeddings | Azure OpenAI `text-embedding-3-large` (3072-dim) |
| Reranking | sentence-transformers (`CrossEncoder`) |
| Cloud | google-cloud-storage, azure-identity |
| Resilience | tenacity (retries) |
| Evaluation | RAGAS v0.4, LangSmith |
| Validation | Pydantic v2, pydantic-settings |

### Prompt Engineering Approach

**Three distinct prompt patterns**:

1. **Generation prompt** (grounding-focused): Enforces citation discipline with explicit RULES sections for GROUNDING, RELEVANCE, and WHEN NO ANSWER EXISTS. Patient context and conversation history are injected as structured blocks.

2. **Contextual header prompt** (retrieval-optimization):

```python
CONTEXT_PROMPT = """You are generating retrieval-optimized context for a women's health medical RAG system. This context is prepended to each chunk and used for both semantic (embedding) and keyword (BM25) search. Your goal: help this chunk match user questions about diagnoses, symptoms, treatments, and clinical findings.

<document_outline>
{doc_outline}
</document_outline>

<chunk>
{chunk_text}
</chunk>

Write 2-3 sentences (50-100 tokens) that maximize retrieval. The context will be prepended to the chunk before embedding and indexing.

REQUIREMENTS:
1. Open with: paper title + section (e.g., "From [Title]: In the Methods/Results section on...")
2. State what question this chunk helps answer (query-mirroring)
3. Include exact medical terms users search for—use these canonical terms where relevant:
   - Diagnoses: bacterial vaginosis (BV), yeast infection, endometriosis, PCOS...
   - Symptoms: vaginal pH, discharge, odor, pelvic pain...
   - Treatments: hormone therapy, HRT, antibiotics...
   - Populations: premenopausal, menopausal, pregnant, age range
4. Front-load the most searchable terms in the first sentence
5. Be specific and factual—avoid generic phrases like "discusses various topics"

Output only the context. No preamble, explanations, or markdown."""
```

This is inspired by Anthropic's contextual retrieval research (claimed 49% error reduction). Instead of raw first 8000 chars, it builds a document outline from title + abstract + section headings.

3. **Metadata extraction prompt** (structured output): Uses controlled vocabulary with canonical mappings (e.g., "candida" → "Yeast infection", "BV" → "Bacterial vaginosis") and strict "only extract if explicitly stated" rules to prevent false positive metadata.

### RAG / Embeddings Implementation Details

**8-stage ingestion pipeline** (see `pipeline.py`):

```
Parse (Docling JSON) → Chunk (HybridChunker 1024 tokens) → Extract Metadata (gpt-4o) →
Filter (remove bibliography/noise, ~40-60% removed) → Table-to-NL (gpt-4o) →
Contextual Headers (gpt-4o-mini) → Stamp Metadata → Embed (text-embedding-3-large 3072-dim) → Store (pgvector)
```

- **Chunk filtering**: Uses Docling structural metadata (headings, labels) to remove references, page headers/footers, and document indices, plus content heuristics (noise patterns, alpha ratio < 0.4, short chunks < 15 words)
- **Table transformation**: Detects table chunks via Docling's `doc_items` label, converts markdown tables to natural language sentences for better vector/BM25 searchability
- **Metadata extraction fallback chain**: abstract → methods/results → introduction → first 3 chunks. Cleans Docling unicode artifacts (fi/fl ligatures, smart quotes)

---

## 5. CHALLENGES & LESSONS (Inferred from Code)

### What Failed / Iterative Decisions

- **`FactualCorrectness` metric disabled**: Comment says "Temporarily disabled" in RAGAS evaluation — suggests it was producing unreliable scores or was too expensive to run consistently
- **Health context dilution**: Comment in `retrieve_node` says "CHANGED: Use raw query for retrieval (no health context dilution)" — earlier versions likely appended health profile to the search query, degrading retrieval precision
- **Alpha tuning**: The `alpha=0.3` (BM25-heavy) with a comment "BM25-heavy for medical terminology precision" implies they tested different ratios and found that semantic-heavy search missed exact medical terms
- **Phase 2 pipeline**: The entire ingestion pipeline is labeled "Phase 2: Enhanced" with notes about targeting "data quality at ingestion time to improve retrieval precision" — Phase 1 likely had noisier chunks and worse retrieval metrics

### Tradeoffs Made

| Decision | Tradeoff |
|---|---|
| Linear graph (no loops/routing) | Simpler to debug and reason about, but can't self-correct or retry retrieval |
| BM25-heavy hybrid search (alpha=0.3) | Better for exact medical terms, worse for paraphrased/colloquial queries |
| gpt-4o for generation, gpt-4o-mini for query gen | Cost/latency optimization — but query quality affects everything downstream |
| Contextual headers per chunk (LLM call) | Better retrieval but adds ~N LLM calls per paper during ingestion (cost) |
| Structured output (Pydantic schema) | Guaranteed parseable response but constrains response format |
| In-memory `MemorySaver` | Simple but loses conversation state on server restart (not production-persistent) |
| No authentication | Placeholder `AuthenticationError` exists but auth isn't implemented yet |

### What Would Be Different at Scale

- `MemorySaver` should be replaced with Redis or PostgreSQL-backed checkpointer for persistent multi-turn memory
- The cross-encoder reranker (200MB model) is loaded in-process — at scale, this should be a separate service or use GPU acceleration
- No rate limiting is implemented (the `RateLimitError` class exists but isn't used anywhere)
- QueryLog model exists in the schema but isn't being written to in the query endpoint

---

## 6. USER EXPERIENCE

### Edge Case Handling

- **No relevant docs found**: Returns "I wasn't able to find information about that in the available research documents." with empty citations
- **No messages in state**: Returns "No query provided." from retrieve_node
- **Azure OpenAI not configured**: Returns HTTP 503 with clear error message
- **Follow-up without history**: Falls back to raw user message if conversation history can't be loaded
- **Very short chunks during ingestion**: Skipped for contextual header generation (< 20 words)

### Medical Domain Specificity

The system is narrowly scoped to **women's vaginal health / pH analysis**:

- The `QueryRequest` schema models a real mobile health app form with fields for discharge type, vulva symptoms, odor, urinary symptoms, birth control methods, fertility journey, and hormone therapy
- pH thresholds are configured: normal (3.8-4.5), concerning (> 5.0)
- The `build_health_context()` utility creates a structured patient context from the form data for the LLM prompt
- Metadata extraction uses a controlled vocabulary of 11 diagnoses, 16 symptoms, 9 birth control types, 5 hormone therapies, and 6 fertility treatments — all specific to women's health

### Example Real Query Flow

**Input** (from `schemas.py` example):

```json
{
  "ph_value": 4.8,
  "age": 28,
  "diagnoses": ["Polycystic ovary syndrome (PCOS)", "Endometriosis"],
  "menstrual_cycle": "Irregular",
  "symptoms": { "discharge": ["Creamy"], "vulva_vagina": ["Itchy"] }
}
```

**Processing**:

1. gpt-4o-mini generates a clinical search query from this structured data
2. Hybrid retrieval (BM25 + vector) finds 15 candidate chunks across ingested research papers
3. Cross-encoder reranks to top 5 most relevant
4. gpt-4o generates a grounded response with inline citations `[1]`, `[2]`...
5. Response includes `used_citations` list for auditability
6. Streamlit UI renders citations as hover tooltips with paper title and preview

**Response structure**: Agent reply with inline citations, list of citation metadata (paper title, page, preview, relevance score), medical disclaimers, processing time in ms, and session ID for follow-up continuity.
