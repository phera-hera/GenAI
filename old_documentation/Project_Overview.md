# Medical RAG Agent - Interview Preparation Guide

> **Role:** AI Engineer | **Project:** FemTech Medical RAG Agent (pHera)
> **Stack:** Python 3.11 | LangGraph | LlamaIndex | Azure OpenAI | PostgreSQL + pgvector | FastAPI | GCP

---

## Table of Contents

1. [Elevator Pitch](#1-elevator-pitch)
2. [Architecture Overview](#2-architecture-overview)
3. [Technical Design Decisions & Trade-offs](#3-technical-design-decisions--trade-offs)
4. [Component Deep Dive](#4-component-deep-dive)
5. [Prompt Engineering](#5-prompt-engineering)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Infrastructure & Deployment](#7-infrastructure--deployment)
8. [Security, Privacy & Medical Safety](#8-security-privacy--medical-safety)
9. [Known Limitations & Future Improvements](#9-known-limitations--future-improvements)
10. [Likely Interview Questions & Answers](#10-likely-interview-questions--answers)

---

## 1. Elevator Pitch

**What is it?**
A privacy-first, evidence-based health insights platform for women's vaginal health. Users provide a vaginal pH reading (from a test strip) along with their health profile (age, symptoms, diagnoses, menstrual cycle, etc.), and the system retrieves relevant peer-reviewed medical research and generates a personalized, citation-backed analysis with risk categorization.

**What makes it interesting from an AI/ML perspective?**
- **Hybrid RAG pipeline** combining BM25 keyword search + dense vector search over medical literature
- **Multi-framework architecture**: LangGraph for workflow orchestration, LlamaIndex for retrieval, LangChain for LLM interaction
- **Structured output** with Pydantic for reliable citation tracking
- **Medical domain constraints**: strict grounding in retrieved documents, no hallucination tolerance, controlled vocabularies for metadata
- **Vision-based PDF parsing** (Docling) for complex medical papers with tables, figures, and OCR

**Scale:**
- 250-500 medical research papers ingested
- 3072-dimensional embeddings (text-embedding-3-large)
- Hybrid search (BM25 + vector) over PostgreSQL + pgvector
- Sub-second retrieval, ~3-5s end-to-end response generation

---

## 2. Architecture Overview

### High-Level System Diagram

```
                         STREAMLIT UI
                              |
                         FASTAPI API
                        /api/v1/query
                              |
                    +---------+---------+
                    |   LANGGRAPH       |
                    |   WORKFLOW        |
                    |                   |
                    |  START            |
                    |    |              |
                    |  retrieve_node    |-----> LlamaIndex Retriever
                    |    |              |         (Hybrid: BM25 + Vector)
                    |  generate_node   |-----> Azure OpenAI GPT-4o
                    |    |              |         (LangChain)
                    |  END             |
                    +-------------------+
                              |
                    +---------+---------+
                    |  PostgreSQL       |
                    |  + pgvector       |
                    |  (3072-dim)       |
                    +-------------------+
```

### Two Pipelines

```
INGESTION (Offline):
  PDF --> Docling Parser --> HybridChunker (512 tokens) --> Medical Metadata Extraction (LLM)
      --> Azure OpenAI Embeddings (3072-dim) --> pgvector Storage

QUERY (Online):
  User {pH, health_profile, message} --> LangGraph --> Retrieve (Hybrid Search)
      --> Generate (GPT-4o + Structured Output) --> Response {risk_level, analysis, citations}
```

### Module Structure

```
src/medical_agent/
├── core/                    # Domain core (zero dependencies)
│   ├── config.py            # Pydantic Settings + env vars
│   ├── exceptions.py        # Hierarchical exception classes
│   └── paper_manager.py     # Paper CRUD with coordinated DB+GCP deletion
│
├── infrastructure/          # External service adapters
│   ├── azure_openai.py      # Azure OpenAI client factory
│   ├── gcp_storage.py       # GCP Cloud Storage client
│   └── database/
│       ├── base.py          # SQLAlchemy base + mixins (UUID, Timestamps)
│       ├── models.py        # ORM models (Paper, PaperChunk, User, HealthProfile)
│       └── session.py       # Async session factory + connection pool
│
├── ingestion/               # Document processing pipeline
│   ├── pipeline.py          # MedicalIngestionPipeline (parse → chunk → embed → store)
│   └── metadata.py          # LLM-based medical metadata extraction (Pydantic schema)
│
├── agents/                  # LangGraph RAG workflow
│   ├── graph.py             # StateGraph definition (retrieve → generate)
│   ├── nodes.py             # Node implementations + MedicalResponse schema
│   ├── state.py             # MedicalAgentState (TypedDict with add_messages reducer)
│   ├── llamaindex_retrieval.py  # Hybrid retriever builder (PGVector + BM25)
│   └── utils.py             # Citation formatting + health context builder
│
├── api/                     # Web layer
│   ├── main.py              # FastAPI app factory with lifespan management
│   ├── schemas.py           # Request/response Pydantic models
│   └── routes/
│       ├── query.py         # POST /api/v1/query (main endpoint)
│       └── health.py        # Health checks (liveness, readiness, cloud services)
│
└── evaluation/              # RAG quality evaluation
    ├── run_evaluation.py    # RAGAS v0.4 test runner
    ├── generate_testset.py  # Synthetic test case generation
    └── ragas_config.py      # Evaluation LLM/embedding config
```

---

## 3. Technical Design Decisions & Trade-offs

### 3.1 Why Hybrid Search (BM25 + Vector)?

**Decision:** Use `vector_store_query_mode="hybrid"` with `text_search_config="english"` in PGVectorStore.

**Reasoning:**
- Medical literature contains domain-specific terms (e.g., "Lactobacillus crispatus", "bacterial vaginosis") that benefit from **exact keyword matching** (BM25)
- General health concepts ("vaginal health", "infection risk") benefit from **semantic similarity** (vector search)
- Hybrid search fuses both result sets, giving the best of both worlds

**Trade-off:** Hybrid search requires maintaining a tsvector column alongside the embedding column, doubling index storage. We accepted this because retrieval quality is paramount in a medical application.

**Alternative considered:** Pure vector search -- rejected because medical synonyms are inconsistent (e.g., "BV" vs "bacterial vaginosis") and exact term matching catches these cases.

### 3.2 Why Multi-Framework (LangGraph + LlamaIndex + LangChain)?

**Decision:** Use three LLM frameworks instead of one.

| Framework | Used For | Why |
|-----------|----------|-----|
| **LangGraph** | Workflow orchestration (retrieve → generate) | Best-in-class stateful graph execution with checkpointing |
| **LlamaIndex** | Document ingestion, vector store, hybrid retrieval | Superior abstraction for document parsing, chunking, and pgvector integration |
| **LangChain** | LLM interaction (AzureChatOpenAI, structured output) | Best `with_structured_output()` support for Pydantic models |

**Trade-off:** Increased dependency surface and learning curve. Accepted because each framework excels in its area, and they integrate cleanly at data boundaries (LlamaIndex returns `NodeWithScore` → formatted as text → LangChain generates response).

### 3.3 Why pgvector over Pinecone/Weaviate/Qdrant?

**Decision:** Use PostgreSQL + pgvector instead of a dedicated vector database.

**Reasoning:**
- **Single database** for both relational data (papers, users) and vector data (embeddings) -- simpler ops
- **Hybrid search** built into pgvector (BM25 via tsvector + vector via IVFFlat) without a separate search engine
- **ACID transactions** across relational and vector operations (e.g., paper deletion removes both the metadata row and all chunk embeddings atomically)
- **Cost:** No additional managed service fees; GCP Cloud SQL supports pgvector natively

**Trade-off:** pgvector is slower than dedicated vector DBs at scale (>1M vectors). Acceptable for our corpus size (250-500 papers, ~10-25K chunks).

### 3.4 Why Docling over PyMuPDF/Unstructured?

**Decision:** Use Docling (IBM's vision-based PDF parser) as the primary parser.

**Reasoning:**
- Medical papers have **complex layouts**: multi-column text, tables with merged cells, figures with captions, footnotes
- Docling uses a **vision-based approach** (TableFormer) that handles table extraction far better than text-based parsers
- Supports OCR for scanned papers (common in older medical literature)
- Exports as **structured JSON** (not flat text), preserving section hierarchy, table structure, and paragraph boundaries

**Trade-off:** Slower than PyMuPDF (~5-10s per paper vs ~1s). Accepted because parsing quality directly impacts retrieval quality, and we parse offline.

### 3.5 Why Structural + Token-Bounded Chunking?

**Decision:** Use Docling's `HybridChunker` (512 max tokens, 64 min tokens, merge peers, headings as metadata).

**Reasoning:**
- **Structure-aware**: Respects document boundaries (sections, paragraphs, tables are atomic units)
- **Token-bounded**: Prevents chunks from exceeding embedding model limits
- **Heading preservation**: `heading_as_metadata=True` keeps section headings out of the embedding (avoiding "diluted" embeddings) while preserving them for retrieval filtering
- **Peer merging**: Adjacent small chunks from the same section level get merged, preventing fragmentation
- **Zero overlap**: `chunk_overlap_chars=0` because Docling's structural chunking preserves semantic boundaries naturally (unlike naive character-based splitting which needs overlap for context)

**Alternative considered:** LangChain's `RecursiveCharacterTextSplitter` -- rejected because it doesn't understand document structure and would split mid-table or mid-paragraph.

### 3.6 Why Structured Output for Citation Tracking?

**Decision:** Use `llm.with_structured_output(MedicalResponse)` to get both the response text and a machine-readable list of used citations.

```python
class MedicalResponse(BaseModel):
    response: str          # Text with inline [1][2] markers
    used_citations: list[int]  # Which citations were actually referenced
```

**Reasoning:**
- Avoids regex-based post-processing to extract citation numbers from text
- The LLM explicitly declares which citations it used, enabling the API to return only relevant citation metadata
- Pydantic validation ensures the response conforms to the expected schema

**Trade-off:** Structured output adds ~100-200ms latency compared to raw text generation. Acceptable for citation reliability.

### 3.7 Why temperature=0.0 for Medical Responses?

**Decision:** Set `temperature=0.0` for both metadata extraction and response generation.

**Reasoning:**
- **Reproducibility**: Same query + same documents should produce the same response
- **Reduced hallucination**: Lower temperature = LLM picks highest-probability tokens, reducing creative (but potentially fabricated) outputs
- **Medical domain requirement**: Consistency matters more than creativity in health information

### 3.8 Why Context-Enriched Retrieval?

**Decision:** Inject the patient's health context (pH, age, symptoms, diagnoses) into the retrieval query, not just the generation prompt.

```python
enhanced_query = f"{user_query}\n\nHealth Context:\n{health_context}"
```

**Reasoning:** By embedding patient context into the search query, the vector similarity search finds documents relevant to both the question AND the patient's specific profile. For example, a query about "vaginal discharge" from a patient with PCOS will retrieve PCOS-specific discharge studies, not generic ones.

**Trade-off:** Longer queries can sometimes "dilute" the search intent. Mitigated by keeping health context concise and structured.

---

## 4. Component Deep Dive

### 4.1 Ingestion Pipeline (`ingestion/pipeline.py`)

**Class: `MedicalIngestionPipeline`**

```
PDF bytes → Temp file → DoclingReader → [Documents]
    → DoclingNodeParser + HybridChunker → [Nodes] (512-token chunks)
    → SimplifiedMedicalMetadataExtractor → [Nodes with metadata]
    → AzureOpenAIEmbedding → [Nodes with embeddings]
    → PGVectorStore.add() → PostgreSQL
```

**Key implementation details:**

1. **Deduplication (dual strategy):**
   ```python
   existing = await session.execute(
       select(Paper).where(
           (Paper.file_hash == file_hash) | (Paper.gcp_path == gcp_path)
       )
   )
   ```
   - SHA-256 content hash catches identical files at different paths
   - GCP path match catches re-uploads with different content
   - Both checked in a single SQL query with OR

2. **Two-phase commit pattern:**
   - Paper record created with `is_processed=False`
   - Chunks stored in pgvector
   - Paper updated to `is_processed=True`
   - If crash occurs between steps, paper remains as "unprocessed" (recoverable)

3. **Stage-level tracking** via `PipelineResult`:
   ```python
   @dataclass
   class PipelineResult:
       parsed: bool = False
       chunked: bool = False
       embedded: bool = False
       stored: bool = False
       parse_time_ms: int = 0
       pipeline_time_ms: int = 0
       # ...
   ```
   Enables pinpointing exactly which stage failed.

4. **Title extraction fallback chain:**
   ```
   LLM-extracted title → Document metadata title → GCP filename stem
   ```

### 4.2 Medical Metadata Extraction (`ingestion/metadata.py`)

**Schema: `MedicalMetadata` (Pydantic BaseModel)**

14 fields across 3 categories:

| Category | Fields |
|----------|--------|
| **Paper-level** | title, publication_year, author, doi |
| **Medical terms** (8 controlled vocabularies) | ethnicities, diagnoses, symptoms, menstrual_status, birth_control, hormone_therapy, fertility_treatments, prevalence_conditions |
| **Quality** | age_mentioned, age_range, confidence (0.0-1.0) |

**Key design: Soft vocabulary constraints via prompt engineering**

Instead of Pydantic validators that would reject non-standard terms:
```
CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY MENTIONED in the text
2. DO NOT infer, extrapolate, or hallucinate
3. Use EXACT standardized terms from field descriptions

MAPPING RULES:
- "PCOS" → "Polycystic ovary syndrome (PCOS)"
- "candida" → "Yeast infection"
- "birth control pill" → "Pill"
```

**Why not hard Pydantic validation?** Hard validation would crash the extraction when the LLM outputs slight variations. Soft constraints via prompting are more robust, and the LLM handles synonym normalization at extraction time.

**Document-level caching:**
```python
doc_id = nodes[0].ref_doc_id or "unknown"
if doc_id in self._cache:
    return [self._cache[doc_id].copy() for _ in nodes]
```
All chunks from the same document share metadata. The LLM is called **once per document** (not per chunk), saving ~50x API calls for a 50-chunk paper.

**Historical context:** This module replaced 650+ lines of custom code (MedicalMetadataExtractor, MetadataLLMClient, TermNormalizer) with ~100 lines using LlamaIndex's `astructured_predict` + Pydantic schema + prompt engineering.

### 4.3 LangGraph Workflow (`agents/graph.py`)

```python
graph = StateGraph(MedicalAgentState)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("generate_node", generate_node)
graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "generate_node")
graph.add_edge("generate_node", END)

medical_rag_app = graph.compile(checkpointer=MemorySaver())
```

**State schema** (`MedicalAgentState`):
```python
class MedicalAgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-appended conversation history
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str                           # Formatted citation text from retrieval
    citations: list[dict[str, Any]]          # All retrieved citation metadata
    used_citations: list[int]                # Citations actually used by LLM
```

**Why `Annotated[list, add_messages]`?** The `add_messages` reducer appends new messages rather than replacing them. When `generate_node` returns `{"messages": [AIMessage(...)]}`, it's appended to the existing conversation. Combined with `MemorySaver`, this enables multi-turn conversation without manual message management.

**Why `TypedDict` over Pydantic?** LangGraph requires `TypedDict` because nodes return partial state updates (only the keys they modify). Pydantic models would require returning the entire state.

### 4.4 Retrieve Node (`agents/nodes.py`)

```python
def retrieve_node(state: MedicalAgentState) -> dict:
    user_query = state["messages"][-1].content
    health_context = build_health_context(ph_value, health_profile)
    enhanced_query = f"{user_query}\n\nHealth Context:\n{health_context}"

    nodes = retrieve_nodes(query=enhanced_query, similarity_top_k=2)
    docs_text, citations = format_retrieved_nodes(nodes)

    return {"docs_text": docs_text, "citations": citations}
```

**Note:** `similarity_top_k=2` (overrides the default of 5). This is a conservative choice -- fewer chunks means less noise in the context window, but risks missing relevant information.

### 4.5 Generate Node (`agents/nodes.py`)

```python
def generate_node(state: MedicalAgentState) -> dict:
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        temperature=0.0,
    )
    structured_llm = llm.with_structured_output(MedicalResponse)
    # ... builds system prompt with docs_text, health_context, conversation history ...
    result: MedicalResponse = structured_llm.invoke([HumanMessage(content=system_prompt)])

    return {
        "messages": [AIMessage(content=result.response)],
        "used_citations": result.used_citations,
    }
```

### 4.6 Hybrid Retriever (`agents/llamaindex_retrieval.py`)

```python
def build_retriever(similarity_top_k=5):
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-large",
        # ... with fallback credentials ...
    )

    vector_store = PGVectorStore.from_params(
        table_name="paper_chunks",       # LlamaIndex adds "data_" prefix → data_paper_chunks
        embed_dim=3072,
        hybrid_search=True,              # BM25 + vector fusion
        text_search_config="english",    # English stemming/tokenization
        perform_setup=False,             # Schema managed by Alembic, not LlamaIndex
    )

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model)
    return index.as_retriever(
        similarity_top_k=similarity_top_k,
        vector_store_query_mode="hybrid",
    )
```

**Note on `perform_setup=False`:** The database schema is managed by Alembic migrations, not auto-created by LlamaIndex. This is the production-grade approach.

**Note on credential fallback:**
```python
embed_api_key = settings.azure_openai_embedding_api_key or settings.azure_openai_api_key
```
Supports separate Azure OpenAI deployments for embeddings vs. chat. Common in production where embedding and LLM workloads have different rate limits.

### 4.7 Citation Formatting (`agents/utils.py`)

```python
def format_retrieved_nodes(nodes: list[NodeWithScore]) -> tuple[str, list[dict]]:
    # Output format: "[1]: [PaperTitle:p23]: Insulin resistance is a key factor..."
    docs_text_parts.append(f"[{i}]: [{title}:p{page_no}]: {chunk_text}")
    citations.append({
        "id": i, "file": title, "page": str(page_no),
        "score": round(node_score.score, 3), "preview": chunk_text[:100],
        "node_id": node.node_id,
    })
```

**Triple-fallback for title:**
```python
title = node.metadata.get("title") or Path(node.metadata.get("gcp_path", "")).stem or "Unknown Paper"
```
Uses `or` (not `.get(key, default)`) because `title` can be explicitly `None`.

**Page number extraction** from Docling's provenance metadata:
```python
page_no = doc_items[0]["prov"][0]["page_no"] if doc_items and doc_items[0].get("prov") else "N/A"
```

### 4.8 API Layer (`api/`)

**Query endpoint** (`routes/query.py`):
```python
@router.post("/api/v1/query")
async def analyze_ph(request: QueryRequest) -> QueryResponse:
    # 1. Validate Azure OpenAI is configured
    # 2. Build health_profile dict from request fields
    # 3. Invoke LangGraph workflow
    result = medical_rag_app.invoke(
        {"messages": [HumanMessage(content=request.user_message)],
         "ph_value": request.ph_value,
         "health_profile": health_profile},
        config={"configurable": {"thread_id": request.session_id}}
    )
    # 4. Determine risk level from pH value
    # 5. Filter citations to only those used by LLM
    # 6. Return structured response
```

**Risk level determination:**
```python
if ph_normal_min <= ph_value <= ph_normal_max:    # 3.8-4.5
    risk_level = "NORMAL"
elif ph_value < ph_normal_min:                      # < 3.8
    risk_level = "MONITOR"
elif ph_value <= ph_concerning_threshold:            # 4.5-5.0
    risk_level = "MONITOR"
else:                                                # > 5.0
    risk_level = "CONCERNING"
```

**Health checks** (Kubernetes-compatible):
- `GET /health` -- basic liveness
- `GET /health/detailed` -- component status
- `GET /health/ready` -- readiness probe (checks DB + Azure + GCP)
- `GET /health/live` -- liveness probe
- `GET /health/cloud-services` -- GCP, Azure OpenAI, LangSmith connectivity

**Health check pattern:** Three-tier status: `healthy` (all configured services connected), `degraded` (some services connected), `unhealthy` (none connected).

### 4.9 Configuration (`core/config.py`)

**Pydantic Settings** with `lru_cache`:
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )
    # 40+ settings across 10 categories
    # Computed fields: database_connection_string, sync_database_connection_string

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Key computed fields:**
- `database_connection_string` -- async (asyncpg) for the app
- `sync_database_connection_string` -- sync (psycopg2) for Alembic migrations

**Configuration validation methods:**
```python
settings.is_azure_openai_configured()           # LLM ready?
settings.is_azure_openai_embedding_configured()  # Embeddings ready? (with fallback)
settings.is_gcp_configured()                    # GCP ready?
settings.is_langsmith_configured()              # Observability ready?
```

### 4.10 Database Models (`infrastructure/database/`)

**Mixins:**
```python
class UUIDPrimaryKeyMixin:       # uuid-ossp server-side generation
class TimestampMixin:            # created_at, updated_at with server_default=func.now()
```

**Automatic table naming:**
```python
class Base(DeclarativeBase):
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return camel_to_snake(cls.__name__)  # PaperChunk → paper_chunk
```

**Key models:**
- `Paper` -- title, authors, journal, DOI, gcp_path, file_hash, is_processed
- `PaperChunk` -- LlamaIndex-managed (id, text, embedding, metadata_ JSONB, _node_content)
- `User` -- placeholder for Zitadel OAuth2 (not yet implemented)
- `HealthProfile` -- age, symptoms, diagnoses (FK to User)
- `QueryLog` -- audit trail (future use)

**JSONB for chunk metadata:** `PaperChunk.metadata_` stores paper_id, title, doi, and all 8 medical metadata fields as JSONB. This is flexible and queryable:
```python
PaperChunk.metadata_["paper_id"].astext == str(paper_id)
```

### 4.11 Paper Manager (`core/paper_manager.py`)

**Coordinated deletion across DB + GCP:**
```
1. Retrieve paper metadata (idempotent: returns success if already deleted)
2. Count chunks (for reporting)
3. Delete chunks from DB (reversible via rollback)
4. Delete paper record from DB (reversible via rollback)
5. Commit DB transaction
6. Delete PDF from GCP (LAST - irreversible)
```

**Key design: "Reversible operations first, irreversible last."**

If GCP deletion fails after DB commit, the paper is still removed from the DB. An orphan file in GCP is acceptable (wasted storage, not data inconsistency). The opposite (deleted from GCP but still in DB) would be worse -- the system would reference a non-existent file.

---

## 5. Prompt Engineering

### 5.1 Generation Prompt (Full Template)

```
You are a medical research assistant specialized in women's reproductive health
and vaginal pH analysis.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided medical documents below
2. Cite sources inline using the citation markers [1], [2], etc.
3. If the documents do not contain relevant information, set response to:
   "No relevant medical research found in the available documents."
   and used_citations to empty list []
4. DO NOT use external knowledge or make assumptions beyond the documents
5. Be concise and medically accurate
6. Focus on evidence-based information from peer-reviewed research
7. IMPORTANT: In used_citations, list ONLY the citation numbers you actually
   referenced in your response

PATIENT CONTEXT:
{health_context}

MEDICAL DOCUMENTS:
{docs_text}

[PREVIOUS CONVERSATION:       ← only if multi-turn]
{conversation_history}

CURRENT QUESTION:
{current_query}

Provide a clear, evidence-based answer with inline citations:
```

### 5.2 Why This Prompt Design Matters

| Instruction | Purpose |
|-------------|---------|
| "Answer ONLY using information from the provided medical documents" | **Closed-book RAG** -- enforces grounding, prevents parametric knowledge leakage |
| "If documents do not contain relevant information..." | **Graceful degradation** -- explicit fallback prevents hallucination when documents are insufficient |
| "DO NOT use external knowledge" | **Double enforcement** of closed-book constraint |
| "In used_citations, list ONLY..." | **Citation accuracy** -- prevents the LLM from listing all citations regardless of usage |
| "Be concise and medically accurate" | **Domain-appropriate tone** -- medical information should be precise, not verbose |

### 5.3 Metadata Extraction Prompt

```
You are extracting metadata from research papers for a women's health application.

CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY MENTIONED in the text
2. DO NOT infer, extrapolate, or hallucinate
3. Use EXACT standardized terms from field descriptions

MAPPING RULES:
- "PCOS" → "Polycystic ovary syndrome (PCOS)"
- "candida" → "Yeast infection"
- "birth control pill" → "Pill"
```

**Key insight:** "EXPLICITLY MENTIONED" appears 8 times across the prompt and field descriptions. In medical AI, inferred diagnoses are dangerous. The repetition is deliberate anti-hallucination engineering.

---

## 6. Evaluation Framework

### 6.1 RAGAS v0.4 Metrics

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Faithfulness** | Does the response match the retrieved context? | Prevents hallucination beyond retrieved docs |
| **Factual Correctness** | Is the response factually accurate vs. ground truth? | Medical accuracy is non-negotiable |
| **Context Precision** | Are the retrieved contexts relevant to the query? | Measures retrieval quality |
| **Context Recall** | Are all relevant contexts retrieved? | Ensures important information isn't missed |
| **Response Relevancy** | Is the response relevant to the query? | Measures end-to-end quality |

### 6.2 Evaluation Pipeline

```python
# 1. Load test set from CSV (question, ground_truth, reference_contexts)
# 2. For each test case:
#    - Invoke medical_rag_app with the test question
#    - Parse docs_text into context list
# 3. Evaluate using RAGAS metrics
# 4. Generate report with per-question + aggregate metrics
```

### 6.3 Test Set Format

```csv
question,ground_truth,reference_contexts
"What causes elevated vaginal pH?","Elevated pH is commonly caused by...","[\"Bacterial vaginosis (BV) is...\"]"
```

### 6.4 How to Talk About Evaluation in Interviews

> "We use RAGAS with 5 metrics to evaluate our RAG pipeline. Faithfulness and Factual Correctness are our primary metrics because in a medical application, generating information not grounded in the retrieved context or producing factually incorrect statements is unacceptable. Context Precision and Recall help us tune our retrieval -- we want high recall (don't miss relevant papers) without sacrificing precision (don't retrieve irrelevant noise)."

---

## 7. Infrastructure & Deployment

### 7.1 Multi-Cloud Architecture

| Cloud | Services | Why |
|-------|----------|-----|
| **Azure** | OpenAI (GPT-4o, text-embedding-3-large) | Best LLM/embedding models, enterprise compliance |
| **GCP** | Cloud Run, Cloud SQL (PostgreSQL + pgvector), Cloud Storage | Serverless compute, managed database, EU region for GDPR |

**Why multi-cloud?** Best-of-breed strategy. Azure has the best enterprise LLM offering (OpenAI models). GCP has superior managed infrastructure (Cloud Run is simpler than Azure Container Apps, Cloud SQL supports pgvector natively).

### 7.2 Docker & Deployment

```dockerfile
FROM python:3.11-slim
# System deps for Docling: libgl1-mesa-glx, poppler-utils, tesseract-ocr
# Package manager: uv (astral-sh) instead of pip -- 10-100x faster
# Non-root user: appuser (security best practice)
# Cloud Run compatible: uses $PORT environment variable
CMD ["uvicorn", "medical_agent.api.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
```

**Layer caching strategy:** `pyproject.toml` is copied first, dependencies installed, then source code is copied. This means dependency layers are cached and only rebuilt when dependencies change.

### 7.3 Database Migrations (Alembic)

6 migrations tell the architectural evolution story:
1. `001_initial_schema` -- Initial tables
2. `002_update_embedding_dimension` -- Changed embedding dimensions (likely 1536 → 3072)
3. `align_schema_with_llamaindex` -- Adapted to LlamaIndex's expected schema
4. `drop_custom_columns_use_metadata_jsonb` -- Moved from typed columns to flexible JSONB
5. `rename_table_to_data_paper_chunks` -- Match LlamaIndex's `data_` prefix convention
6. `add_text_search_tsv_for_hybrid_search` -- Added tsvector column for BM25

**Key detail:** Alembic uses the **synchronous** connection string (`settings.sync_database_connection_string`) because Alembic doesn't support async natively, while the app uses the **async** string (asyncpg).

### 7.4 Database Indexes

```sql
-- Vector similarity search (cosine distance)
CREATE INDEX ON data_paper_chunks USING ivfflat (embedding vector_cosine_ops);

-- JSONB metadata queries (e.g., filter by paper_id)
CREATE INDEX ON data_paper_chunks USING gin (metadata_);

-- Full-text search for BM25 hybrid retrieval
CREATE INDEX ON data_paper_chunks USING gin (text_search_tsv);
```

### 7.5 Operational CLI Tools

| Script | Purpose |
|--------|---------|
| `scripts/ingest_papers.py` | Interactive paper ingestion (download from GCP → parse → chunk → embed → store) |
| `scripts/paper_deletion.py` | Safe paper deletion with double-confirmation for bulk ops |
| `scripts/setup_infrastructure.py` | Infrastructure verification + setup guide (runbook-as-code) |
| `scripts/init-db.sql` | Docker initdb: enables pgvector + uuid-ossp extensions |

---

## 8. Security, Privacy & Medical Safety

### 8.1 Medical Guardrails

- **System is NOT diagnostic:** Provides informational guidance only
- **System is NOT prescriptive:** No medication/treatment recommendations
- **Medical disclaimers** on every response
- **Risk levels** (NORMAL, MONITOR, CONCERNING, URGENT) guide actionability, not medical advice
- **`MedicalGuardrailError`** exception class for safety boundary violations
- **Closed-book RAG:** LLM is strictly grounded in retrieved documents, cannot use parametric knowledge

### 8.2 Privacy by Design

- **No user data persistence** (development stage): Health profiles are in-memory only
- **Queries are NOT stored:** No logging of user questions or health information
- **Only paper data is persisted:** Research papers and their embeddings
- **GCP Storage in EU region:** GDPR compliance for European users
- **Planned:** Zitadel OAuth2 authentication, field-level encryption, audit logging

### 8.3 Exception Hierarchy

```
AppException (base)
├── ValidationError (422)
├── NotFoundError (404)
├── DatabaseException (500)
├── ExternalServiceException (502)
├── AuthenticationError (401)       ← placeholder for Zitadel
├── AuthorizationError (403)        ← placeholder
├── RateLimitError (429)            ← placeholder
├── MedicalGuardrailError (400)     ← medical safety
├── StorageError (500)
├── DocumentParsingError (500)
├── LLMError (500)
└── ObservabilityError (500)
```

Each exception includes `error_code` (machine-readable) and `to_dict()` for consistent API error responses.

---

## 9. Known Limitations & Future Improvements

### Current Limitations

| Limitation | Impact | Potential Fix |
|-----------|--------|---------------|
| Retriever rebuilt per query (no caching) | ~200ms overhead per query for DB connection + index creation | Module-level singleton or LRU-cached retriever |
| `similarity_top_k=2` (conservative) | May miss relevant documents | Increase to 5 with re-ranking step |
| No retry/error handling around LLM calls | API failures propagate to user | Add tenacity retry with exponential backoff |
| No re-ranking after retrieval | Raw retrieval order may not be optimal | Add cross-encoder re-ranker (e.g., ColBERT) |
| No hallucination detection node | Relies entirely on prompt engineering | Add a verification node that checks response against retrieved context |
| MemorySaver (in-memory checkpointer) | Conversation history lost on restart | Switch to PostgreSQL-backed checkpointer |
| No authentication | API is open to anyone | Implement Zitadel OAuth2 |
| No rate limiting | Vulnerable to abuse | Add FastAPI middleware or API gateway |
| LLM instantiated per invocation | Slight overhead | Cache the LLM client at module level |

### Planned Improvements

1. **Re-ranking:** Add a cross-encoder re-ranking step after hybrid retrieval to improve precision
2. **Guardrails node:** Add a LangGraph node before `generate_node` that filters out-of-scope or harmful queries
3. **Answer validation node:** Add a post-generation node that verifies the response is grounded in the retrieved context
4. **Streaming responses:** Use LangGraph's streaming capabilities for real-time response display
5. **Metadata-filtered retrieval:** Use the extracted medical metadata (diagnoses, symptoms) to pre-filter chunks before vector search
6. **Multi-language support:** Extend to support non-English medical literature

---

## 10. Likely Interview Questions & Answers

### Architecture & Design

**Q: Walk me through the end-to-end flow when a user submits a pH reading.**

> The user submits a pH value (e.g., 5.2), their health profile (age, symptoms, diagnoses), and optionally a question through the Streamlit UI. This hits the FastAPI `POST /api/v1/query` endpoint, which validates the input using Pydantic schemas and invokes the LangGraph workflow.
>
> The workflow has two nodes. First, the **retrieve node** takes the user's question, enriches it with their health context (pH + profile), and runs a hybrid search (BM25 + vector) against our pgvector database of medical paper chunks. It retrieves the top-2 most relevant chunks and formats them with citation markers like `[1]: [PaperTitle:p23]: text...`.
>
> Second, the **generate node** takes the retrieved context, the patient's health profile, and any conversation history, and sends it to Azure OpenAI GPT-4o with a strict medical prompt that enforces closed-book RAG (only use provided documents). The LLM returns a structured response (Pydantic model) with the analysis text containing inline `[1][2]` citations and an explicit list of which citations were used.
>
> The API layer then determines the risk level from the pH value (NORMAL: 3.8-4.5, MONITOR: outside normal, CONCERNING: >5.0), filters the citations to only those actually used by the LLM, adds a medical disclaimer, and returns the structured response.

**Q: Why did you choose LangGraph over a simple function chain?**

> Three reasons: (1) **Stateful conversation** -- LangGraph's `MemorySaver` checkpointer with the `add_messages` reducer gives us multi-turn conversation out of the box, without manually managing message history. (2) **Extensibility** -- we plan to add guardrails and answer validation nodes, and LangGraph makes it trivial to insert new nodes into the graph. (3) **Observability** -- LangGraph integrates with LangSmith for tracing, which lets us inspect exactly what each node received and produced during debugging.

**Q: Why not use a single LLM framework?**

> Each framework excels at different things. LlamaIndex has the best abstraction for document ingestion (DoclingReader + HybridChunker + PGVectorStore with hybrid search). LangChain has the best `with_structured_output()` for forcing the LLM to return Pydantic models. LangGraph is the best for stateful multi-step workflows. The frameworks integrate cleanly at data boundaries -- LlamaIndex returns NodeWithScore objects, we format them as text, and LangChain handles the LLM interaction. The alternative would be forcing one framework to do everything, resulting in workarounds and anti-patterns.

### RAG-Specific

**Q: How does your hybrid search work? Why not pure vector search?**

> Our pgvector store is configured with `hybrid_search=True` and `text_search_config="english"`. Every query runs two searches simultaneously: a dense vector search using 3072-dimensional embeddings from text-embedding-3-large, and a BM25 sparse search using PostgreSQL's built-in full-text search with English stemming. The results are fused by pgvector's built-in reciprocal rank fusion.
>
> Pure vector search struggles with domain-specific medical terms. For example, "Lactobacillus crispatus" has a very specific meaning in vaginal health, and BM25's exact keyword matching catches this reliably. Meanwhile, a query like "what causes vaginal infections" benefits from semantic similarity because the relevant papers might use different terminology (e.g., "vaginitis", "dysbiosis"). Hybrid search gives us both.

**Q: How do you handle hallucination in a medical context?**

> Multiple layers: (1) **Closed-book RAG** -- the system prompt explicitly forbids using external knowledge ("Answer ONLY using information from the provided medical documents"). (2) **Graceful degradation** -- if no relevant documents are found, the LLM is instructed to say "No relevant medical research found" rather than fabricating an answer. (3) **Structured output** -- the LLM must explicitly declare which citations it used, making it auditable. (4) **temperature=0.0** -- deterministic output reduces creative (potentially hallucinated) responses. (5) **RAGAS evaluation** -- we measure Faithfulness (is the response grounded in context?) and Factual Correctness as primary metrics.
>
> What we're planning to add: a post-generation verification node that checks the response against the retrieved context for factual consistency.

**Q: Why did you choose text-embedding-3-large (3072 dimensions) over smaller models?**

> Medical text is semantically dense -- a single sentence can contain diagnosis names, treatment protocols, statistical findings, and patient demographics. Higher-dimensional embeddings can capture more of these semantic nuances. We accepted the storage trade-off (3072 vs 1536 dims = 2x storage) because retrieval quality is the most critical factor in a medical RAG system. If we retrieve the wrong documents, the entire response is compromised. The cost of a missed relevant paper is much higher than the cost of extra storage.

**Q: How do you chunk medical papers, and why?**

> We use Docling's HybridChunker, which is a two-level strategy: structural chunking first, then token-bounded. Docling understands document structure (sections, paragraphs, tables), so it never splits a table row mid-way or breaks a paragraph at an arbitrary character position. On top of that, we enforce a 512-token maximum and 64-token minimum. Small adjacent chunks from the same section are merged (`merge_peers=True`), and section headings are stored as metadata rather than embedded (`heading_as_metadata=True`), so the embedding captures only the content semantics.
>
> We use zero overlap because Docling's structural boundaries are semantically meaningful -- unlike character-based splitters where overlap is needed to prevent context loss at boundaries.

### Infrastructure

**Q: Why PostgreSQL + pgvector instead of a dedicated vector database?**

> Three reasons: (1) **Operational simplicity** -- one database for both relational data (papers, users) and vector data (embeddings). (2) **Transactional consistency** -- when we delete a paper, both the metadata row and all chunk embeddings are deleted in a single ACID transaction. With a separate vector DB, we'd need distributed transaction coordination. (3) **Hybrid search** -- pgvector supports BM25 + vector fusion natively via tsvector, so we don't need a separate search engine like Elasticsearch.
>
> The trade-off is performance at scale -- pgvector is slower than Pinecone or Qdrant beyond ~1M vectors. But with our corpus size (250-500 papers, ~10-25K chunks), pgvector is more than sufficient, and the operational simplicity is worth it.

**Q: How would you scale this system?**

> Several axes: (1) **Read scaling** -- add read replicas for pgvector queries. (2) **Index tuning** -- switch from IVFFlat to HNSW indexes for faster approximate nearest neighbor search. (3) **Caching** -- cache the retriever (currently rebuilt per query) and consider caching frequent queries with Redis. (4) **Async everywhere** -- the current `PGVectorStore.add()` is synchronous; migrating to async would remove a bottleneck. (5) **Batch embedding** -- process multiple chunks in a single API call instead of one-by-one. (6) **Horizontal scaling** -- Cloud Run auto-scales the API layer; the database is the bottleneck, so connection pooling (PgBouncer) would be the first scaling lever.

### Medical Domain

**Q: How do you ensure the system doesn't give dangerous medical advice?**

> The system is designed with multiple safety layers. First, it's explicitly **not diagnostic** -- it provides evidence-based information from peer-reviewed research, not diagnoses. Second, the risk levels (NORMAL, MONITOR, CONCERNING, URGENT) are based on well-established pH thresholds from medical literature, and they guide the user toward seeking professional care rather than self-diagnosing. Third, every response includes a medical disclaimer. Fourth, the `MedicalGuardrailError` exception class exists for catching and rejecting requests that would violate medical safety boundaries. Fifth, the closed-book RAG design means the system can only reference published, peer-reviewed research -- it cannot generate novel medical claims.

### Code Quality

**Q: How do you test this system?**

> Three levels: (1) **Unit tests** with pytest (async mode) for individual components. (2) **RAGAS evaluation** for end-to-end RAG quality, measuring faithfulness, factual correctness, context precision, context recall, and response relevancy. (3) **Infrastructure verification** via the `setup_infrastructure.py` script, which checks connectivity to PostgreSQL, GCP, Azure OpenAI, and LangSmith. The RAGAS evaluation is the most important -- it's our regression test for retrieval and generation quality.

**Q: What would you do differently if you started this project over?**

> (1) I'd add a **re-ranking step** from the beginning -- hybrid search gets you 80% of the way, but a cross-encoder re-ranker significantly improves precision. (2) I'd use a **persistent checkpointer** (PostgreSQL-backed) instead of MemorySaver from the start, so conversation history survives restarts. (3) I'd implement **metadata-filtered retrieval** earlier -- we extract 8 medical metadata fields during ingestion but don't use them to filter at query time yet. (4) I'd cache the retriever as a module-level singleton rather than rebuilding it per query. (5) I'd add **streaming responses** for better UX on the Streamlit frontend.

---

## Quick Reference: Key Numbers

| Metric | Value |
|--------|-------|
| Embedding model | text-embedding-3-large |
| Embedding dimensions | 3,072 |
| Chunk size | 512 tokens (max), 64 tokens (min) |
| Retrieval top-k | 2 (at query time) |
| LLM | GPT-4o (Azure OpenAI) |
| Temperature | 0.0 |
| pH normal range | 3.8 - 4.5 |
| pH concerning threshold | > 5.0 |
| Medical metadata fields | 8 controlled vocabularies |
| RAGAS metrics | 5 (faithfulness, factual correctness, context precision, context recall, response relevancy) |
| Database indexes | 3 (IVFFlat for vectors, GIN for JSONB, GIN for tsvector) |
| Corpus size | 250-500 papers |

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `src/medical_agent/agents/graph.py` | LangGraph workflow definition |
| `src/medical_agent/agents/nodes.py` | Retrieve + generate node implementations |
| `src/medical_agent/agents/state.py` | State schema (TypedDict with add_messages) |
| `src/medical_agent/agents/llamaindex_retrieval.py` | Hybrid retriever (BM25 + vector) |
| `src/medical_agent/agents/utils.py` | Citation formatting + health context builder |
| `src/medical_agent/ingestion/pipeline.py` | Full ingestion pipeline (parse → chunk → embed → store) |
| `src/medical_agent/ingestion/metadata.py` | Medical metadata extraction (Pydantic + LLM) |
| `src/medical_agent/api/routes/query.py` | Main API endpoint |
| `src/medical_agent/core/config.py` | Pydantic Settings configuration |
| `src/medical_agent/evaluation/run_evaluation.py` | RAGAS evaluation runner |
