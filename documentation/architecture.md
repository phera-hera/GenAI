# Architecture

## Overview

Medical RAG system for vaginal pH analysis using **LlamaIndex CitationQueryEngine** with hybrid search retrieval. The system provides evidence-based health insights from curated medical research papers.

**Key Design Principles:**
- Use native LlamaIndex components (no custom wrappers)
- Direct CitationQueryEngine for retrieval with inline citations
- Hybrid search (BM25 + semantic) for better retrieval
- Structured outputs with Pydantic validation

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                          │
│          POST /api/v1/query {ph_value, health_profile}          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FASTAPI LAYER                           │
│  src/medical_agent/api/routes/query.py                          │
│  - Validates request                                            │
│  - Builds health profile context                               │
│  - Calls RAG retrieval                                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RAG RETRIEVAL (LlamaIndex)                     │
│  src/medical_agent/rag/llamaindex_retrieval.py                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Query Builder                                          │  │
│  │  - pH value + health profile context                   │  │
│  │  - Formatted medical query prompt                       │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CitationQueryEngine (GPT-4o)                           │  │
│  │  - Hybrid search (BM25 + vector similarity)             │  │
│  │  - Returns response with inline citations [1], [2]      │  │
│  │  - Captures citations in CitationRegistry               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Structured Output (Pydantic)                           │  │
│  │  - MedicalAnalysisResponse                              │  │
│  │  - Guaranteed schema validation                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VECTOR STORE LAYER                          │
│  LlamaIndex PGVectorStore (native)                              │
│  - Table: data_paper_chunks                                     │
│  - Hybrid search: BM25 + vector similarity                      │
│  - Metadata filtering via JSONB                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Ingestion Pipeline
**File:** `src/medical_agent/ingestion/pipeline.py`

```
PDF → Docling Reader (JSON export)
    → DoclingNodeParser (HybridChunker: section-aware + token limits)
    → Medical Metadata Extraction (8 fields: ethnicities, diagnoses, etc.)
    → Azure Embeddings (text-embedding-3-large, 3072 dims)
    → PGVectorStore (hybrid_search=True)
```

**Key Features:**
- **Docling**: Handles tables, sections, and complex layouts
- **HybridChunker**: Max 512 tokens/chunk (prevents embedding truncation)
- **Metadata**: Stored in JSONB `metadata_` field for filtering
- **Hybrid Search**: Automatic BM25 + vector fusion

---

### 2. Query Pipeline
**File:** `src/medical_agent/rag/llamaindex_retrieval.py`

```
User Query → Build query prompt (pH + health profile)
           → CitationQueryEngine
           → Hybrid Search (top_k=5)
           → LLM generates response with citations [1], [2]
           → Capture source nodes in CitationRegistry
           → Structured output (Pydantic)
           → Return with citations
```

**Key Features:**
- **Direct retrieval** with CitationQueryEngine (no multi-step reasoning)
- **Inline citations** in response text
- **Citations**: Captured automatically via CitationRegistry
- **Structured output**: Guaranteed schema via Pydantic

---

## Key Components

### CitationQueryEngine Configuration
```python
CitationQueryEngine.from_args(
    index=VectorStoreIndex.from_vector_store(vector_store),
    llm=AzureOpenAI(model="gpt-4o"),
    similarity_top_k=5,
    citation_chunk_size=512
)
```

### Vector Store Setup
```python
PGVectorStore.from_params(
    table_name="paper_chunks",
    hybrid_search=True,          # BM25 + vector fusion
    text_search_config="english",
    perform_setup=False          # Use existing schema
)
```

---

## Database Schema

### Papers Table
```sql
CREATE TABLE papers (
    id UUID PRIMARY KEY,
    title TEXT,
    authors TEXT,
    journal TEXT,
    publication_year INTEGER,
    doi TEXT,
    abstract TEXT,
    gcp_path TEXT,
    file_hash TEXT UNIQUE,
    is_processed BOOLEAN,
    processed_at TIMESTAMP
);
```

### Chunks Table (LlamaIndex managed)
```sql
CREATE TABLE data_paper_chunks (
    id VARCHAR PRIMARY KEY,              -- LlamaIndex node ID
    embedding VECTOR(3072),              -- Azure OpenAI embeddings
    text TEXT,                           -- Chunk content
    metadata_ JSONB,                     -- All metadata (paper_id, medical fields)
    _node_content TEXT                   -- LlamaIndex node serialization
);

-- Indexes
CREATE INDEX ON data_paper_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON data_paper_chunks USING GIN (metadata_);
CREATE INDEX ON data_paper_chunks USING GIN (to_tsvector('english', text));
```

**Metadata Structure:**
```json
{
    "paper_id": "uuid",
    "title": "Paper title",
    "doi": "10.xxxx/xxxxx",
    "ethnicities": ["Asian", "Caucasian"],
    "diagnoses": ["BV", "Candidiasis"],
    "symptoms": ["discharge", "odor"],
    "menstrual_status": ["premenopausal"],
    "birth_control": ["IUD", "pill"],
    "hormone_therapy": ["estrogen"],
    "fertility_treatments": []
}
```

---

## Configuration

### Required Environment Variables
```bash
# Azure OpenAI (LLM + Embeddings)
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_ENDPOINT=xxx
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# PostgreSQL with pgvector
DATABASE_URL=postgresql://user:pass@host:5432/db

# GCP Cloud Storage (for PDF storage)
GCP_PROJECT_ID=xxx
GCP_BUCKET_NAME=xxx
```

---

## Usage Examples

### Query the RAG System
```python
from medical_agent.rag import query_medical_rag

response, citations = await query_medical_rag(
    ph_value=5.2,
    health_profile={
        "age": 32,
        "symptoms": ["discharge", "odor"],
        "diagnoses": ["BV"],
        "ethnicity": ["Asian"]
    }
)

print(response.risk_level)  # "CONCERNING"
print(response.summary)
print(f"Found {len(citations)} citations")
```

### Ingest a Paper
```python
from medical_agent.ingestion.pipeline import MedicalIngestionPipeline

pipeline = MedicalIngestionPipeline()
result = await pipeline.process_paper(
    session=session,
    pdf_content=pdf_bytes,
    gcp_path="gs://bucket/paper.pdf"
)

print(f"Stored {result.stored_count} chunks")
```

---

## Performance Characteristics

- **Ingestion**: ~30-60 seconds per paper (depends on length)
- **Query**: 3-8 seconds end-to-end
- **Embedding**: 3072 dimensions (text-embedding-3-large)
- **Chunk size**: Max 512 tokens (optimal for retrieval)
- **Top-k**: 5 chunks per query
- **Hybrid search**: 0.7 vector + 0.3 BM25 (default fusion weights)

---

## Future Enhancements

When needed (based on production requirements):
- **LangGraph integration**: Wrap RAG retrieval in LangGraph for corrective RAG flow (grade → rewrite → reflect)
- **Conversation memory**: Track multi-turn conversations with MemorySaver
- **Human-in-the-loop**: Add approval steps for high-risk assessments
- **Custom retrievers**: Implement domain-specific retrieval strategies
- **Evaluation metrics**: Track RAG quality with LlamaIndex evaluators

---

For deployment and cloud setup, see [cloud-setup.md](cloud-setup.md).
