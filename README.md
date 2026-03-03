# FemTech Medical RAG Agent

A mobile-first diagnostic platform for women's vaginal health using RAG-based medical reasoning over curated research papers.

## ⚠️ Medical Disclaimer

**This system is purely informational and is NOT intended to:**
- Diagnose medical conditions
- Prescribe treatments or medications  
- Replace professional medical advice

Always consult with a qualified healthcare provider for medical concerns.

## Overview

This platform takes pH values from test strip photos (provided by a separate CV model) along with user health profiles and reasons over 250-500 curated medical research papers to provide personalized, evidence-based health insights.

### Key Features

- **pH-Based Analysis**: Analyze vaginal pH readings with health context
- **Evidence-Based Insights**: Retrieve information from curated medical research
- **Risk Assessment**: Categorize readings into actionable risk levels (NORMAL, MONITOR, CONCERNING, URGENT)
- **Medical Guardrails**: Strong safeguards to prevent diagnostic or prescriptive advice

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Package Manager | uv |
| Backend | FastAPI |
| LLM | Azure OpenAI (GPT-4o) |
| Embeddings | Azure OpenAI (text-embedding-3-large) |
| Document Parsing | Docling (LlamaIndex) |
| Vector DB | pgvector (PostgreSQL) with hybrid search |
| RAG System | LlamaIndex CitationQueryEngine |
| RAG Framework | LlamaIndex |
| Observability | LangSmith |
| Deployment | GCP Cloud Run |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Docker and Docker Compose
- Azure OpenAI API access
- GCP account (for Cloud Storage and Cloud SQL)

### Installation

1. **Clone the repository**
   ```bash
   cd Medical_Agent
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the development server**
   ```bash
   uvicorn medical_agent.api.main:app --reload
   ```

7. **Run the containerized API (optional)**
   ```bash
   docker-compose up -d app
   ```

8. **Verify cloud services setup**
   ```bash
   python scripts/setup_infrastructure.py --check-all
   ```

### Accessing the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Detailed Health**: http://localhost:8000/health/detailed
- **Cloud Services Health**: http://localhost:8000/health/cloud-services

## Project Structure

```
Medical_Agent/
├── src/
│   └── medical_agent/                    # Main package namespace
│       ├── core/                         # Domain core (no dependencies)
│       │   ├── config.py                 # Application configuration
│       │   └── exceptions.py             # Custom exceptions
│       │
│       ├── infrastructure/               # External services & integrations
│       │   ├── azure_openai.py           # Azure OpenAI client factories
│       │   ├── gcp_storage.py           # GCP Cloud Storage client
│       │   └── database/                # Database layer
│       │       ├── models.py            # SQLAlchemy models
│       │       ├── session.py           # DB session management
│       │       └── base.py              # Base classes
│       │
│       ├── ingestion/                   # Document processing pipeline
│       │   ├── pipeline.py              # LlamaIndex ingestion pipeline
│       │   ├── metadata.py              # Medical metadata extraction
│       │   ├── contextual_chunking.py   # Section-aware chunk headers
│       │   └── table_transformer.py     # Table chunk normalization
│       │
│       ├── agents/                      # LangGraph medical RAG workflow
│       │   ├── graph.py                 # Compiled workflow graph
│       │   ├── nodes.py                 # Retrieval/rerank/generation nodes
│       │   ├── llamaindex_retrieval.py  # Retriever + citation formatting
│       │   └── state.py                 # Graph state schema
│       │
│       └── api/                         # Web layer (FastAPI)
│           ├── main.py                  # Application entry point
│           ├── schemas.py               # Pydantic request/response models
│           └── routes/                  # API endpoints
│               ├── health.py            # Health check endpoints
│               └── query.py             # pH analysis endpoint
│
├── streamlit_app.py                     # Streamlit demo client
├── tests/                               # Test suite
├── scripts/                             # CLI utilities
├── alembic/                             # Database migrations
├── infrastructure/gcp/                  # Terraform IaC for GCP
├── data/golden_set/                     # Test cases
├── documentation/                       # Additional documentation
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

### Architecture Layers

The project follows clean architecture with clear separation of concerns:

`api (FastAPI) -> agents (LangGraph workflow) -> llamaindex retrieval + reranking -> infrastructure/core`

**Layer responsibilities:**
- **api**: HTTP endpoints and request/response handling
- **agents**: Orchestration across query rewriting, retrieval, reranking, and response generation
- **ingestion**: Document processing (Docling parsing, chunking, embeddings, metadata extraction)
- **infrastructure**: External services (Azure OpenAI, GCP Storage, PostgreSQL)
- **core**: Configuration and domain exceptions

## Risk Assessment Logic

| pH Range | Symptoms | Risk Level |
|----------|----------|------------|
| 3.8-4.5 | None | NORMAL |
| 3.8-4.5 | Mild symptoms | MONITOR |
| 4.5-5.0 | Any | CONCERNING |
| Above 5.0 | Any | URGENT |

## Development

### Running Tests
```bash
pytest
```

### Importing the Package

The project uses the `src/` layout. All imports use the `medical_agent` namespace:

```python
# Core configuration
from medical_agent.core.config import settings
from medical_agent.core.exceptions import AppException

# Infrastructure (embedding models for retrieval)
from medical_agent.infrastructure import get_llama_index_embed_model
from medical_agent.infrastructure.database.models import Paper, PaperChunk

# Ingestion pipeline
from medical_agent.ingestion.pipeline import MedicalIngestionPipeline, PipelineConfig
from medical_agent.ingestion.metadata import MedicalMetadata, create_medical_metadata_extractor

# Agent graph + retrieval helpers
from medical_agent.agents import medical_rag_app, retrieve_nodes

# API application
from medical_agent.api.main import app
```

When running scripts outside the virtual environment:
```bash
PYTHONPATH=src python scripts/your_script.py
```

### Code Quality
```bash
# Linting
ruff check src/

# Type checking
mypy src/medical_agent

# Format code
ruff format src/
```

### Database Migrations
```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Cloud Infrastructure Setup

For detailed cloud infrastructure setup instructions, see:
- [Cloud Setup Guide](documentation/cloud-setup.md) - Complete setup documentation
- [GCP Terraform README](infrastructure/gcp/README.md) - Infrastructure as Code

### Quick Setup

1. **Interactive setup guide**:
   ```bash
   python scripts/setup_infrastructure.py --setup-guide
   ```

2. **Terraform (GCP resources)**:
   ```bash
   cd infrastructure/gcp
   terraform init
   cp variables.tfvars.example terraform.tfvars
   # Edit terraform.tfvars
   terraform apply
   ```

3. **Verify all services**:
   ```bash
   python scripts/setup_infrastructure.py --check-all
   ```

## Environment Variables

Copy `.env.example` to `.env`, then set these values.

### Core Application

| Variable | Description |
|----------|-------------|
| `ENVIRONMENT` | Runtime mode (`development`, `staging`, `production`) |
| `DEBUG` | Enable debug behavior/logging |
| `APP_NAME` | App display name |
| `APP_VERSION` | App version string |
| `HOST` | API bind host |
| `PORT` | API bind port |

### Database

| Variable | Description |
|----------|-------------|
| `POSTGRES_USER` | PostgreSQL username |
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `POSTGRES_DB` | PostgreSQL database name |
| `POSTGRES_HOST` | PostgreSQL hostname |
| `POSTGRES_PORT` | PostgreSQL port |
| `DATABASE_URL` | Optional override for full DB URL (recommended in production) |

### Azure OpenAI (LLM)

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_API_KEY` | API key for chat/completions model |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_VERSION` | API version for chat/completions |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Main model deployment (for generation) |
| `AZURE_OPENAI_MINI_DEPLOYMENT_NAME` | Smaller model deployment (query rewriting/helpers) |

### Azure OpenAI (Embeddings)

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_EMBEDDING_API_KEY` | API key for embedding model (can match main key) |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | Endpoint for embedding model (can match main endpoint) |
| `AZURE_OPENAI_EMBEDDING_API_VERSION` | API version for embedding model |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | Embedding deployment name (`text-embedding-3-large`) |

### GCP

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | GCP project ID |
| `GCP_BUCKET_NAME` | Cloud Storage bucket containing papers |
| `GCP_CREDENTIALS_PATH` | Local path to service account JSON key |
| `GCP_CLOUD_SQL_INSTANCE` | Cloud SQL instance name (`project:region:instance`) |

### LangSmith (Optional Observability)

| Variable | Description |
|----------|-------------|
| `LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_TRACING` | Enable LangSmith tracing |
| `LANGSMITH_PROJECT` | LangSmith project name |

### Retrieval + Medical Thresholds

| Variable | Description |
|----------|-------------|
| `EMBEDDING_DIMENSION` | Embedding vector size (defaults to 3072) |
| `VECTOR_SIMILARITY_TOP_K` | Number of retrieved chunks before downstream filtering |
| `PH_NORMAL_MIN` | Lower bound for normal pH |
| `PH_NORMAL_MAX` | Upper bound for normal pH |
| `PH_CONCERNING_THRESHOLD` | Threshold for concerning/urgent risk logic |

## Evaluation

Install evaluation extras, generate a synthetic testset from ingested chunks, then run RAGAS scoring.

```bash
uv pip install -e ".[evaluation]"
```

### Generate testset

```bash
python -m medical_agent.evaluation.generate_testset --size 20 --limit 200
```

- Optional flags:
  - `--paper "keyword"` limits chunks to papers matching title text
  - `--seed 42` enables reproducible chunk ordering
- Output: CSV written to `src/medical_agent/evaluation/testsets/`

### Run evaluation

```bash
python -m medical_agent.evaluation.run_evaluation --testset src/medical_agent/evaluation/testsets/testset_<timestamp>.csv
```

- Required columns in testset CSV: `user_input`, `reference`
- Output: JSON report written to `src/medical_agent/evaluation/results/`

## API Endpoints

### Health
- `GET /health` - Basic health check
- `GET /health/detailed` - Component health status
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe
- `GET /health/cloud-services` - Cloud services connectivity check

### Query
- `POST /api/v1/query` - Submit pH reading for analysis with health profile

For detailed API documentation, visit http://localhost:8000/docs when the server is running.

## License

MIT


