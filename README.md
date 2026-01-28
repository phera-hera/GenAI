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
| Observability | Langfuse |
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

5. **Start the database**
   ```bash
   docker-compose up -d postgres
   ```

6. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

7. **Start the development server**
   ```bash
   uvicorn medical_agent.api.main:app --reload
   ```

   Or with Docker:
   ```bash
   docker-compose up
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
│       │   ├── azure_openai.py          # Azure OpenAI client
│       │   ├── langfuse_client.py       # Observability client
│       │   ├── gcp_storage.py           # GCP Cloud Storage client
│       │   └── database/                # Database layer
│       │       ├── models.py            # SQLAlchemy models
│       │       ├── session.py           # DB session management
│       │       └── base.py              # Base classes
│       │
│       ├── ingestion/                   # Document processing pipeline
│       │   ├── pipeline.py              # LlamaIndex ingestion pipeline
│       │   │                            # (Docling + HybridChunker + Metadata + Embeddings)
│       │   └── metadata/                # Medical metadata extraction
│       │
│       ├── rag/                         # Medical RAG retrieval
│       │   └── llamaindex_retrieval.py # LlamaIndex CitationQueryEngine
│       │                                # (Direct retrieval with citations)
│       │
│       └── api/                         # Web layer (FastAPI)
│           ├── main.py                  # Application entry point
│           ├── schemas.py               # Pydantic request/response models
│           └── routes/                  # API endpoints
│               ├── health.py            # Health check endpoints
│               └── query.py             # pH analysis endpoint
│
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

```
api (FastAPI) → rag (CitationQueryEngine) → ingestion (LlamaIndex pipeline) → infrastructure → core
```

**Layer responsibilities:**
- **api**: HTTP endpoints and request/response handling
- **rag**: Medical research retrieval with LlamaIndex CitationQueryEngine
- **ingestion**: Document processing (Docling parsing, chunking, embeddings, metadata extraction)
- **infrastructure**: External services (Azure OpenAI, GCP Storage, PostgreSQL, Langfuse)
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
from medical_agent.ingestion.metadata import MedicalMetadataExtractor, ExtractedMetadata

# RAG (LlamaIndex CitationQueryEngine)
from medical_agent.rag import query_medical_rag, MedicalAnalysisResponse

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

See `.env.example` for all available configuration options. Key variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `GCP_PROJECT_ID` | GCP project for Cloud Storage |
| `GCP_BUCKET_NAME` | Cloud Storage bucket name |
| `LANGFUSE_PUBLIC_KEY` | Langfuse observability key |

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


