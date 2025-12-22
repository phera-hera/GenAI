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
| Embeddings | Azure OpenAI (text-embedding-3-small) |
| Document Parsing | Azure Document Intelligence |
| Vector DB | pgvector (PostgreSQL) |
| Agent Framework | LangGraph |
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
   uvicorn app.main:app --reload
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
├── app/
│   ├── api/routes/          # FastAPI endpoints
│   ├── core/                # Config, security, exceptions
│   ├── db/                  # Database models and repositories
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   └── main.py
├── agent/
│   ├── graph.py             # LangGraph workflow
│   ├── state.py             # Agent state schema
│   ├── nodes/               # Individual agent nodes
│   └── prompts/             # System prompts and templates
├── ingestion/
│   ├── pipeline.py          # Main orchestrator
│   ├── parsers/             # Document parsing
│   ├── chunkers/            # Medical paper chunking
│   ├── embedders/           # Embedding generation
│   └── storage/             # Storage operations
├── rag/
│   ├── index.py             # LlamaIndex setup
│   ├── retriever.py         # Custom retriever
│   └── query_engine.py
├── evaluation/              # Testing framework
├── infrastructure/
│   └── gcp/                 # Terraform IaC for GCP
├── scripts/                 # CLI utilities
├── tests/
├── data/golden_set/         # Test cases
├── alembic/                 # Database migrations
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

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

### Code Quality
```bash
# Linting
ruff check .

# Type checking
mypy app
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
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Document parsing endpoint |
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

### Query (Coming Soon)
- `POST /api/v1/query` - Submit pH reading for analysis

### Papers (Coming Soon)
- `GET /api/v1/papers` - List ingested papers
- `POST /api/v1/papers/ingest` - Trigger paper ingestion

### Users (Coming Soon)
- `POST /api/v1/users` - Create user profile
- `GET /api/v1/users/{id}/profile` - Get health profile

## License

MIT


