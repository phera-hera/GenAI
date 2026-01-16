---
name: Complete Medical Agent MVP
overview: Implement the missing API endpoints, lifecycle management, basic testing, and sample data ingestion to make the Medical Agent fully functional and production-ready.
todos:
  - id: implement_query_api
    content: Create query endpoint that integrates with existing agent workflow
    status: pending
  - id: implement_papers_api
    content: Create papers management endpoints for ingestion and listing
    status: pending
  - id: implement_users_api
    content: Create user profile management endpoints
    status: pending
  - id: complete_lifecycle_hooks
    content: Implement database, Langfuse, and client initialization in app startup
    status: pending
  - id: define_request_response_schemas
    content: Create comprehensive Pydantic models for all API endpoints
    status: pending
  - id: add_basic_testing
    content: Implement minimal integration and workflow tests
    status: pending
  - id: add_sample_data
    content: Include sample medical papers and test data
    status: pending
  - id: update_router_registration
    content: Wire new API routes into main router
    status: pending
---

# Complete Medical Agent MVP

## Overview

Transform the current health-check-only API into a fully functional medical reasoning system by implementing the missing API endpoints, proper lifecycle management, and basic testing infrastructure.

## Current State Analysis

- **Infrastructure Layer**: Complete (database models, Azure OpenAI, GCP storage, Langfuse, vector store)
- **Agent Logic**: Complete (LangGraph workflow, retriever, risk assessment, response generation)
- **API Layer**: Only health endpoints implemented
- **Testing**: Only health endpoint tests
- **Lifecycle**: TODOs for database/client initialization

## Implementation Strategy

### Phase 1: Core API Endpoints

Implement the three main API endpoint groups mentioned in the README:

**Query Endpoint** - [`src/medical_agent/api/routes/query.py`](src/medical_agent/api/routes/query.py)

- `POST /api/v1/query` - Main endpoint that accepts pH value, health profile, and optional query text
- Integrates with existing `run_medical_agent()` function from [`src/medical_agent/agent/graph.py`](src/medical_agent/agent/graph.py)
- Request/response schemas in [`src/medical_agent/api/schemas.py`](src/medical_agent/api/schemas.py)

**Papers Management** - [`src/medical_agent/api/routes/papers.py`](src/medical_agent/api/routes/papers.py)

- `GET /api/v1/papers` - List ingested papers with metadata
- `POST /api/v1/papers/ingest` - Trigger ingestion pipeline for new papers
- Uses existing ingestion pipeline from [`src/medical_agent/ingestion/pipeline.py`](src/medical_agent/ingestion/pipeline.py)

**User Profiles** - [`src/medical_agent/api/routes/users.py`](src/medical_agent/api/routes/users.py)

- `POST /api/v1/users` - Create/update user health profiles
- `GET /api/v1/users/{id}/profile` - Retrieve user health profile
- Extends existing database models in [`src/medical_agent/infrastructure/database/models.py`](src/medical_agent/infrastructure/database/models.py)

### Phase 2: Application Lifecycle Management

Complete the TODOs in [`src/medical_agent/api/main.py`](src/medical_agent/api/main.py):

**Startup Initialization**:

- Database connection pool setup using existing session management
- Langfuse client initialization with proper configuration
- Azure OpenAI client warming and connection validation
- Embedding model preloading for faster first requests

**Shutdown Cleanup**:

- Graceful database connection closure
- Langfuse trace flushing and client cleanup
- Resource cleanup for all external service clients

### Phase 3: Request/Response Schema Definition

Extend [`src/medical_agent/api/schemas.py`](src/medical_agent/api/schemas.py) with comprehensive Pydantic models:

**Query Schemas**:

```python
class QueryRequest(BaseModel):
    ph_value: float = Field(ge=0, le=14)
    health_profile: Optional[HealthProfile] = None
    query_text: Optional[str] = None
    is_pregnant: bool = False

class QueryResponse(BaseModel):
    session_id: str
    risk_level: RiskLevel
    response: ResponseContent
    citations: List[Citation]
    processing_time_ms: int
```

### Phase 4: Basic Testing Infrastructure

Add minimal tests to ensure system stability:

**API Integration Tests** - [`tests/test_api_integration.py`](tests/test_api_integration.py)

- Test query endpoint with valid pH values
- Test papers listing and ingestion endpoints
- Test user profile creation and retrieval
- Mock external services (Azure OpenAI, GCP) for reliable testing

**Agent Workflow Tests** - [`tests/test_agent_workflow.py`](tests/test_agent_workflow.py)

- Test complete agent pipeline with sample data
- Verify risk assessment logic with different pH ranges
- Test response generation and citation extraction

### Phase 5: Sample Data and Documentation

**Sample Medical Papers** - [`data/sample_papers/`](data/sample_papers/)

- Include 3-5 sample medical research papers (PDF format)
- Cover different aspects: pH balance, infections, pregnancy considerations
- Demonstrate the ingestion and retrieval pipeline

**API Documentation Enhancement**:

- Update FastAPI OpenAPI documentation with comprehensive examples
- Add request/response examples for each endpoint
- Include error handling documentation

## Technical Implementation Details

### Database Schema Extensions

Add user profile tables to existing schema:

```sql
-- New migration in alembic/versions/
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR NOT NULL UNIQUE,
    age INTEGER,
    is_pregnant BOOLEAN DEFAULT FALSE,
    medical_history JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Error Handling Strategy

- Consistent error responses using existing `AppException` framework
- Proper HTTP status codes (400 for validation, 500 for internal errors)
- Structured error messages with error codes for client handling

### Security Considerations

- Input validation for pH values (0-14 range)
- Sanitization of user-provided text inputs
- Rate limiting considerations for expensive operations (agent queries)

## File Structure After Implementation

```
src/medical_agent/api/routes/
├── __init__.py          # Updated to include new routers
├── health.py           # Existing health checks
├── query.py            # NEW: Main agent query endpoint
├── papers.py           # NEW: Paper management endpoints
└── users.py            # NEW: User profile endpoints

tests/
├── test_health.py      # Existing health tests
├── test_api_integration.py  # NEW: API endpoint tests
└── test_agent_workflow.py   # NEW: Agent pipeline tests

data/sample_papers/     # NEW: Sample medical papers for demo
```

## Success Criteria

1. All API endpoints return proper responses with valid data
2. Agent workflow completes successfully for sample queries
3. Papers can be ingested and retrieved from vector store
4. User profiles can be created and managed
5. Application starts/stops cleanly with proper resource management
6. Basic tests pass and can be run in CI/CD pipeline

## Deployment Readiness

After completion, the system will be ready for:

- Docker deployment using existing `Dockerfile` and `docker-compose.yml`
- GCP Cloud Run deployment using existing Terraform configuration
- Production monitoring via Langfuse observability
- Horizontal scaling with proper database connection pooling