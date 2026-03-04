# Project 2 - MVP (User-Facing Product)

## Overview

The MVP is Phera's first public-facing product. It combines:

- A computer vision (CV) service that reads a physical pH strip photo and returns an exact pH value
- A RAG backend that generates personalized, evidence-based health insights

Users can use the app in two modes:

- **Anonymous mode** (no login): user can still receive personalized analysis for the current submission
- **Authenticated mode** (logged in): data is associated with a persistent user profile and saved over time

This product is designed to convert a home pH test result into understandable, clinically grounded guidance tailored to each user's context.

## Purpose

The MVP answers two core user questions:

1. **What is my exact pH value?**  
   Answered by the CV Service from the uploaded strip image.
2. **What does this mean for me specifically?**  
   Answered by the RAG Backend using submitted health context and medical research retrieval.

The experience intentionally separates these two layers:

- **Step 1**: objective pH measurement + rule-based interpretation
- **Step 2**: personalized interpretation grounded in retrieved evidence

## Target Users

- Women using the Phera home testing kit
- Users who prefer no-account usage (anonymous flow)
- Users who want history/profile continuity (logged-in flow)

## End-to-End Product Flow

```text
User
  |
  v
Frontend (MVP App)
  |
  |-- STEP 0: Optional authentication
  |     - User can register/login via Zitadel
  |     - Or continue without authentication
  |
  |-- STEP 1: pH detection and first-level interpretation [CV Service]
  |     - User follows SOP and captures strip image
  |     - Frontend sends photo to CV Service
  |     - CV Service returns exact pH (example: 4.5)
  |     - Frontend shows:
  |         -> Exact pH value
  |         -> Rule-based (non-personalized) interpretation
  |         -> "Get Personalized Result" call-to-action
  |
  |-- STEP 2: Personalized analysis request [RAG Backend via BFF]
  |     - User clicks "Get Personalized Result"
  |     - Frontend shows optional health context form:
  |         age, diagnoses, symptoms, ethnic background,
  |         birth control, hormone therapy, fertility journey
  |     - User submits form to BFF
  |
  v
BFF (MVP Mode)
  |
  |-- Checks login state
  |
  |-- If logged in:
  |     - Validates Zitadel JWT
  |     - Extracts user id (token subject)
  |     - Forwards to RAG Backend with:
  |         X-API-Key: <secret>
  |         X-User-Id: <zitadel_user_id>
  |         + request payload
  |
  '-- If anonymous:
        - Forwards with:
            X-API-Key: <secret>
            + request payload
          (no X-User-Id)
  |
  v
RAG Backend (MVP Instance)
  |
  |-- Validates X-API-Key
  |     - Invalid key -> reject with 401
  |
  |-- Detects identity mode via X-User-Id presence
  |
  |-- Builds health profile from submitted fields
  |
  |-- Uses GPT-4o-mini to convert structured fields
  |   into a natural language retrieval query
  |
  |-- Runs LangGraph pipeline:
  |
  |   Retrieve Node:
  |     -> Hybrid retrieval (pgVector semantic + BM25 keyword)
  |     -> Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
  |     -> Top relevant research chunks
  |
  |   Reasoning Node (GPT-4o):
  |     -> Interprets evidence relative to user profile
  |     -> Prioritizes clinically relevant context
  |     -> Structures reasoning for generation
  |
  |   Generate Node (GPT-4o):
  |     -> Produces personalized answer with inline citations
  |     -> Includes disclaimer in response text
  |
  |-- Persistence behavior:
  |
  |   If X-User-Id present:
  |     -> Find User by external_id
  |     -> Create User if first authenticated query
  |     -> Save QueryLog linked to user
  |     -> Save/update HealthProfile from current submission
  |
  '-- If X-User-Id absent:
        -> Save anonymous QueryLog (user_id = null)
  |
  '-- Returns to frontend:
        agent_reply (contains disclaimer), citations[]
  |
  v
Frontend renders personalized result + citations
```

## Authentication and Trust Boundary

Authentication is delegated to Zitadel and enforced by the BFF.  
The RAG Backend does **not** validate Zitadel JWTs directly.

| Responsibility | System |
|---|---|
| User login/registration and JWT issuance | Zitadel |
| JWT validation | BFF |
| User ID extraction | BFF |
| Trusted identity forwarding (`X-User-Id`) | BFF |
| Service-to-service authorization (`X-API-Key`) | RAG Backend middleware |
| User lookup/creation and profile persistence | RAG Backend |

This design keeps token logic centralized in the BFF while giving the backend a simple, trusted identity signal.

## Data Persistence Model

The RAG Backend owns all query and profile storage.

| Entity | Stored | Condition / behavior |
|---|---|---|
| QueryLog | Yes | Always stored. Authenticated requests are linked with `user_id`; anonymous requests store `user_id = null`. |
| User | Yes | Created on first authenticated request if no record exists for `external_id`. |
| HealthProfile | Yes | Saved/updated on authenticated requests using submitted form data. |

### Stored Query Context

Query logs include operational and response context such as:

- pH value
- generated/retrieval query
- response text
- citations
- timing/latency metadata

## Frontend Experience Contract

### Step 1 - After CV response

Frontend must display:

- Exact pH value returned by CV Service
- Rule-based interpretation based only on pH number (hardcoded, non-personalized)
- Visible "Get Personalized Result" action on the same screen

### Step 2 - After RAG response

Frontend must display:

- `agent_reply` (includes embedded disclaimer)
- `citations[]`

## MVP Scope Boundaries (Non-Goals)

- No production Swagger/OpenAPI exposure
- No sharing of user health/query data outside the RAG Backend database

## Integration Notes

### BFF Requirements

- Must run in **`mvp` mode**
- For authenticated requests:
  - Validate Zitadel JWT
  - Extract `sub` claim
  - Forward user id as `X-User-Id`
- For all requests to RAG Backend:
  - Include `X-API-Key`
- For anonymous requests:
  - Send payload with `X-API-Key` only
  - Omit `X-User-Id`
- RAG Backend base URL and API key are deployment-provided configuration

### Frontend Requirements

- Step 1:
  - Capture strip image
  - Call CV Service
  - Render pH + rule-based interpretation + personalized CTA
- Step 2:
  - Collect optional health context form values
  - Submit to BFF
  - Render `agent_reply` and `citations[]`
- Authentication UI/session is Zitadel-backed

### CV Service Requirements

- Accepts pH strip image from frontend
- Returns exact pH value
- Full API contract to be finalized in a dedicated integration spec

## Deployment Model

### RAG Backend

- Deploy as a dedicated Cloud Run instance (separate from Beta)
- Environment mode: `DEPLOYMENT_MODE=mvp`
- Same Docker image as Beta; behavior controlled by configuration
- Base URL: shared post-deployment

### CV Service

- Independently deployed by CV engineering
- Separate Cloud Run service
- No shared runtime infrastructure requirement with RAG Backend
- Detailed integration configuration to be added once finalized

## Technical Stack

### RAG Backend

| Component | Technology |
|---|---|
| API framework | FastAPI (Python) |
| Agent orchestration | LangGraph |
| Retrieval | LlamaIndex + pgVector (hybrid search) |
| Reranking | sentence-transformers (`ms-marco-MiniLM-L-6-v2`) |
| Reasoning model | Azure OpenAI GPT-4o |
| Generation model | Azure OpenAI GPT-4o |
| Query generation model | Azure OpenAI GPT-4o-mini |
| Vector DB | PostgreSQL + pgVector (GCP Cloud SQL) |
| Research corpus storage | GCP Cloud Storage |
| Identity provider | Zitadel |
| Service-to-service auth | Custom FastAPI API-key middleware |

### CV Service

- Tech stack: pending confirmation

