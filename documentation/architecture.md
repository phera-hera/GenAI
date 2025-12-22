# Architecture Overview

## Goals

- Deliver safe, informational responses with clear medical disclaimers.
- Keep core services modular to evolve RAG and agent logic independently.
- Support observability and future production hardening (scalability, readiness).

---

## High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│                    (Mobile App / Web / Internal Tools)                      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTPS
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ /health       │  │ /api/v1/query │  │ /api/v1/papers│  │ /api/v1/users│  │
│  │ (probes)      │  │ (pH analysis) │  │ (ingestion)   │  │ (profiles)   │  │
│  └───────────────┘  └───────┬───────┘  └───────┬───────┘  └──────────────┘  │
└─────────────────────────────┼───────────────────┼───────────────────────────┘
                              │                   │
              ┌───────────────┘                   └───────────────┐
              ▼                                                   ▼
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│         SERVICE LAYER           │         │       INGESTION PIPELINE        │
│  ┌───────────────────────────┐  │         │  ┌───────────────────────────┐  │
│  │     Query Service         │  │         │  │   Document Parser         │  │
│  │  (orchestrates agent)     │  │         │  │   (Azure Doc Intelligence)│  │
│  └─────────────┬─────────────┘  │         │  └─────────────┬─────────────┘  │
│                │                │         │                │                │
│                ▼                │         │                ▼                │
│  ┌───────────────────────────┐  │         │  ┌───────────────────────────┐  │
│  │   LANGGRAPH AGENT         │  │         │  │   Chunker                 │  │
│  │  ┌─────────────────────┐  │  │         │  │   (semantic splitting)    │  │
│  │  │ Input Validation    │  │  │         │  └─────────────┬─────────────┘  │
│  │  └──────────┬──────────┘  │  │         │                │                │
│  │             ▼             │  │         │                ▼                │
│  │  ┌─────────────────────┐  │  │         │  ┌───────────────────────────┐  │
│  │  │ RAG Retrieval       │──┼──┼─────────┼─▶│   Embedder                │  │
│  │  └──────────┬──────────┘  │  │         │  │   (Azure OpenAI)          │  │
│  │             ▼             │  │         │  └─────────────┬─────────────┘  │
│  │  ┌─────────────────────┐  │  │         │                │                │
│  │  │ Medical Reasoning   │  │  │         │                ▼                │
│  │  │ (GPT-4o + context)  │  │  │         │  ┌───────────────────────────┐  │
│  │  └──────────┬──────────┘  │  │         │  │   Storage                 │  │
│  │             ▼             │  │         │  │   (pgvector)              │  │
│  │  ┌─────────────────────┐  │  │         │  └───────────────────────────┘  │
│  │  │ Guardrails          │  │  │         │                                 │
│  │  │ (safety + disclaimers)│ │  │         └─────────────────────────────────┘
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                       │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │   PostgreSQL + pgvector     │    │   GCP Cloud Storage                 │ │
│  │   - User profiles           │    │   - Raw PDF papers                  │ │
│  │   - Query history           │    │   - Parsed documents                │ │
│  │   - Document chunks         │    │                                     │ │
│  │   - Embeddings (vectors)    │    │                                     │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OBSERVABILITY                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │   Langfuse                                                              ││
│  │   - Agent step traces                                                   ││
│  │   - RAG retrieval quality metrics                                       ││
│  │   - Latency monitoring                                                  ││
│  │   - Token usage tracking                                                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Query Request Flow

Shows the complete journey when a user submits a pH reading for analysis:

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Client  │    │  FastAPI │    │  Agent   │    │   RAG    │    │  LLM     │
│          │    │  Route   │    │ (Graph)  │    │ Retriever│    │ (GPT-4o) │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │               │               │
     │ POST /query   │               │               │               │
     │ {pH, profile} │               │               │               │
     │──────────────▶│               │               │               │
     │               │               │               │               │
     │               │ validate &    │               │               │
     │               │ start agent   │               │               │
     │               │──────────────▶│               │               │
     │               │               │               │               │
     │               │               │ build query   │               │
     │               │               │ from context  │               │
     │               │               │──────────────▶│               │
     │               │               │               │               │
     │               │               │               │ vector search │
     │               │               │               │ (pgvector)    │
     │               │               │               │───────┐       │
     │               │               │               │       │       │
     │               │               │               │◀──────┘       │
     │               │               │               │               │
     │               │               │ relevant docs │               │
     │               │               │◀──────────────│               │
     │               │               │               │               │
     │               │               │ reason with   │               │
     │               │               │ context + pH  │               │
     │               │               │──────────────────────────────▶│
     │               │               │               │               │
     │               │               │         response + risk level │
     │               │               │◀──────────────────────────────│
     │               │               │               │               │
     │               │               │ apply         │               │
     │               │               │ guardrails    │               │
     │               │               │───────┐       │               │
     │               │               │       │       │               │
     │               │               │◀──────┘       │               │
     │               │               │               │               │
     │               │ safe response │               │               │
     │               │ + disclaimer  │               │               │
     │               │◀──────────────│               │               │
     │               │               │               │               │
     │ {risk, info,  │               │               │               │
     │  citations,   │               │               │               │
     │  disclaimer}  │               │               │               │
     │◀──────────────│               │               │               │
     │               │               │               │               │
```

---

## Ingestion Pipeline Flow

Shows how medical research papers are processed and stored:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GCP       │     │   Parser    │     │   Chunker   │     │  Embedder   │
│   Storage   │     │  (Azure DI) │     │             │     │ (Azure OAI) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │  PDF upload       │                   │                   │
       │   (trigger)       │                   │                   │
       │──────────────────▶│                   │                   │
       │                   │                   │                   │
       │                   │ extract text,     │                   │
       │                   │ tables, layout    │                   │
       │                   │─────────┐         │                   │
       │                   │         │         │                   │
       │                   │◀────────┘         │                   │
       │                   │                   │                   │
       │                   │ structured doc    │                   │
       │                   │──────────────────▶│                   │
       │                   │                   │                   │
       │                   │                   │ split into        │
       │                   │                   │ semantic chunks   │
       │                   │                   │ (preserve context)│
       │                   │                   │─────────┐         │
       │                   │                   │         │         │
       │                   │                   │◀────────┘         │
       │                   │                   │                   │
       │                   │                   │ chunks + metadata │
       │                   │                   │──────────────────▶│
       │                   │                   │                   │
       │                   │                   │                   │ generate
       │                   │                   │                   │ embeddings
       │                   │                   │                   │────┐
       │                   │                   │                   │    │
       │                   │                   │                   │◀───┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   PostgreSQL        │
                           │   + pgvector        │
                           │                     │
                           │  - chunk text       │
                           │  - embedding vector │
                           │  - source metadata  │
                           │  - paper reference  │
                           └─────────────────────┘
```

---

## System Shape

- **FastAPI backend** as the single HTTP surface for mobile and internal clients.
- **Domain services** split into layers: API routes → service/logic layer → data/RAG layer.
- **RAG pipeline** separated from ingestion so retrieval/indexing can iterate without impacting the API surface.
- **Agent (LangGraph)** encapsulated as an orchestrator that can call retrieval, reasoning, and guardrails.
- **PostgreSQL with pgvector** for storing embeddings and metadata; decoupled via repository layer for future swaps.
- **Background ingestion pipeline** for document parsing, chunking, embedding, and storage.
- **Observability via Langfuse** to trace agent steps, retrieval quality, and latency.
- **Configuration via environment** (.env) to keep deployments environment-specific.

---

## Reasoning Behind This Design

| Principle | How It's Addressed |
|-----------|-------------------|
| **Separation of concerns** | API stays thin while domain logic, RAG, and ingestion evolve independently. |
| **Testability** | Clear layers make it easy to unit-test services, mock RAG/LLM calls, and run health probes. |
| **Safety** | Centralized guardrails and disclaimers in the agent/service layer reduce risk of unsafe outputs. |
| **Iteration speed** | RAG and ingestion are isolated; schema changes or new embedding models won't break the API. |
| **Operational readiness** | Health/readiness endpoints and observability hooks prepare for Cloud Run/Kubernetes. |
| **Extensibility** | Adding new data sources or additional retrieval strategies only touches the RAG/ingestion layers. |

---

## Layer Responsibilities

| Layer | Responsibility | Key Components |
|-------|---------------|----------------|
| **API** | HTTP handling, validation, routing | FastAPI routes, Pydantic schemas |
| **Service** | Business logic orchestration | Query service, user service |
| **Agent** | Multi-step reasoning workflow | LangGraph nodes, prompts |
| **RAG** | Retrieval and context building | LlamaIndex, vector search |
| **Ingestion** | Document processing pipeline | Parsers, chunkers, embedders |
| **Data** | Persistence and queries | Repositories, pgvector |
| **Observability** | Tracing and monitoring | Langfuse integration |
