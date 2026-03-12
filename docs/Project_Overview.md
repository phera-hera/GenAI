# Project Overview — pHera Medical RAG Platform

## 1) Project Understanding

pHera is a medical retrieval-augmented generation (RAG) platform built for women's vaginal health insights. The product converts a patient context (pH reading plus optional profile signals such as age, symptoms, and diagnoses) into evidence-grounded guidance using curated medical literature and citation-backed responses.

The platform is designed as an internal research and evaluation system with a strong focus on medical safety, deterministic behavior, traceability, and operational reliability. Instead of producing open-ended LLM responses, the system is intentionally constrained to generate answers grounded in retrieved research chunks and supported by inline references.

---

## 2) What We Are Solving

Traditional health Q&A systems either:
- rely too heavily on model memory (hallucination risk), or
- return unstructured outputs with weak explainability.

pHera addresses this by combining:
- patient-context-aware retrieval,
- hybrid search over curated medical papers,
- reranking for relevance precision,
- structured generation with source references,
- and deployment-ready APIs for product integration.

The result is a medically safer, more auditable approach to AI-assisted health insights.

---

## 3) End-to-End System Flow (Ingestion to Response)

### A. Data Ingestion Pipeline (Offline)

The ingestion pipeline transforms research PDFs into retrieval-ready knowledge units:

1. Curated papers are sourced from cloud storage.
2. Documents are parsed and normalized for medical content structure.
3. Content is chunked into retrieval-friendly segments with contextual metadata.
4. Embeddings are generated using Azure OpenAI embedding models.
5. Chunks + metadata + vectors are stored in PostgreSQL with pgvector.
6. Hybrid indexing supports both semantic similarity and keyword search.

This creates a production-grade medical knowledge base that is continuously queryable and citation-ready.

### B. Query & Inference Pipeline (Online)

When a request arrives:

1. API receives pH value and optional health profile context.
2. Query is enriched/rewritten for retrieval quality.
3. **Retrieve Node** performs hybrid retrieval (BM25 + vector).
4. Candidate chunks are reranked by cross-encoder to improve precision.
5. Top evidence set is passed to downstream reasoning/generation.
6. **Reasoning & Verification Node** validates retrieval quality, checks evidence consistency, and performs agentic ranking/selection before response synthesis.
7. **Generate Node** produces medically framed, citation-backed output.
8. API returns structured response with answer, disclaimers, and citations.

This workflow ensures the model response is not only relevant, but also evidence-anchored and verifiable.

---

## 4) Technical Architecture

### Core Stack

- **Backend API:** FastAPI (async architecture)
- **Workflow Orchestration:** LangGraph
- **Retrieval Layer:** LlamaIndex + pgvector hybrid retrieval
- **Generation Layer:** Azure OpenAI GPT-4o (deterministic configuration)
- **Embeddings:** Azure OpenAI `text-embedding-3-large`
- **Reranking:** Cross-encoder model for relevance refinement
- **Database:** PostgreSQL 16 + pgvector
- **Storage:** GCP Cloud Storage for paper assets

### Architectural Design Principles

- **Medical grounding first:** retrieval-constrained answers over free generation.
- **Determinism for safety:** controlled model settings for stable outputs.
- **Traceability:** explicit citations and structured response contracts.
- **Separation of concerns:** ingestion, retrieval, reasoning, and generation are modular.
- **Cloud-operational readiness:** containerized deployment, health checks, and scalable serving.

---

## 5) What Has Been Implemented and Delivered

### Product and AI Capabilities

- Working medical RAG workflow for pH-contextualized health insights.
- Hybrid retrieval tuned for medical terminology precision.
- Over-retrieval + reranking strategy to improve evidence quality.
- Citation-aware response generation with disclaimer-driven output style.
- Structured API contracts for integration with clients.

### Data and Platform Foundations

- Curated medical corpus ingestion pipeline operationalized.
- Vectorized paper chunk storage in pgvector-backed PostgreSQL.
- Metadata-enriched indexing for targeted retrieval behavior.
- Query logging and evaluation-ready interfaces for research cycles.

### Engineering and Quality

- Async-first backend architecture for scalable request handling.
- Health endpoints and service checks for operational confidence.
- Evaluation tooling for dataset generation and quality benchmarking.
- Environment-based configuration for local, beta, and production modes.

---

## 6) Deployment Journey and Current Production Posture

The system has been deployed as a cloud-native backend for internal beta usage.

### Deployment Path

1. Containerized backend image built and versioned.
2. Image published to cloud artifact registry.
3. Service deployed to Cloud Run (EU region alignment with data services).
4. Runtime environment wired with model, storage, and database credentials.
5. Cloud SQL socket integration configured for secure database connectivity.
6. Health and query endpoints validated post-deployment.

### Current Deployed Characteristics

- Region-aligned deployment for lower latency and data-residency alignment.
- Warm instance strategy to reduce cold-start impact.
- Bounded autoscaling profile for cost-performance balance.
- Public beta endpoint posture for controlled internal testing.
- Log-driven observability and rollback-ready deployment flow.

---

## 7) Reasoning Node and Agentic Verification Layer

As the system evolved beyond a linear retrieve-generate pattern, a dedicated **Reasoning & Verification Node** was introduced between retrieval and generation to increase trustworthiness and response quality.

### Role of the Reasoning Node

- Validates whether retrieved chunks are truly relevant to the patient context.
- Checks internal consistency across retrieved evidence.
- Filters low-confidence or weakly grounded passages.
- Reorders evidence through agentic ranking logic before synthesis.
- Ensures only high-signal, medically pertinent context reaches generation.

### Impact

- Improves factual grounding and reduces noisy evidence leakage.
- Increases citation precision by aligning generated claims to stronger sources.
- Provides clearer auditability for why specific evidence was used.
- Creates a more robust foundation for future multi-agent medical workflows.

---

## 8) Business and Review-Ready Outcomes

From a performance-review lens, the project demonstrates delivery across the full AI product lifecycle:

- **Research to production:** converted medical paper corpus into a deployable intelligence layer.
- **Architecture ownership:** established modular RAG design with retrieval, reasoning, and generation separation.
- **Cloud execution:** shipped a functioning deployed backend with operational controls.
- **Quality orientation:** prioritized medical safety, deterministic behavior, and citation traceability.
- **Scalability readiness:** created a foundation that can extend to richer agentic orchestration and expanded product modes.

In short, pHera is not just an LLM integration; it is a structured medical AI system spanning ingestion, retrieval intelligence, reasoning validation, generation, and cloud deployment in a coherent production workflow.
