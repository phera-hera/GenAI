# Tech Stack and Rationale

This document explains the technologies used in the FemTech Medical RAG Agent, why each was chosen, and what role it plays in the system.

---

## Quick Summary

| Category | Technology | One-Line Purpose |
|----------|------------|------------------|
| Language | Python 3.11+ | Core programming language |
| Package Manager | uv | Fast dependency management |
| API Framework | FastAPI | Handles web requests |
| LLM | Azure OpenAI (GPT-4o) | Powers medical reasoning |
| Embeddings | text-embedding-3-small | Converts text to searchable vectors |
| Document Parsing | LlamaParser (LlamaCloud) | Extracts text from PDFs |
| Database | PostgreSQL + pgvector | Stores data and enables similarity search |
| Agent Framework | LangGraph | Orchestrates multi-step AI workflows |
| RAG Framework | LlamaIndex | Manages retrieval and querying |
| Observability | Langfuse | Monitors AI behavior and performance |
| Containers | Docker | Packages the application for deployment |
| Testing | Pytest, Ruff, MyPy | Ensures code quality |

---

## Detailed Breakdown

### Language and Runtime

#### Python 3.11+

**What it is:**  
Python is a programming language known for readability and a rich ecosystem of AI/ML libraries.

**Why we chose it:**
- Industry standard for AI/ML applications
- Excellent library support for LLMs, vector databases, and web APIs
- Version 3.11+ offers significant performance improvements
- Large community means easier hiring and troubleshooting

---

#### uv (Package Manager)

**What it is:**  
A modern, extremely fast Python package manager that replaces pip and virtualenv.

**Why we chose it:**
- Installs dependencies 10-100x faster than traditional pip
- Creates reproducible environments (same setup on every machine)
- Handles Python version management
- Growing adoption in the Python community

**For non-technical readers:** Think of it as a tool that ensures everyone working on the project has exactly the same setup, preventing "it works on my machine" problems.

---

### API Framework

#### FastAPI

**What it is:**  
A modern web framework for building APIs (the interface through which mobile apps and other systems communicate with our backend).

**Why we chose it:**
- **Async-first**: Handles many simultaneous requests efficiently (important for AI workloads that can be slow)
- **Automatic documentation**: Generates interactive API docs at `/docs` without extra work
- **Type safety**: Catches errors before they reach production
- **High performance**: One of the fastest Python web frameworks
- **OpenAPI standard**: Makes integration with other systems straightforward

**For non-technical readers:** FastAPI is like a receptionist that takes requests from users (via their phones or computers), routes them to the right department, and returns responses—all very quickly and with detailed records of every interaction.

---

### LLM and Embeddings

#### Azure OpenAI (GPT-4o)

**What it is:**  
Microsoft's hosted version of OpenAI's GPT-4o model—a large language model capable of understanding and generating human-like text.

**Why we chose it:**
- **Enterprise-grade**: Microsoft's infrastructure with SLAs, compliance certifications, and data privacy guarantees
- **GPT-4o capabilities**: Latest model with strong reasoning, medical knowledge, and instruction-following
- **Regional availability**: Can be deployed in specific geographic regions for data residency requirements
- **Integrated billing**: Consolidated with other Azure services

**Role in our system:** Performs the actual medical reasoning—taking pH values, user health context, and retrieved research to generate safe, informational responses.

**For non-technical readers:** This is the "brain" of the system. It reads the user's question, looks at relevant medical research, and writes a helpful response—all while following strict safety guidelines.

---

#### text-embedding-3-small

**What it is:**  
A model that converts text into numerical vectors (lists of numbers) that capture meaning.

**Why we chose it:**
- **Semantic understanding**: Similar concepts get similar vectors, enabling "find me something like this" searches
- **Cost-effective**: Smaller model = lower cost per embedding
- **1536 dimensions**: Good balance between accuracy and storage efficiency
- **Azure-hosted**: Same enterprise benefits as GPT-4o

**Role in our system:** When we ingest research papers, this model converts each chunk into a vector. When a user asks a question, it converts the question into a vector too. We then find research chunks with similar vectors.

**For non-technical readers:** Imagine every piece of medical research is assigned a unique GPS coordinate based on its meaning. When you ask a question, we find your question's "coordinate" and look for research papers at nearby coordinates—papers that are about similar topics.

---

#### LlamaParser (LlamaCloud)

**What it is:**  
A cloud service from LlamaIndex that extracts text, tables, and structure from documents like PDFs using advanced AI models.

**Why we chose it:**
- **Handles complex layouts**: Medical papers have multi-column text, tables, figures, headers—LlamaParser understands all of it
- **High accuracy**: Uses AI-powered extraction for superior quality
- **Table extraction**: Critical for medical papers with data tables
- **LlamaIndex integration**: Seamless integration with our RAG framework

**Role in our system:** When we upload a medical research PDF, LlamaParser reads it and extracts the text in a structured way, preserving the meaning and organization in markdown format.

**For non-technical readers:** Medical papers are complex documents with graphs, tables, and special formatting. LlamaParser "reads" those papers like a human would, understanding that a table is a table and a heading is a heading.

---

### Retrieval and Storage

#### PostgreSQL + pgvector

**What it is:**  
PostgreSQL is a powerful, open-source relational database. pgvector is an extension that adds vector similarity search capabilities.

**Why we chose it:**
- **Proven reliability**: PostgreSQL has 30+ years of production use
- **Familiar SQL**: Standard query language that developers know
- **Vector search built-in**: pgvector adds AI-native capabilities without a separate database
- **Managed options**: Available on AWS, GCP, Azure as managed services
- **Cost-effective**: No separate vector database licensing or infrastructure
- **ACID compliance**: Guarantees data integrity for healthcare applications

**Role in our system:**
- Stores user profiles and query history
- Stores document chunks with their embeddings
- Performs fast similarity searches to find relevant research

**For non-technical readers:** This is our filing cabinet and search engine combined. It stores all the information (user data, medical research) and can quickly find research that's relevant to any question—not by matching exact words, but by understanding meaning.

---

### Agent and RAG

#### LangGraph

**What it is:**  
A framework for building AI agents as directed graphs—workflows where each step can make decisions about what to do next.

**Why we chose it:**
- **Controllable**: Each step is explicit, making behavior predictable and testable
- **State management**: Tracks conversation context across multiple steps
- **Conditional logic**: Can branch based on risk levels, missing information, etc.
- **Guardrails**: Easy to insert safety checks between steps
- **Debugging**: Clear visibility into what the agent did and why

**Role in our system:** Orchestrates the entire reasoning process:
1. Validate input
2. Retrieve relevant research
3. Perform medical reasoning
4. Apply safety guardrails
5. Format response with disclaimers

**For non-technical readers:** Think of this as a workflow manager. When a question comes in, it follows a specific checklist: check the input, find relevant research, think about it carefully, make sure the response is safe, then send it back. Every step is logged and can be reviewed.

---

#### LlamaIndex

**What it is:**  
A framework specifically designed for connecting LLMs to external data sources (like our medical research database).

**Why we chose it:**
- **RAG-focused**: Built specifically for retrieval-augmented generation
- **Abstraction layer**: Simplifies complex retrieval logic
- **Multiple retrieval strategies**: Hybrid search, re-ranking, filtering
- **Query engines**: Pre-built patterns for common use cases
- **Active development**: Rapidly evolving with LLM best practices

**Role in our system:** Handles the "R" in RAG—retrieval. When we need to find relevant medical research, LlamaIndex manages the vector search, result ranking, and context assembly.

**For non-technical readers:** This is the research assistant. When the AI needs information to answer a question, LlamaIndex searches through hundreds of medical papers and brings back the most relevant sections.

---

### Observability

#### Langfuse

**What it is:**  
An observability platform designed specifically for LLM applications—tracking what the AI does, how well it performs, and where issues occur.

**Why we chose it:**
- **LLM-native**: Understands the unique challenges of AI applications
- **Trace visualization**: See every step the agent took for any request
- **Quality metrics**: Track whether retrievals are relevant and responses are helpful
- **Latency breakdown**: Identify which steps are slow
- **Cost tracking**: Monitor token usage and API costs
- **Debugging**: Reproduce and investigate issues

**Role in our system:**
- Records every agent execution with full detail
- Tracks retrieval quality (did we find relevant research?)
- Monitors response quality over time
- Alerts on anomalies or errors

**For non-technical readers:** This is our quality control dashboard. It records everything the AI does so we can review it, find problems, and continuously improve. If something goes wrong, we can replay exactly what happened.

---

### Containerization and Orchestration

#### Docker & Docker Compose

**What it is:**  
Docker packages applications into containers—self-contained units that include everything needed to run the software. Docker Compose manages multi-container setups.

**Why we chose it:**
- **Consistency**: Same behavior on developer laptops and production servers
- **Isolation**: Each service runs independently
- **Reproducibility**: Anyone can spin up the entire system with one command
- **Cloud-ready**: Containers deploy directly to Cloud Run, Kubernetes, etc.
- **Industry standard**: Universal tooling and knowledge

**Role in our system:**
- Packages the FastAPI application
- Runs PostgreSQL + pgvector locally for development
- Enables one-command local setup: `docker-compose up`

**For non-technical readers:** Imagine shipping a complete office (desk, computer, files, everything) rather than just instructions for setting one up. Docker ensures that our application runs exactly the same way everywhere—no surprises.

---

### Testing and Quality

#### Pytest

**What it is:**  
The most popular Python testing framework.

**Why we chose it:**
- **Simple syntax**: Easy to write and read tests
- **Rich ecosystem**: Plugins for async testing, coverage, fixtures
- **Fast execution**: Parallel test running for quick feedback
- **Great reporting**: Clear output showing what passed/failed

**Role in our system:** Runs automated tests that verify the application works correctly—from API endpoints to individual functions.

---

#### Ruff

**What it is:**  
An extremely fast Python linter (code quality checker).

**Why we chose it:**
- **Speed**: 10-100x faster than traditional linters
- **Comprehensive**: Replaces multiple tools (flake8, isort, pyupgrade, etc.)
- **Auto-fix**: Automatically fixes many issues
- **Modern**: Understands latest Python features

**Role in our system:** Catches code style issues and potential bugs before they reach production.

---

#### MyPy

**What it is:**  
A static type checker for Python.

**Why we chose it:**
- **Catches bugs early**: Finds type mismatches before running code
- **Documentation**: Type hints serve as inline documentation
- **IDE support**: Enables better autocomplete and error detection
- **Confidence**: Especially important for healthcare applications

**Role in our system:** Verifies that functions receive and return the correct data types, preventing a class of bugs before they happen.

---

## Why This Combination?

The technologies were selected to work together as a cohesive system:

```
┌──────────────────────────────────────────────────────────────────┐
│                         DEVELOPER EXPERIENCE                      │
│  uv (fast deps) + Ruff (fast linting) + Pytest (fast tests)     │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                           APPLICATION                             │
│  FastAPI (async API) + LangGraph (agent) + LlamaIndex (RAG)      │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                          AI SERVICES                              │
│  Azure OpenAI (LLM + embeddings) + LlamaParser                   │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                              │
│  PostgreSQL + pgvector (unified storage and vector search)       │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY                             │
│  Langfuse (AI-native tracing, quality monitoring, debugging)     │
└──────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT                               │
│  Docker → Cloud Run / Kubernetes (scalable, production-ready)    │
└──────────────────────────────────────────────────────────────────┘
```

**Key principles:**
1. **Best-of-breed tools**: Azure for LLM and embeddings, LlamaCloud for document parsing - each service optimized for its purpose
2. **Python ecosystem**: All tools are Python-native for seamless integration
3. **Speed at every layer**: uv, Ruff, FastAPI, pgvector—all chosen for performance
4. **AI-native observability**: Langfuse understands LLM applications in ways generic tools don't
5. **Production-ready**: Every choice supports the path from development to production deployment
