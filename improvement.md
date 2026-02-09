# Medical RAG Pipeline — Improvement Plan

> **Status**: Phase 1 ✅ COMPLETED (Feb 9, 2026) | Phase 1.5 ✅ COMPLETED | Phase 2-5 pending
> **Created**: February 2026
> **Baseline RAGAS Scores (k=5)**: Faithfulness 0.7476 | Context Recall 0.6857 | Context Precision 0.6000 | Answer Relevancy 0.8372 | Factual Correctness 0.3200
> **Target**: All metrics ≥ 0.85, Factual Correctness ≥ 0.70
>
> **Phase 1 Changes**:
> - ✅ Separated health context from search query (no more query dilution)
> - ✅ Added cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
> - ✅ Tuned hybrid search alpha to 0.3 (70% BM25, 30% vector)
> - ✅ Implemented retriever caching (singleton pattern)
>
> **Phase 1.5 Changes (Conversational Retrieval)**:
> - ✅ LLM-based initial query generation from form data
> - ✅ Conversational query rewriting with history for follow-ups
> - ✅ Integrated with LangGraph memory (loads chat history)
> - ✅ Uses gpt-4o-mini for cost-effective query enhancement (~$0.001/query)

---

## Table of Contents

1. [Current State & Root Cause Analysis](#1-current-state--root-cause-analysis)
2. [Phase 0 — Evaluation Foundation](#2-phase-0--evaluation-foundation)
3. [Phase 1 — Quick Retrieval Wins](#3-phase-1--quick-retrieval-wins)
4. [Phase 2 — Data Quality & Re-Ingestion](#4-phase-2--data-quality--re-ingestion)
5. [Phase 3 — Advanced Retrieval](#5-phase-3--advanced-retrieval)
6. [Phase 4 — Generation Quality](#6-phase-4--generation-quality)
7. [Phase 5 — Monitoring & Continuous Improvement](#7-phase-5--monitoring--continuous-improvement)
8. [Implementation Timeline](#8-implementation-timeline)
9. [Files Changed Per Phase](#9-files-changed-per-phase)
10. [Risk Mitigation](#10-risk-mitigation)
11. [References](#11-references)

---

## 1. Current State & Root Cause Analysis

### Architecture
```
LangGraph (orchestration) + LlamaIndex (retrieval/ingestion) + LangChain (generation)

Ingestion:  Docling PDF → HybridChunker(512 tokens) → LLM metadata → Azure embeddings(3072-dim) → pgvector
Retrieval:  PGVectorStore hybrid_search=True → BM25+vector fusion → raw top-k results
Generation: Azure gpt-4o → structured output (MedicalResponse) → inline [1][2] citations
Graph:      START → retrieve_node → generate_node → END (linear, no routing)
```

### Root Causes (Ranked by Impact)

| # | Root Cause | Impact | Evidence | File:Line |
|---|-----------|--------|----------|-----------|
| 1 | **Query dilution** | HIGH | Health context concatenated directly into embedding query. Irrelevant tokens shift the embedding vector away from the actual medical question. | `nodes.py:70` |
| 2 | **Noisy chunks in vector store** | HIGH | Bibliography sections, image placeholders (`[Figure X]`), equations, dots-only rows, headers/footers all stored and retrieved. These waste retrieval slots. | `pipeline.py` (no filtering) |
| 3 | **No reranking** | HIGH | Raw hybrid search scores used directly. Cross-encoder reranking typically boosts precision by 15-25%. | `llamaindex_retrieval.py:76-79` |
| 4 | **Metadata extracted but never used** | MEDIUM | 8 medical metadata fields extracted by LLM during ingestion but never used as retrieval filters. | `metadata.py` extracts, `llamaindex_retrieval.py` ignores |
| 5 | **Retriever rebuilt per call** | LOW | `build_retriever()` creates new PGVectorStore + VectorStoreIndex + embed_model on every query. ~200ms overhead. | `llamaindex_retrieval.py:94` |
| 6 | **Tiny evaluation set** | CRITICAL | Only 5 auto-generated questions. RAGAS scores are statistically unreliable at this sample size. Improvement can't be measured. | `evaluation/` |
| 7 | **Linear graph (no error recovery)** | MEDIUM | No hallucination check, no relevance grading, no query rewriting loop. | `graph.py:44-46` |
| 8 | **Hybrid alpha not tuned** | MEDIUM | Default alpha used (LlamaIndex default varies by version). Medical text benefits from BM25-heavy alpha ~0.3-0.4. | `llamaindex_retrieval.py:78` |

### Key Finding: More Context = Worse Scores

When we increased k from 2 to 5:
- **Faithfulness**: 0.90 → 0.75 (dropped 17%)
- **Context Precision**: 0.90 → 0.60 (dropped 33%)
- **Factual Correctness**: 0.42 → 0.32 (dropped 24%)
- **Context Recall**: 0.69 → 0.69 (unchanged — the relevant docs were already in top-2)

This proves the extra chunks (positions 3-5) are **noise that hurts generation** without helping recall. The fix is better chunks + reranking, not fewer results.

---

## 2. Phase 0 — Evaluation Foundation

> **Goal**: Build a reliable evaluation baseline before changing anything.
> **Why first**: Without reliable measurement, we can't tell if changes help or hurt.

### 0.1 Build Golden Test Set (30-50 questions)

Create a CSV file with manually curated question-answer pairs covering:

**Question categories** (aim for 6-8 per category):
- **Direct fact retrieval**: "What pH range indicates bacterial vaginosis?"
- **Multi-hop reasoning**: "How does PCOS affect vaginal pH during the luteal phase?"
- **Comparative**: "Compare the effectiveness of IUD vs pill on vaginal flora"
- **Negation/absence**: "Does the research mention any link between thyroid disorders and pH?"
- **Specificity test**: "What were the findings of the Smith et al. 2023 study on BV treatment?"
- **Out-of-scope**: "What is the treatment for lung cancer?" (should return "no relevant docs")

**CSV format** (`evaluation/testsets/golden_v1.csv`):
```csv
user_input,reference,category,difficulty
"What pH range is associated with bacterial vaginosis?","A vaginal pH above 4.5 is associated with bacterial vaginosis. Normal vaginal pH is typically between 3.8 and 4.5.",direct_fact,easy
"How does the menstrual cycle affect vaginal pH?","During menstruation, vaginal pH increases due to the alkaline nature of blood...",multi_hop,medium
```

**File**: `src/medical_agent/evaluation/testsets/golden_v1.csv`

### 0.2 Add Per-Question Diagnostic Logging

Enhance the evaluation runner to capture more detail:

**File**: `src/medical_agent/evaluation/run_evaluation.py`

Add to `pipeline_results`:
```python
pipeline_results.append({
    "question": question,
    "response": result["response"],
    "num_contexts": len(result["retrieved_contexts"]),
    "elapsed_ms": result["elapsed_ms"],
    "status": "success",
    # NEW diagnostic fields:
    "contexts_preview": [c[:200] for c in result["retrieved_contexts"]],
    "citations_used": result.get("citations", []),
    "retrieval_scores": [c.get("score", 0) for c in result.get("citations", [])],
    "category": row.get("category", "unknown"),
})
```

### 0.3 Run Baseline Evaluation

Run the golden test set against the current pipeline (no changes):
```bash
python -m medical_agent.evaluation.run_evaluation --testset evaluation/testsets/golden_v1.csv
```

Save this result as `eval_baseline.json` — all future changes will be measured against it.

### 0.4 Add Score Comparison Script

**New file**: `src/medical_agent/evaluation/compare_results.py`

Simple script that takes two eval JSON files and prints a diff table:
```
Metric               Baseline    Current     Delta
─────────────────────────────────────────────────
Faithfulness         0.7476      0.8200      +0.0724
Context Recall       0.6857      0.7500      +0.0643
Context Precision    0.6000      0.7800      +0.1800
Answer Relevancy     0.8372      0.8500      +0.0128
Factual Correctness  0.3200      0.5100      +0.1900
```

---

## 3. Phase 1 — Quick Retrieval Wins

> **Goal**: Fix retrieval without re-ingestion. Expected improvement: +10-15% on context metrics.
> **Estimated effort**: 1-2 days

### 1.1 Separate Health Context from Search Query

**Problem**: `nodes.py:70` concatenates health context into the embedding query:
```python
# CURRENT (BAD)
enhanced_query = f"{user_query}\n\nHealth Context:\n{health_context}"
```

This shifts the embedding vector. If user asks "What causes BV?", the actual embedded query becomes "What causes BV?\n\nHealth Context:\npH Value: 5.2\nAge: 28\nDiagnoses: PCOS..." — the embedding drifts toward PCOS/pH content instead of BV causes.

**Fix**: Use the raw user query for retrieval, pass health context only to generation.

**File**: `src/medical_agent/agents/nodes.py`

```python
def retrieve_node(state: MedicalAgentState) -> dict[str, Any]:
    """Retrieval node: search with clean query, pass context separately."""
    logger.info("Executing retrieve_node")

    if not state.get("messages"):
        return {"docs_text": "No query provided.", "citations": []}

    last_message = state["messages"][-1]
    user_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))

    # CHANGED: Use raw query for retrieval (no health context dilution)
    logger.info(f"Retrieving nodes for query: {user_query[:100]}...")
    nodes = retrieve_nodes(query=user_query, similarity_top_k=10)  # Over-retrieve for reranking

    docs_text, citations = format_retrieved_nodes(nodes)
    logger.info(f"Retrieved {len(citations)} citations")

    return {
        "docs_text": docs_text,
        "citations": citations,
    }
```

The `generate_node` already has access to `ph_value` and `health_profile` in state and builds its own `health_context` for the prompt — so no change needed there.

### 1.2 Add Cross-Encoder Reranking

**Why**: Cross-encoders score query-document pairs jointly (not independently like bi-encoders). They catch semantic relevance that embedding similarity misses. Industry standard shows 15-25% precision improvement.

**Strategy**: Over-retrieve k=15-20, rerank with cross-encoder, keep top-5.

**New file**: `src/medical_agent/agents/reranker.py`

```python
"""Cross-encoder reranker for retrieved nodes."""

import logging
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Module-level singleton (loaded once, ~200MB model)
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Get or create the cross-encoder reranker (singleton)."""
    global _reranker
    if _reranker is None:
        logger.info("Loading cross-encoder model: ms-marco-MiniLM-L-6-v2")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder loaded")
    return _reranker


def rerank_nodes(
    query: str,
    nodes: list[NodeWithScore],
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using a cross-encoder.

    Args:
        query: Original user query (not enhanced)
        nodes: Retrieved nodes from hybrid search
        top_k: Number of top results to keep after reranking

    Returns:
        Top-k nodes reranked by cross-encoder score
    """
    if not nodes:
        return nodes

    reranker = get_reranker()

    # Create query-document pairs for cross-encoder
    pairs = [(query, node.node.text) for node in nodes]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Attach reranker scores and sort
    for node, score in zip(nodes, scores):
        node.score = float(score)  # Replace hybrid score with reranker score

    # Sort by reranker score (descending) and keep top_k
    reranked = sorted(nodes, key=lambda n: n.score, reverse=True)[:top_k]

    logger.info(
        f"Reranked {len(nodes)} → {len(reranked)} nodes. "
        f"Top score: {reranked[0].score:.3f}, Bottom: {reranked[-1].score:.3f}"
    )

    return reranked
```

**Update `nodes.py`** to use reranker:
```python
from medical_agent.agents.reranker import rerank_nodes

def retrieve_node(state: MedicalAgentState) -> dict[str, Any]:
    # ... (get user_query as before)

    # Over-retrieve, then rerank
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)
    nodes = rerank_nodes(query=user_query, nodes=nodes, top_k=5)

    docs_text, citations = format_retrieved_nodes(nodes)
    return {"docs_text": docs_text, "citations": citations}
```

### 1.3 Tune Hybrid Search Alpha

**Problem**: LlamaIndex's `PGVectorStore` hybrid search uses a default alpha that may not be optimal for medical text. Alpha controls the blend: `0.0 = pure BM25`, `1.0 = pure vector`.

**Research finding**: For domain-specific medical text with precise terminology (BV, PCOS, pH), BM25 keyword matching is very valuable. Recommended: **alpha = 0.3** (70% BM25, 30% vector).

**File**: `src/medical_agent/agents/llamaindex_retrieval.py`

```python
return index.as_retriever(
    similarity_top_k=similarity_top_k,
    vector_store_query_mode="hybrid",
    alpha=0.3,  # NEW: BM25-heavy for medical terminology precision
)
```

### 1.4 Cache the Retriever (Singleton)

**Problem**: `build_retriever()` is called every query, recreating PGVectorStore, VectorStoreIndex, and AzureOpenAIEmbedding each time.

**Fix**: Module-level singleton.

**File**: `src/medical_agent/agents/llamaindex_retrieval.py`

```python
_retriever_cache = {}

def build_retriever(similarity_top_k: int = 5):
    """Build or return cached retriever."""
    cache_key = similarity_top_k
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    # ... existing build logic ...

    retriever = index.as_retriever(
        similarity_top_k=similarity_top_k,
        vector_store_query_mode="hybrid",
        alpha=0.3,
    )

    _retriever_cache[cache_key] = retriever
    return retriever
```

### 1.5 Run Evaluation After Phase 1

```bash
python -m medical_agent.evaluation.run_evaluation --testset evaluation/testsets/golden_v1.csv
python -m medical_agent.evaluation.compare_results results/eval_baseline.json results/eval_phase1.json
```

**Expected improvements**:
- Context Precision: 0.60 → 0.75+ (reranking + query fix)
- Faithfulness: 0.75 → 0.82+ (less noisy context)
- Answer Relevancy: 0.84 → 0.88+ (focused retrieval)

---

## 4. Phase 2 — Data Quality & Re-Ingestion

> **Goal**: Clean the vector store. Remove garbage, transform useful tables, add document context.
> **Estimated effort**: 3-5 days
> **Requires**: Re-ingestion of all papers after changes

### 2.0 Increase Chunk Size

**Problem**: Current chunk size is 512 tokens, optimized for smaller embedding models. With large 3072-dimensional Azure OpenAI embeddings, larger chunks are both feasible and beneficial.

**Why larger chunks work**:
- High-dimensional embeddings (3072-dim) can capture complex semantic relationships in longer texts
- Larger chunks reduce fragmentation — related information stays together instead of being split across multiple chunks
- Reduces vector store size and retrieval latency (fewer vectors to search)
- Better context for reranking — cross-encoders benefit from more surrounding text

**Recommendation**: Increase chunk size to **1024-1500 tokens**
- `1024 tokens` for general medical content (reasonable balance)
- `1500 tokens` for dense research papers with many citations and complex findings
- Keep `max_overlap` at 128 tokens (same as before) to preserve continuity

**File**: `src/medical_agent/ingestion/pipeline.py`

Update the `HybridChunker` initialization:
```python
from llama_index.core.node_parser import HybridChunker

# CHANGE FROM:
chunker = HybridChunker(chunk_size=512, chunk_overlap=128)

# CHANGE TO:
chunker = HybridChunker(chunk_size=1024, chunk_overlap=128)  # Tuned for 3072-dim embeddings
```

**Expected impact**:
- Vector store size reduction: ~30-40% fewer chunks
- Improved context recall: Less fragmentation means better semantic cohesion
- Faster retrieval: Fewer vectors to score during search
- Cost reduction: Fewer embeddings to generate/store (~$0.30 savings per re-ingestion)

---

### 2.1 Heuristic Chunk Filter (Pre-Embedding)

Add a filter after chunking but before embedding to discard low-quality chunks.

**New file**: `src/medical_agent/ingestion/chunk_filter.py`

```python
"""
Heuristic chunk quality filter.

Removes chunks that waste embedding budget and retrieval slots:
- Bibliography/reference sections
- Image/figure placeholders
- Equation-only chunks
- Dots-only or whitespace-heavy chunks
- Headers/footers
- Very short chunks with no medical content
"""

import re
import logging
from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

# Patterns for low-quality content
BIBLIOGRAPHY_PATTERNS = [
    r"^\s*\[\d+\]\s+[A-Z][a-z]+",           # [1] Author Name...
    r"^\s*\d+\.\s+[A-Z][a-z]+.*et al",       # 1. Author et al.
    r"^\s*references?\s*$",                     # "References" heading
    r"^\s*bibliography\s*$",                    # "Bibliography" heading
    r"doi\.org|DOI:\s*10\.",                    # DOI links
    r"PMID:\s*\d+",                             # PubMed IDs
]

NOISE_PATTERNS = [
    r"^\s*\[?fig(ure)?\s*\d+\]?",             # Figure references
    r"^\s*\[?table\s*\d+\]?",                  # Table references (caption only)
    r"^[\s\.·•\-_=]{10,}$",                    # Dots, bullets, lines
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",      # Page numbers
    r"^\s*\d+\s*$",                             # Standalone numbers
    r"^[^\w]*$",                                 # No word characters at all
]

HEADER_FOOTER_PATTERNS = [
    r"journal\s+of\s+\w+",                     # Journal headers
    r"copyright\s+©?\s*\d{4}",                 # Copyright notices
    r"all\s+rights\s+reserved",                # Rights notices
    r"downloaded\s+from",                       # Download notices
    r"accepted\s+\d{1,2}\s+\w+\s+\d{4}",     # Acceptance dates
]

# Compiled patterns
_bib_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BIBLIOGRAPHY_PATTERNS]
_noise_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]
_header_re = [re.compile(p, re.IGNORECASE) for p in HEADER_FOOTER_PATTERNS]


def is_low_quality_chunk(node: BaseNode, min_words: int = 15) -> bool:
    """
    Check if a chunk is low quality and should be filtered out.

    Args:
        node: The chunk node to evaluate
        min_words: Minimum word count threshold

    Returns:
        True if chunk should be REMOVED (is low quality)
    """
    text = node.get_content().strip()

    # Empty or very short
    if len(text) < 20 or len(text.split()) < min_words:
        return True

    # Check section type in metadata
    section = (node.metadata.get("dl_doc_hash", "") or "").lower()
    chunk_type = (node.metadata.get("chunk_type", "") or "").lower()

    if chunk_type in ("references", "bibliography", "acknowledgements"):
        return True

    # Bibliography patterns
    bib_matches = sum(1 for p in _bib_re if p.search(text))
    if bib_matches >= 2:  # Multiple bibliography indicators
        return True

    # Pure noise
    lines = text.strip().split("\n")
    noise_lines = sum(1 for line in lines if any(p.match(line.strip()) for p in _noise_re))
    if noise_lines / max(len(lines), 1) > 0.5:
        return True

    # Header/footer content (short text matching header patterns)
    if len(text.split()) < 30:
        if any(p.search(text) for p in _header_re):
            return True

    # High ratio of non-alphabetic characters (equations, symbols)
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return True

    return False


def filter_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Filter out low-quality chunks.

    Args:
        nodes: List of chunk nodes from parser

    Returns:
        Filtered list with garbage chunks removed
    """
    original_count = len(nodes)
    filtered = [n for n in nodes if not is_low_quality_chunk(n)]
    removed = original_count - len(filtered)

    if removed > 0:
        logger.info(f"Chunk filter: {original_count} → {len(filtered)} (removed {removed} low-quality chunks)")

    return filtered
```

### 2.2 LLM-Based Chunk Quality Scoring

For chunks that pass heuristic filters, use gpt-4o-mini to score medical informativeness (1-5 scale). Drop chunks scoring 1-2.

**Why LLM**: Heuristic filters catch obvious garbage. LLM catches subtler issues: chunks that are technically "text" but contain no useful medical information (e.g., "The study was approved by the ethics committee and all participants provided informed consent.").

**Add to `chunk_filter.py`**:

```python
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

QUALITY_PROMPT = """Score this medical research text chunk on a scale of 1-5 for medical informativeness:

1 = No medical information (admin text, acknowledgements, ethics statements, formatting artifacts)
2 = Minimal medical info (generic methodology, sample demographics without findings)
3 = Some medical info (general background, non-specific health claims)
4 = Good medical info (specific findings, data points, clinical observations)
5 = Excellent medical info (key findings, statistical results, clinical recommendations)

Text: {text}

Return ONLY the number (1-5):"""


async def score_chunks_with_llm(
    nodes: list[BaseNode],
    min_score: int = 3,
    batch_size: int = 10,
) -> list[BaseNode]:
    """
    Score chunks using gpt-4o-mini and filter by quality.

    Uses gpt-4o-mini for cost efficiency (~$0.15/1M input tokens).
    Processes in batches to respect rate limits.

    Args:
        nodes: Chunks that passed heuristic filter
        min_score: Minimum quality score to keep (1-5)
        batch_size: Number of chunks per LLM call

    Returns:
        Chunks scoring >= min_score
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    scored_nodes = []

    for node in nodes:
        text = node.get_content()[:1000]  # Truncate for cost
        try:
            response = await llm.ainvoke(QUALITY_PROMPT.format(text=text))
            score = int(response.content.strip())
            if score >= min_score:
                node.metadata["quality_score"] = score
                scored_nodes.append(node)
            else:
                logger.debug(f"Dropping chunk (score={score}): {text[:80]}...")
        except Exception as e:
            logger.warning(f"LLM scoring failed, keeping chunk: {e}")
            scored_nodes.append(node)  # Keep on error

    logger.info(f"LLM scoring: {len(nodes)} → {len(scored_nodes)} chunks (min_score={min_score})")
    return scored_nodes
```

**Cost estimate**: ~30 papers × ~50 chunks/paper × ~200 tokens/chunk = 300K tokens. At gpt-4o-mini rates (~$0.15/1M input), total cost ≈ $0.05.

### 2.3 Table-to-Natural-Language Conversion

**Problem**: Docling extracts tables as structured data, but table content becomes hard to search via embeddings. Queries like "what were the BV treatment outcomes?" won't match a table with columns "Treatment | n | Cure Rate | p-value".

**Solution**: Use LLM to convert each table into natural language sentences that are embeddable and searchable.

**New file**: `src/medical_agent/ingestion/table_transformer.py`

```python
"""Convert table chunks to natural language for better embedding search."""

import logging
from llama_index.core.schema import BaseNode, TextNode
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

TABLE_PROMPT = """Convert this research table data into clear, factual natural language sentences.
Each sentence should be a standalone fact that could answer a research question.
Preserve ALL numbers, percentages, p-values, and statistical data exactly.
Include the context (what the table is about) in each sentence.

Table data:
{table_text}

Write 2-5 factual sentences summarizing the key data points:"""


async def transform_table_chunks(nodes: list[BaseNode]) -> list[BaseNode]:
    """
    Find table chunks and convert them to natural language.

    Replaces table text with LLM-generated natural language while
    preserving original table data in metadata for reference.

    Args:
        nodes: All chunk nodes from parser

    Returns:
        Nodes with table chunks transformed to natural language
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    result_nodes = []

    for node in nodes:
        chunk_type = (node.metadata.get("chunk_type", "") or "").lower()
        doc_items = node.metadata.get("doc_items", [])
        is_table = chunk_type == "table" or (
            doc_items and doc_items[0].get("type") == "table"
        )

        if is_table:
            table_text = node.get_content()
            try:
                response = await llm.ainvoke(TABLE_PROMPT.format(table_text=table_text))
                natural_text = response.content.strip()

                # Create new node with natural language text
                new_node = TextNode(
                    text=natural_text,
                    metadata={
                        **node.metadata,
                        "original_table_text": table_text[:500],  # Keep reference
                        "table_transformed": True,
                    },
                    id_=node.node_id,
                )
                result_nodes.append(new_node)
                logger.info(f"Transformed table chunk: {table_text[:80]}... → {natural_text[:80]}...")
            except Exception as e:
                logger.warning(f"Table transform failed, keeping original: {e}")
                result_nodes.append(node)
        else:
            result_nodes.append(node)

    table_count = sum(1 for n in result_nodes if n.metadata.get("table_transformed"))
    logger.info(f"Table transform: {table_count} tables converted to natural language")
    return result_nodes
```

### 2.4 Contextual Chunking (Anthropic's Approach)

**Problem**: Individual chunks lack document-level context. A chunk saying "The treatment group showed 85% improvement" doesn't tell the retriever what treatment, what condition, or what study.

**Solution**: Prepend a short document context summary to each chunk before embedding. This is Anthropic's "contextual retrieval" technique that reduced retrieval failures by 49% (combined with hybrid search).

**New file**: `src/medical_agent/ingestion/contextual_chunking.py`

```python
"""
Contextual chunking: prepend document context to each chunk.

Based on Anthropic's contextual retrieval research (2024):
- Generates a short context for each chunk using the full document
- Prepends context to chunk text before embedding
- Reduces retrieval failures by 49% when combined with hybrid search
"""

import logging
from llama_index.core.schema import BaseNode, TextNode
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{doc_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


async def add_contextual_headers(
    nodes: list[BaseNode],
    full_document_text: str,
) -> list[BaseNode]:
    """
    Add document-level context to each chunk.

    Calls gpt-4o-mini once per chunk to generate a 2-3 sentence
    context prefix that situates the chunk within the document.

    Args:
        nodes: Chunk nodes from parser
        full_document_text: Complete document text (truncated to first 8000 chars for context window)

    Returns:
        Nodes with contextual headers prepended to text
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Truncate document for context window
    doc_context = full_document_text[:8000]

    result_nodes = []

    for node in nodes:
        chunk_text = node.get_content()
        try:
            response = await llm.ainvoke(
                CONTEXT_PROMPT.format(doc_text=doc_context, chunk_text=chunk_text)
            )
            context = response.content.strip()

            # Prepend context to chunk
            contextualized_text = f"{context}\n\n{chunk_text}"

            new_node = TextNode(
                text=contextualized_text,
                metadata={
                    **node.metadata,
                    "contextual_header": context,
                },
                id_=node.node_id,
            )
            result_nodes.append(new_node)

        except Exception as e:
            logger.warning(f"Contextual header failed, using original: {e}")
            result_nodes.append(node)

    logger.info(f"Added contextual headers to {len(result_nodes)} chunks")
    return result_nodes
```

**Cost estimate**: ~30 papers × 50 chunks × ~300 tokens/call = 450K tokens ≈ $0.07 (gpt-4o-mini).

### 2.5 Semantic Deduplication

**Problem**: Multiple papers may cite the same foundational research, or Docling may create overlapping chunks from section boundaries. Near-duplicate chunks waste retrieval slots.

**Add to pipeline** (post-embedding, using cosine similarity):

```python
"""Semantic deduplication using embedding cosine similarity."""

import numpy as np
from llama_index.core.schema import BaseNode

def deduplicate_nodes(
    nodes: list[BaseNode],
    similarity_threshold: float = 0.95,
) -> list[BaseNode]:
    """
    Remove near-duplicate chunks using cosine similarity.

    Args:
        nodes: Nodes with embeddings already computed
        similarity_threshold: Cosine similarity above which chunks are duplicates

    Returns:
        Deduplicated node list (keeps first occurrence)
    """
    if len(nodes) <= 1:
        return nodes

    # Get embeddings
    embeddings = [n.embedding for n in nodes if n.embedding]
    if not embeddings:
        return nodes

    emb_matrix = np.array(embeddings)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    normalized = emb_matrix / (norms + 1e-10)

    keep_indices = set(range(len(nodes)))

    for i in range(len(normalized)):
        if i not in keep_indices:
            continue
        for j in range(i + 1, len(normalized)):
            if j not in keep_indices:
                continue
            sim = np.dot(normalized[i], normalized[j])
            if sim >= similarity_threshold:
                keep_indices.discard(j)  # Remove later duplicate

    deduped = [nodes[i] for i in sorted(keep_indices)]
    removed = len(nodes) - len(deduped)
    if removed > 0:
        logger.info(f"Deduplication: {len(nodes)} → {len(deduped)} (removed {removed} near-duplicates)")

    return deduped
```

### 2.6 Update Ingestion Pipeline

Integrate all Phase 2 components into `pipeline.py`:

**File**: `src/medical_agent/ingestion/pipeline.py`

The pipeline flow becomes:
```
PDF → Docling Parse → DoclingNodeParser + HybridChunker
    → Heuristic Filter (remove garbage)
    → Table-to-NL Transform (convert tables)
    → Contextual Headers (add document context)
    → LLM Quality Scoring (drop score < 3)
    → Metadata Extraction
    → Embedding Generation
    → Semantic Deduplication
    → Store in pgvector
```

Key change in `process_paper()`: After `self.pipeline.arun()`, add the new steps. Or better, create a custom `IngestionPipeline` subclass that includes the filters.

### 2.7 Re-Ingest All Papers

After pipeline changes:
1. Clear existing chunks: `TRUNCATE data_paper_chunks;`
2. Reset paper processing status: `UPDATE papers SET is_processed = false, processed_at = null;`
3. Re-run ingestion for all papers
4. Run evaluation to measure improvement

### 2.8 Run Evaluation After Phase 2

```bash
python -m medical_agent.evaluation.run_evaluation --testset evaluation/testsets/golden_v1.csv
python -m medical_agent.evaluation.compare_results results/eval_baseline.json results/eval_phase2.json
```

**Expected improvements**:
- Context Recall: 0.69 → 0.80+ (contextual headers help retriever find relevant content)
- Faithfulness: 0.75 → 0.85+ (less noise, better chunks)
- Factual Correctness: 0.32 → 0.50+ (table-to-NL makes data searchable)

---

## 5. Phase 3 — Advanced Retrieval

> **Goal**: Smart query handling and metadata-filtered retrieval.
> **Estimated effort**: 2-3 days

### 3.1 Metadata-Filtered Retrieval

**Problem**: We extract 8 medical metadata fields (diagnoses, symptoms, menstrual_status, etc.) during ingestion but never use them.

**Solution**: When the user's health profile contains specific values (e.g., diagnoses=["PCOS"]), use them as metadata filters to narrow retrieval before vector search.

**File**: `src/medical_agent/agents/llamaindex_retrieval.py`

```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator

def build_metadata_filters(health_profile: dict) -> MetadataFilters | None:
    """
    Build metadata filters from user health profile.

    Only filters on fields that have specific values and match
    the extracted metadata fields in chunk metadata.
    """
    filters = []

    # Map health profile fields to metadata fields
    if diagnoses := health_profile.get("diagnoses"):
        for d in diagnoses:
            filters.append(MetadataFilter(
                key="diagnoses",
                value=d,
                operator=FilterOperator.CONTAINS,
            ))

    if symptoms := health_profile.get("symptoms"):
        symptom_list = symptoms if isinstance(symptoms, list) else []
        if isinstance(symptoms, dict):
            symptom_list = [v for vals in symptoms.values() for v in (vals if isinstance(vals, list) else [vals])]
        for s in symptom_list:
            filters.append(MetadataFilter(
                key="symptoms",
                value=s,
                operator=FilterOperator.CONTAINS,
            ))

    if menstrual_cycle := health_profile.get("menstrual_cycle"):
        filters.append(MetadataFilter(
            key="menstrual_status",
            value=menstrual_cycle,
            operator=FilterOperator.CONTAINS,
        ))

    if not filters:
        return None

    # Use OR logic — retrieve docs matching ANY filter
    return MetadataFilters(filters=filters, condition="or")


def retrieve_nodes(
    query: str,
    similarity_top_k: int = 5,
    health_profile: dict | None = None,
) -> list[NodeWithScore]:
    """Retrieve with optional metadata filtering."""
    retriever = build_retriever(similarity_top_k=similarity_top_k)

    metadata_filters = None
    if health_profile:
        metadata_filters = build_metadata_filters(health_profile)

    if metadata_filters:
        # Use filtered retrieval
        nodes = retriever.retrieve(query, filters=metadata_filters)
        if len(nodes) < 3:
            # Fall back to unfiltered if too few results
            logger.info("Metadata filter too restrictive, falling back to unfiltered")
            nodes = retriever.retrieve(query)
    else:
        nodes = retriever.retrieve(query)

    return nodes
```

### 3.2 Query Decomposition for Complex Questions

**Problem**: Complex questions like "How does PCOS affect vaginal pH during the luteal phase compared to women without PCOS?" may not retrieve well as a single query.

**Solution**: Use LLM to decompose complex queries into sub-queries, retrieve for each, then merge results.

**New file**: `src/medical_agent/agents/query_decomposer.py`

```python
"""Query decomposition for complex medical questions."""

import logging
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


class DecomposedQuery(BaseModel):
    """Decomposed query output."""
    sub_queries: list[str] = Field(
        description="List of 1-3 simpler sub-queries. If the query is already simple, return it as-is in a single-element list."
    )
    needs_decomposition: bool = Field(
        description="True if the original query was complex enough to warrant decomposition."
    )


DECOMPOSE_PROMPT = """You are a medical research query optimizer.

Given a user question about women's health / vaginal pH, determine if it should be
decomposed into simpler sub-queries for better document retrieval.

Rules:
- Simple factual questions → return as-is (needs_decomposition=false)
- Complex multi-part questions → split into 2-3 focused sub-queries
- Each sub-query should be independently searchable
- Preserve medical terminology exactly

Question: {query}"""


async def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into simpler sub-queries.

    Returns the original query as-is if decomposition isn't needed.
    """
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(DecomposedQuery)

    try:
        result = structured_llm.invoke(DECOMPOSE_PROMPT.format(query=query))
        if result.needs_decomposition and len(result.sub_queries) > 1:
            logger.info(f"Decomposed query into {len(result.sub_queries)} sub-queries")
            return result.sub_queries
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")

    return [query]  # Return original if decomposition fails or isn't needed
```

### 3.3 Agentic RAG Graph (Document Grading + Query Rewriting)

**Problem**: Current graph is linear (retrieve → generate). No feedback loop if retrieved docs are irrelevant.

**Solution**: Add a document grading node and optional query rewriting loop.

**File**: `src/medical_agent/agents/graph.py` (update)

New graph flow:
```
START → retrieve → grade_documents ─── relevant ──→ generate → END
                         │
                         └── not relevant → rewrite_query → retrieve (max 1 retry)
```

**New node in `nodes.py`**:

```python
class RelevanceGrade(BaseModel):
    """Grade whether retrieved documents are relevant to the query."""
    is_relevant: bool = Field(description="True if documents contain information relevant to answering the query")
    reason: str = Field(description="Brief reason for the grade")


def grade_documents_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Grade whether retrieved documents are relevant.

    Uses LLM to quickly assess if retrieved chunks can answer the question.
    If not, sets a flag for the router to trigger query rewriting.
    """
    docs_text = state.get("docs_text", "")
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""

    if not docs_text or docs_text == "No relevant medical research documents found.":
        return {"docs_relevant": False, "retry_count": state.get("retry_count", 0)}

    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    structured_llm = llm.with_structured_output(RelevanceGrade)

    prompt = f"""Does the following retrieved context contain information relevant to answering this question?

Question: {user_query}

Context (first 1000 chars): {docs_text[:1000]}

Grade relevance:"""

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "docs_relevant": result.is_relevant,
        "retry_count": state.get("retry_count", 0),
    }
```

**Updated graph**:
```python
def route_after_grading(state: MedicalAgentState) -> str:
    """Route based on document relevance grade."""
    if state.get("docs_relevant", True):
        return "generate"
    if state.get("retry_count", 0) >= 1:
        return "generate"  # Give up after 1 retry, generate with what we have
    return "rewrite_query"


workflow = StateGraph(MedicalAgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grading)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)
```

### 3.4 pgvector HNSW Index Tuning

For 3072-dimensional vectors, the default HNSW parameters may not be optimal.

**SQL migration** (run once):
```sql
-- Drop existing index
DROP INDEX IF EXISTS ix_data_paper_chunks_embedding;

-- Create optimized HNSW index for 3072-dim vectors
CREATE INDEX ix_data_paper_chunks_embedding
ON data_paper_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);

-- Set search-time parameter (per-session)
SET hnsw.ef_search = 150;
```

**Parameters**:
- `m = 32` (default 16): More connections per node → better recall for high-dim vectors
- `ef_construction = 200` (default 64): More candidates during build → better index quality
- `ef_search = 150` (default 40): More candidates during search → better recall

### 3.5 Run Evaluation After Phase 3

**Expected improvements**:
- Context Recall: 0.80 → 0.88+ (metadata filters + query decomposition)
- Context Precision: 0.75 → 0.85+ (document grading removes irrelevant results)
- Faithfulness: 0.85 → 0.90+ (only relevant docs reach generation)

---

## 6. Phase 4 — Generation Quality

> **Goal**: Improve factual correctness and faithfulness of generated answers.
> **Estimated effort**: 1-2 days

### 4.1 Citation-Claim Structured Output

**Problem**: Current `MedicalResponse` has `response` (free text) + `used_citations` (list of ints). The LLM can claim to cite [1] without actually using information from citation [1]. This hurts Factual Correctness.

**Solution**: Force the LLM to output per-claim citations with exact quotes from the source.

**File**: `src/medical_agent/agents/nodes.py`

```python
class CitationClaim(BaseModel):
    """A single claim with its supporting citation."""
    claim: str = Field(description="A single factual claim from the medical documents")
    citation_number: int = Field(description="The citation number [1], [2], etc. supporting this claim")
    supporting_quote: str = Field(description="The exact quote from the cited document that supports this claim (max 100 chars)")


class MedicalResponse(BaseModel):
    """Structured output with per-claim citations for verifiable responses."""
    claims: list[CitationClaim] = Field(
        description="List of factual claims, each with a specific citation and supporting quote"
    )
    summary: str = Field(
        description="A 1-2 sentence summary synthesizing the claims. Use inline [1], [2] references."
    )
    used_citations: list[int] = Field(
        description="List of all citation numbers referenced in claims"
    )
```

Then in `generate_node`, reconstruct the response from claims:
```python
# Build response from claims
claim_texts = [f"- {c.claim} [{c.citation_number}]" for c in result.claims]
response_text = result.summary + "\n\n" + "\n".join(claim_texts)
```

This forces grounded responses because each claim must have a source quote.

### 4.2 Hallucination Check Node (Optional)

Add a lightweight post-generation check using gpt-4o-mini:

```python
def check_hallucination_node(state: MedicalAgentState) -> dict[str, Any]:
    """Check if the response is grounded in retrieved context."""
    response = state["messages"][-1].content
    docs_text = state.get("docs_text", "")

    # Use gpt-4o-mini for cheap hallucination checking
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_mini_deployment_name,
        # ...
    )

    prompt = f"""Is this response fully grounded in the provided documents?

Response: {response}

Documents: {docs_text[:2000]}

Answer YES if every claim in the response is supported by the documents.
Answer NO if the response contains information not found in the documents.
Answer:"""

    result = llm.invoke([HumanMessage(content=prompt)])
    is_grounded = "YES" in result.content.upper()

    return {"is_grounded": is_grounded}
```

### 4.3 Prompt Engineering Refinements

**File**: `src/medical_agent/agents/nodes.py` (system_prompt in generate_node)

Key additions to the prompt:
```
8. When citing, quote a SHORT phrase from the source to prove grounding (e.g., "pH levels above 4.5 are associated with BV [1]")
9. NEVER start your response with "Based on the documents" or "According to the research" — be direct
10. If multiple sources disagree, note the disagreement explicitly
11. For statistical findings, include the exact numbers (p-values, confidence intervals, percentages)
```

### 4.4 Run Evaluation After Phase 4

**Expected improvements**:
- Factual Correctness: 0.50 → 0.70+ (per-claim citations force grounding)
- Faithfulness: 0.90 → 0.95+ (hallucination check catches drift)
- Answer Relevancy: 0.88 → 0.92+ (prompt improvements)

---

## 7. Phase 5 — Monitoring & Continuous Improvement

> **Goal**: Track quality over time, catch regressions, expand test coverage.

### 5.1 Automated Regression Testing

Add a CI-friendly evaluation script that fails if scores drop below thresholds:

```python
# evaluation/regression_check.py
THRESHOLDS = {
    "faithfulness": 0.85,
    "context_recall": 0.80,
    "context_precision": 0.80,
    "response_relevancy": 0.85,
    "factual_correctness": 0.65,
}

def check_regression(results_path: str) -> bool:
    """Return True if all metrics meet thresholds."""
    # ... load results and compare against thresholds
```

### 5.2 LangSmith Dashboard Monitoring

Already partially configured (`ragas_config.py:78-103`). Enhance with:
- Custom tags per evaluation run (e.g., `phase=1`, `change=reranking`)
- Latency tracking per node (retrieve vs generate)
- Token usage tracking

### 5.3 Expand Golden Test Set

After each phase, add questions that specifically test the changes:
- Phase 1: "Does BV affect fertility?" (tests query separation)
- Phase 2: "What were the cure rates in Table 3?" (tests table-to-NL)
- Phase 3: "I have PCOS and BV, how do they interact?" (tests metadata filtering)
- Phase 4: "Quote the exact finding about pH and menstrual cycle" (tests citation grounding)

---

## 8. Implementation Timeline

```
Week 1:  Phase 0 (golden test set + baseline) + Phase 1 (quick wins)
Week 2:  Phase 2 (chunk quality + re-ingestion)
Week 3:  Phase 3 (advanced retrieval + agentic graph)
Week 4:  Phase 4 (generation quality) + Phase 5 (monitoring)
```

Each phase ends with an evaluation run. If scores don't improve, debug before moving on.

---

## 9. Files Changed Per Phase

### Phase 0
| File | Action |
|------|--------|
| `evaluation/testsets/golden_v1.csv` | CREATE |
| `evaluation/run_evaluation.py` | EDIT (add diagnostic fields) |
| `evaluation/compare_results.py` | CREATE |

### Phase 1
| File | Action |
|------|--------|
| `agents/nodes.py` | EDIT (separate query from context, add reranker call) |
| `agents/reranker.py` | CREATE |
| `agents/llamaindex_retrieval.py` | EDIT (cache retriever, add alpha=0.3) |

### Phase 2
| File | Action |
|------|--------|
| `ingestion/chunk_filter.py` | CREATE |
| `ingestion/table_transformer.py` | CREATE |
| `ingestion/contextual_chunking.py` | CREATE |
| `ingestion/pipeline.py` | EDIT (integrate new stages) |

### Phase 3
| File | Action |
|------|--------|
| `agents/llamaindex_retrieval.py` | EDIT (metadata filters) |
| `agents/query_decomposer.py` | CREATE |
| `agents/graph.py` | EDIT (agentic routing) |
| `agents/nodes.py` | EDIT (grade_documents_node, rewrite_query_node) |
| `agents/state.py` | EDIT (add docs_relevant, retry_count) |

### Phase 4
| File | Action |
|------|--------|
| `agents/nodes.py` | EDIT (CitationClaim model, prompt updates) |

### Phase 5
| File | Action |
|------|--------|
| `evaluation/regression_check.py` | CREATE |

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Re-ingestion breaks existing data | Backup database before Phase 2. Use separate table for testing. |
| Cross-encoder model too large for deployment | ms-marco-MiniLM-L-6-v2 is only ~80MB. Alternative: use Cohere rerank API. |
| LLM calls in pipeline increase latency | gpt-4o-mini adds ~200ms per call. Pipeline calls are at ingestion time (offline), not query time. |
| LLM scoring costs | gpt-4o-mini: ~$0.15/1M input tokens. Full re-ingestion ≈ $0.15 total. |
| Agentic graph adds query latency | Document grading via gpt-4o-mini adds ~300ms. Query rewrite loop adds another ~500ms when triggered. Cap at 1 retry. |
| Alpha tuning wrong for some queries | Log alpha effectiveness per query in Phase 5. Consider adaptive alpha based on query type. |

---

## 11. References

1. **Anthropic Contextual Retrieval** (2024) — Prepending chunk context reduced retrieval failures by 49% with hybrid search
2. **RAGAS Documentation** — ragas.io — Evaluation metrics for RAG pipelines
3. **Cross-Encoder Reranking** — SBERT docs — ms-marco-MiniLM-L-6-v2 for passage reranking
4. **pgvector HNSW Tuning** — pgvector GitHub — Index parameter recommendations for high-dimensional vectors
5. **LlamaIndex Hybrid Search** — docs.llamaindex.ai — PGVectorStore alpha parameter for BM25/vector fusion
6. **Docling HybridChunker** — ds4sd/docling — Token-aware section chunking
7. **LangGraph Agentic RAG** — LangGraph documentation — Corrective RAG patterns with document grading
8. **Adaptive RAG** — Arxiv 2024 — Query routing between simple/complex retrieval strategies
