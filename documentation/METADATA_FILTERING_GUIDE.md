# Metadata-Based Filtering Guide

## Overview

With the new Docling + Metadata Extraction pipeline, **every chunk now has rich medical metadata** stamped onto it:

```json
{
  "chunk_metadata": {
    "extracted_metadata": {
      "ethnicities": ["African / Black"],
      "diagnoses": ["PCOS", "Bacterial vaginosis"],
      "symptoms": ["Vaginal Odor", "Itchy"],
      "menstrual_status": ["Premenstrual"],
      "birth_control": ["IUD"],
      "hormone_therapy": [],
      "fertility_treatments": [],
      "age_mentioned": true,
      "age_range": "25-35",
      "confidence": 0.92
    },
    "table_summary": "Comparison of vaginal pH levels...",
    "section_type": "methods"
  }
}
```

## Architecture

```
┌────────────────────────────────────────────┐
│   Agent (LangGraph)                        │
│   - Decides what to search for             │
│   - Creates metadata filters dynamically   │
└────────────────┬───────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────┐
│   MedicalPaperRetriever (LlamaIndex)       │
│   - Accepts metadata filters               │
│   - Passes to VectorStore                  │
└────────────────┬───────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────┐
│   VectorStore + PostgreSQL                 │
│   - JSONB filtering on chunk_metadata      │
│   - Hybrid search (semantic + BM25)        │
└────────────────────────────────────────────┘
```

## Usage Examples

### Example 1: Simple Metadata Filter

```python
from medical_agent.rag.retriever import create_retriever, RetrievalConfig

# Agent wants papers about PCOS in African women
config = RetrievalConfig(
    top_k=10,
    filter_diagnoses=["Polycystic ovary syndrome (PCOS)"],
    filter_ethnicities=["African / Black"],
)

retriever = create_retriever(config=config)
results = await retriever._aretrieve(QueryBundle(query_str="pH levels and hormonal imbalance"))

# Returns ONLY chunks from papers that:
# 1. Mention PCOS (in extracted_metadata.diagnoses)
# 2. Mention African / Black (in extracted_metadata.ethnicities)
# 3. Are semantically similar to the query
```

### Example 2: Complex Multi-Category Filter

```python
# Agent investigating bacterial vaginosis with specific symptoms
config = RetrievalConfig(
    top_k=15,
    # Diagnosis
    filter_diagnoses=["Bacterial vaginosis"],

    # Symptoms
    filter_symptoms=["Vaginal Odor", "Gray", "Creamy"],

    # Birth control context
    filter_birth_control=["IUD", "Pill"],

    # Prefer certain section types
    chunk_types=[ChunkType.ABSTRACT, ChunkType.RESULTS],
)

retriever = create_retriever(config=config)
results = await retriever._aretrieve(QueryBundle(query_str="treatment options and outcomes"))
```

### Example 3: Ethnicity-Specific Research

```python
# Agent researching vaginal health in Hispanic population
config = RetrievalConfig(
    top_k=10,
    filter_ethnicities=["Hispanic / Latina"],

    # Optional: combine with other filters
    filter_symptoms=["Vaginal Dryness", "Burning"],
)

retriever = create_retriever(config=config)
```

### Example 4: Hormone-Related Queries

```python
# Agent exploring hormone therapy and vaginal pH
config = RetrievalConfig(
    top_k=10,
    filter_hormone_therapy=["HRT", "Estrogen"],
    filter_symptoms=["Vaginal Dryness"],
)

retriever = create_retriever(config=config)
```

## Agent Integration

Your LangGraph agent can now **dynamically construct metadata filters** based on user input:

```python
from medical_agent.agent.state import AgentState

def retriever_node(state: AgentState):
    """Agent node that decides what metadata to filter by."""

    # Extract user context
    user_profile = state.get("health_profile", {})
    query = state.get("query", "")

    # Build filters dynamically
    filters = {}

    # If user mentions ethnicity, filter by it
    if user_profile.get("ethnicity"):
        filters["filter_ethnicities"] = [user_profile["ethnicity"]]

    # If user mentions diagnosis
    if "PCOS" in query:
        filters["filter_diagnoses"] = ["Polycystic ovary syndrome (PCOS)"]

    # If user mentions symptoms
    if "odor" in query.lower():
        filters["filter_symptoms"] = ["Vaginal Odor"]

    # If user is on birth control
    if user_profile.get("birth_control"):
        filters["filter_birth_control"] = [user_profile["birth_control"]]

    # Create retriever with filters
    config = RetrievalConfig(top_k=10, **filters)
    retriever = create_retriever(config=config)

    # Retrieve relevant chunks
    results = await retriever._aretrieve(QueryBundle(query_str=query))

    return {"retrieved_chunks": results}
```

## How It Works Under the Hood

### 1. PostgreSQL JSONB Query

When you filter by `filter_diagnoses=["PCOS"]`, this generates:

```sql
SELECT * FROM paper_chunks
WHERE chunk_metadata->'extracted_metadata'->'diagnoses' ?| array['Polycystic ovary syndrome (PCOS)']
AND embedding <=> '[0.1, 0.2, ...]' < 0.5
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 10;
```

The `?|` operator checks if **ANY** of the values in the filter array exist in the JSONB array.

### 2. Empty Metadata Handling

Papers with **no metadata** for a category (empty arrays) will be **excluded** from results when that filter is applied:

```json
// Chunk A (will match filter_diagnoses=["PCOS"])
{
  "extracted_metadata": {
    "diagnoses": ["PCOS", "Endometriosis"]
  }
}

// Chunk B (will NOT match filter_diagnoses=["PCOS"])
{
  "extracted_metadata": {
    "diagnoses": []  // No diagnoses extracted
  }
}
```

This ensures **only papers with explicitly mentioned metadata** are returned.

### 3. Multiple Filters = AND Logic

When you combine multiple filters, they're combined with **AND** logic:

```python
config = RetrievalConfig(
    filter_diagnoses=["PCOS"],           # AND
    filter_symptoms=["Vaginal Odor"],    # AND
    filter_ethnicities=["African / Black"]  # AND
)
# Returns ONLY chunks that have ALL three metadata tags
```

### 4. OR Logic Within Each Filter

Within a single filter, values are combined with **OR** logic:

```python
config = RetrievalConfig(
    filter_ethnicities=["African / Black", "Asian"]  # OR
)
# Returns chunks that have EITHER ethnicity
```

## Benefits for Your Agent

### 1. Personalized Retrieval

```python
# User says: "I'm Hispanic with PCOS and use an IUD"
config = RetrievalConfig(
    filter_ethnicities=["Hispanic / Latina"],
    filter_diagnoses=["Polycystic ovary syndrome (PCOS)"],
    filter_birth_control=["IUD"],
)
# Agent retrieves ONLY papers relevant to this specific context
```

### 2. Higher Precision

Instead of semantic search alone (which might retrieve tangentially related papers), you now get:
- ✅ Semantically similar papers
- ✅ WITH the exact medical context the user needs

### 3. Better Citations

When the agent cites papers, it can say:
> "Based on research in African women with PCOS (Smith et al.), elevated pH levels are associated with..."

Instead of generic:
> "Based on research (Smith et al.), elevated pH levels are associated with..."

## Implementation Steps

### Step 1: Update VectorStore.similarity_search()

Add metadata filter support:

```python
# In src/medical_agent/ingestion/storage/vector_store.py

from medical_agent.ingestion.storage.metadata_filters import build_metadata_filters

async def similarity_search(self, session: AsyncSession, query: SearchQuery):
    # Build base query
    stmt = select(PaperChunk).where(...)

    # Add metadata filters
    metadata_filters = build_metadata_filters(
        filter_ethnicities=query.filter_ethnicities,
        filter_diagnoses=query.filter_diagnoses,
        filter_symptoms=query.filter_symptoms,
        # ... other filters
    )

    if metadata_filters:
        stmt = stmt.where(and_(*metadata_filters))

    # Execute query
    ...
```

### Step 2: Update MedicalPaperRetriever

Pass metadata filters from config to SearchQuery:

```python
# In src/medical_agent/rag/retriever.py

async def _semantic_search(self, query_bundle: QueryBundle):
    search_query = SearchQuery(
        embedding=query_embedding,
        top_k=self.config.fetch_k,
        # ... existing fields

        # NEW: Pass metadata filters
        filter_ethnicities=self.config.filter_ethnicities,
        filter_diagnoses=self.config.filter_diagnoses,
        filter_symptoms=self.config.filter_symptoms,
        # ... other filters
    )

    return await self.vector_store.similarity_search(session, search_query)
```

### Step 3: Use in Your Agent

```python
# In your agent nodes (src/medical_agent/agent/nodes/retriever.py)

def create_filtered_retriever(user_context: dict, query: str):
    """Create a retriever with metadata filters based on user context."""

    filters = {}

    # Extract filters from user context
    if user_context.get("ethnicity"):
        filters["filter_ethnicities"] = [user_context["ethnicity"]]

    if user_context.get("diagnoses"):
        filters["filter_diagnoses"] = user_context["diagnoses"]

    if user_context.get("symptoms"):
        filters["filter_symptoms"] = user_context["symptoms"]

    # Create retriever
    config = RetrievalConfig(top_k=10, **filters)
    return create_retriever(config=config)
```

## Testing

```python
# Test metadata filtering
import asyncio
from medical_agent.rag.retriever import create_retriever, RetrievalConfig

async def test_metadata_filtering():
    # Create filtered retriever
    config = RetrievalConfig(
        top_k=5,
        filter_diagnoses=["Bacterial vaginosis"],
        filter_symptoms=["Vaginal Odor"],
    )

    retriever = create_retriever(config=config)

    # Query
    results = await retriever._aretrieve(
        QueryBundle(query_str="treatment and outcomes")
    )

    # Verify all results have the metadata
    for result in results:
        metadata = result.node.metadata.get("chunk_metadata", {})
        extracted = metadata.get("extracted_metadata", {})

        print(f"Diagnoses: {extracted.get('diagnoses')}")
        print(f"Symptoms: {extracted.get('symptoms')}")
        print(f"Content: {result.node.text[:100]}...")
        print("---")

asyncio.run(test_metadata_filtering())
```

## Summary

✅ **You were right!** LlamaIndex is perfect for retrieval, not ingestion

✅ **Now with metadata extraction**, your agent can:
- Filter papers by ethnicity, diagnoses, symptoms, etc.
- Provide personalized recommendations
- Generate better, more relevant citations
- Combine semantic search with medical context

✅ **Next steps**:
1. Update `VectorStore.similarity_search()` to use `build_metadata_filters()`
2. Update `MedicalPaperRetriever._semantic_search()` to pass filters
3. Integrate into your LangGraph agent nodes
4. Test with real user queries

The infrastructure is ready - just connect the pieces!
