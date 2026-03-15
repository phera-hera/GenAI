# Phase 1: Metadata-Weighted Reranking Implementation Guide

## Overview

**Goal**: Enhance retrieval ranking by blending cross-encoder relevance (70%) with metadata overlap from user's health profile (30%).

**Current state**: Retrieve 15 candidates → rerank with cross-encoder only → keep top 5

**New state**: Retrieve 15 candidates → rerank with 70% cross-encoder + 30% metadata overlap → keep top 5

**Why**: Medical queries benefit from patient-context alignment. A paper discussing PCOS should rank higher for a user with PCOS diagnosis.

---

## Architecture Overview

### Current Flow
```
retrieve_node:
  Step 1: retrieve_nodes(query, 15)
    └─ LlamaIndex hybrid search

  Step 2: rerank_nodes(query, 15→5)
    └─ Cross-encoder only (ms-marco-MiniLM-L-6-v2)
       Output: scores 0-5, normalized to 0-1

  Step 3: format_retrieved_nodes(5)
    └─ Convert to citation text
```

### New Flow (Phase 1)
```
retrieve_node:
  Step 1: retrieve_nodes(query, 15)
    └─ LlamaIndex hybrid search

  Step 2: rerank_nodes_with_metadata(query, 15→5, health_profile) ← NEW
    ├─ Get cross-encoder scores: 0-1 range
    ├─ Get metadata overlap scores: 0-1 range
    └─ final_score = (0.7 × cross_encoder_score) + (0.3 × metadata_overlap_score)

  Step 3: format_retrieved_nodes(5)
    └─ Convert to citation text
```

### Metadata Overlap Calculation

**Example**:
```
User Health Profile:
  diagnoses: [PCOS, Yeast infection]
  symptoms: [Itchy, Yellow discharge]
  birth_control: [IUD]
  ethnicities: [Asian]

Paper 1 Metadata (stamped during ingestion):
  diagnoses: [PCOS]
  symptoms: [Itchy]
  birth_control: [IUD]
  ethnicities: [Asian]

Paper 2 Metadata:
  diagnoses: [Adenomyosis]
  symptoms: [Pelvic pain]
  birth_control: []
  ethnicities: []

Metadata Overlap Calculation (Jaccard per field):

Paper 1:
  diagnoses: intersection=1 (PCOS), union=2 → score=0.5
  symptoms: intersection=1 (Itchy), union=3 → score=0.33
  birth_control: intersection=1 (IUD), union=1 → score=1.0
  ethnicities: intersection=1 (Asian), union=1 → score=1.0
  Average: (0.5 + 0.33 + 1.0 + 1.0) / 4 = 0.71

Paper 2:
  diagnoses: intersection=0, union=3 → score=0.0
  symptoms: intersection=0, union=3 → score=0.0
  birth_control: intersection=0, union=1 → score=0.0
  ethnicities: intersection=0, union=1 → score=0.0
  Average: 0.0

Final Score (if Paper 1 has cross-encoder 0.85, Paper 2 has 0.80):
  Paper 1: (0.7 × 0.85) + (0.3 × 0.71) = 0.595 + 0.213 = 0.808
  Paper 2: (0.7 × 0.80) + (0.3 × 0.0) = 0.56 + 0 = 0.56

→ Paper 1 ranked higher due to metadata alignment
```

---

## Files to Modify

| File | Action | Summary |
|------|--------|---------|
| `src/medical_agent/agents/reranker.py` | ADD 2 functions | Metadata overlap calculation + weighted reranking |
| `src/medical_agent/agents/nodes.py` | MODIFY retrieve_node() | Call new reranking function |

---

## Step-by-Step Implementation

### STEP 1: Add metadata overlap scoring function to reranker.py

**File**: `src/medical_agent/agents/reranker.py`

**Location**: After the `get_reranker()` function (after line 20)

**Code to add**:

```python
def compute_metadata_overlap_score(
    user_profile: dict[str, Any],
    paper_metadata: dict[str, Any]
) -> float:
    """
    Compute overlap between user health profile and paper metadata.

    Uses Jaccard similarity (intersection / union) for each metadata field,
    then averages across all fields to get a 0.0-1.0 overlap score.

    Args:
        user_profile: User's health profile dict with keys like:
            diagnoses, symptoms, birth_control, hormone_therapy,
            ethnicities, menstrual_status, fertility_treatments
        paper_metadata: Paper's extracted metadata dict (same structure)

    Returns:
        float: Overlap score 0.0 (no match) to 1.0 (perfect match)

    Example:
        >>> user = {"diagnoses": ["PCOS"], "symptoms": ["Itchy"]}
        >>> paper = {"diagnoses": ["PCOS"], "symptoms": ["Itchy", "Yellow"]}
        >>> compute_metadata_overlap_score(user, paper)
        0.75  # Assuming equal weighting across fields
    """
    if not user_profile or not paper_metadata:
        return 0.0

    # Metadata fields extracted during ingestion (from metadata.py)
    fields_to_check = [
        "diagnoses",
        "symptoms",
        "birth_control",
        "hormone_therapy",
        "ethnicities",
        "menstrual_status",
        "fertility_treatments",
    ]

    field_scores = []

    for field in fields_to_check:
        # Get values from both user profile and paper metadata
        user_values = set(user_profile.get(field, []))
        paper_values = set(paper_metadata.get(field, []))

        # Skip if both are empty (field not relevant)
        if not user_values and not paper_values:
            continue

        # If only one has values, no overlap for this field
        if not user_values or not paper_values:
            field_scores.append(0.0)
            continue

        # Jaccard similarity: |intersection| / |union|
        intersection = len(user_values & paper_values)
        union = len(user_values | paper_values)
        jaccard_score = intersection / union if union > 0 else 0.0
        field_scores.append(jaccard_score)

    # Average across all fields (return 0.0 if no fields to check)
    return (
        sum(field_scores) / len(field_scores) if field_scores else 0.0
    )
```

**Key design notes**:
- Handles empty health profile gracefully (returns 0.0)
- Uses Jaccard similarity (standard, interpretable)
- Skips fields where both user and paper have no data
- Averages equally across all 7 fields
- Ignores `age` field (less useful for matching)

---

### STEP 2: Add weighted reranking function to reranker.py

**File**: `src/medical_agent/agents/reranker.py`

**Location**: After the `rerank_nodes()` function (after line 62)

**Code to add**:

```python
def rerank_nodes_with_metadata(
    query: str,
    nodes: list[NodeWithScore],
    user_profile: dict[str, Any],
    top_k: int = 5,
) -> list[NodeWithScore]:
    """
    Rerank retrieved nodes using BOTH cross-encoder relevance AND metadata overlap.

    Blends two signals:
    - 70% Cross-encoder relevance (ms-marco-MiniLM-L-6-v2)
    - 30% Metadata overlap with user's health profile

    Process:
    1. Get cross-encoder scores for all nodes
    2. Normalize cross-encoder scores to 0-1 range
    3. Compute metadata overlap scores (0-1)
    4. Blend: final_score = (0.7 × ce_score) + (0.3 × metadata_score)
    5. Sort by final score and keep top-k

    Args:
        query: Original user query (not enhanced)
        nodes: Retrieved nodes from hybrid search (typically 15)
        user_profile: User's health profile dict with metadata fields
        top_k: Number of top results to keep after reranking (default 5)

    Returns:
        List of top-k nodes sorted by blended score (highest first)

    Example:
        >>> nodes = retrieve_nodes(query, 15)
        >>> user_profile = {"diagnoses": ["PCOS"], "symptoms": ["Itchy"]}
        >>> reranked = rerank_nodes_with_metadata(query, nodes, user_profile, 5)
        >>> len(reranked)
        5
    """
    if not nodes:
        return nodes

    logger.info(
        f"Metadata-weighted reranking: {len(nodes)} nodes, "
        f"user profile fields: {list(user_profile.keys())}"
    )

    reranker = get_reranker()

    # ========================================================================
    # Step 1: Get cross-encoder scores
    # ========================================================================
    pairs = [(query, node.node.text) for node in nodes]
    cross_encoder_scores = reranker.predict(pairs)
    logger.debug(f"Raw cross-encoder scores: min={min(cross_encoder_scores):.3f}, max={max(cross_encoder_scores):.3f}")

    # ========================================================================
    # Step 2: Normalize cross-encoder scores to 0-1 range
    # ========================================================================
    # ms-marco model outputs in range ~-5 to 5, so normalize to 0-1
    min_score = min(cross_encoder_scores)
    max_score = max(cross_encoder_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    normalized_ce_scores = [
        (score - min_score) / score_range for score in cross_encoder_scores
    ]
    logger.debug(f"Normalized cross-encoder scores: min={min(normalized_ce_scores):.3f}, max={max(normalized_ce_scores):.3f}")

    # ========================================================================
    # Step 3: Compute metadata overlap scores
    # ========================================================================
    metadata_scores = [
        compute_metadata_overlap_score(user_profile, node.node.metadata)
        for node in nodes
    ]
    logger.debug(f"Metadata overlap scores: min={min(metadata_scores):.3f}, max={max(metadata_scores):.3f}")

    # ========================================================================
    # Step 4: Blend scores (70% cross-encoder + 30% metadata)
    # ========================================================================
    blended_scores = [
        (0.7 * ce_score) + (0.3 * meta_score)
        for ce_score, meta_score in zip(normalized_ce_scores, metadata_scores)
    ]

    # ========================================================================
    # Step 5: Attach blended scores and sort
    # ========================================================================
    for node, blended_score in zip(nodes, blended_scores):
        node.score = blended_score

    reranked = sorted(nodes, key=lambda n: n.score, reverse=True)[:top_k]

    logger.info(
        f"Reranked {len(nodes)} → {len(reranked)} nodes. "
        f"Top score: {reranked[0].score:.3f}, "
        f"Bottom score: {reranked[-1].score:.3f}"
    )

    return reranked
```

**Key design notes**:
- Normalizes cross-encoder to 0-1 (handles the -5 to 5 range)
- Logs min/max at each step for debugging
- Maintains NodeWithScore structure (LlamaIndex compatible)
- Replaces the original score with blended score
- Sorts descending and keeps top-k

---

### STEP 3: Add import to nodes.py

**File**: `src/medical_agent/agents/nodes.py`

**Location**: Around line 15 (in the imports section, after existing reranker imports)

**Current imports** (find this section):
```python
from medical_agent.agents.llamaindex_retrieval import retrieve_nodes
from medical_agent.agents.reranker import rerank_nodes
from medical_agent.agents.state import MedicalAgentState
```

**Add this line**:
```python
from medical_agent.agents.reranker import rerank_nodes, rerank_nodes_with_metadata
```

(Or add `rerank_nodes_with_metadata` to the existing import)

---

### STEP 4: Modify retrieve_node() in nodes.py

**File**: `src/medical_agent/agents/nodes.py`

**Location**: In the `retrieve_node()` function, around lines 68-75

**Current code** (lines 65-77):
```python
    # CHANGED: Use raw query for retrieval (no health context dilution)
    logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Over-retrieve for reranking (15 candidates)
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)

    # Rerank with cross-encoder and keep top-5
    nodes = rerank_nodes(query=user_query, nodes=nodes, top_k=5)

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)
```

**Change to**:
```python
    # CHANGED: Use raw query for retrieval (no health context dilution)
    logger.info(f"Retrieving nodes for query: {user_query[:100]}...")

    # Over-retrieve for reranking (15 candidates)
    nodes = retrieve_nodes(query=user_query, similarity_top_k=15)

    # Extract user health profile from state for metadata-weighted reranking
    health_profile = state.get("health_profile", {})

    # Rerank with cross-encoder PLUS metadata weighting
    # 70% relevance score, 30% patient context alignment
    nodes = rerank_nodes_with_metadata(
        query=user_query,
        nodes=nodes,
        user_profile=health_profile,
        top_k=5
    )

    # Format nodes into citation text
    docs_text, citations = format_retrieved_nodes(nodes)
```

**Key changes**:
- Extract `health_profile` from state
- Call `rerank_nodes_with_metadata()` instead of `rerank_nodes()`
- Pass `user_profile=health_profile` to new function

---

## Type Hints & Dependencies

Make sure these imports are present in `agents/reranker.py`:

```python
from typing import Any
from llama_index.core.schema import NodeWithScore
```

These should already exist; if not, add them to the top of the file.

---

## Testing Checklist

### Unit Tests (Optional but recommended)

Create `tests/test_metadata_reranking.py`:

```python
import pytest
from medical_agent.agents.reranker import compute_metadata_overlap_score

def test_compute_metadata_overlap_score_perfect_match():
    """Test when user and paper metadata are identical."""
    user = {
        "diagnoses": ["PCOS"],
        "symptoms": ["Itchy"],
    }
    paper = {
        "diagnoses": ["PCOS"],
        "symptoms": ["Itchy"],
    }
    score = compute_metadata_overlap_score(user, paper)
    assert score > 0.5  # Should be high overlap

def test_compute_metadata_overlap_score_no_match():
    """Test when user and paper metadata have no overlap."""
    user = {
        "diagnoses": ["PCOS"],
        "symptoms": ["Itchy"],
    }
    paper = {
        "diagnoses": ["Adenomyosis"],
        "symptoms": ["Pelvic pain"],
    }
    score = compute_metadata_overlap_score(user, paper)
    assert score == 0.0

def test_compute_metadata_overlap_score_empty_profile():
    """Test with empty user profile."""
    user = {}
    paper = {
        "diagnoses": ["PCOS"],
    }
    score = compute_metadata_overlap_score(user, paper)
    assert score == 0.0

def test_rerank_nodes_with_metadata_shape():
    """Test that reranking returns correct number of nodes."""
    from medical_agent.agents.reranker import rerank_nodes_with_metadata
    from llama_index.core.schema import NodeWithScore, TextNode

    # Create mock nodes
    nodes = [
        NodeWithScore(node=TextNode(text=f"doc {i}"), score=0.5)
        for i in range(10)
    ]
    user_profile = {"diagnoses": ["PCOS"]}

    # This will fail without proper setup, but structure is correct
    # result = rerank_nodes_with_metadata("query", nodes, user_profile, 5)
    # assert len(result) == 5
```

### Integration Tests (Recommended)

1. **Test with actual API call**:
   ```bash
   # Start the API
   uvicorn medical_agent.api.main:app --reload

   # Call with health profile
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{
       "ph_value": 5.2,
       "diagnoses": ["PCOS"],
       "symptoms": ["Itchy"]
     }'
   ```

2. **Verify metadata boost**:
   - Check logs for "Metadata-weighted reranking" messages
   - Verify top results mention PCOS (if papers exist in DB)
   - Compare with pure cross-encoder reranking (revert to old `rerank_nodes()` temporarily)

---

## Verification Steps

### 1. Check imports work
```bash
cd /Users/ayushsingh/Documents/pHera/AI\ Nation/RAG/pHera\ Rag\ Code/Medical_Agent
python -c "from medical_agent.agents.reranker import rerank_nodes_with_metadata; print('✓ Import successful')"
```

### 2. Check syntax
```bash
python -m py_compile src/medical_agent/agents/reranker.py
python -m py_compile src/medical_agent/agents/nodes.py
```

### 3. Run linting
```bash
ruff check src/medical_agent/agents/reranker.py
ruff check src/medical_agent/agents/nodes.py
ruff format src/medical_agent/agents/
```

### 4. Run tests
```bash
pytest tests/ -v
```

### 5. Type checking (optional)
```bash
mypy src/medical_agent/agents/reranker.py
mypy src/medical_agent/agents/nodes.py
```

---

## Debugging & Logs

The implementation includes detailed logging. When running, look for:

**In retrieve_node logs**:
```
INFO - Metadata-weighted reranking: 15 nodes, user profile fields: ['diagnoses', 'symptoms']
DEBUG - Raw cross-encoder scores: min=0.123, max=0.856
DEBUG - Normalized cross-encoder scores: min=0.000, max=1.000
DEBUG - Metadata overlap scores: min=0.000, max=0.714
INFO - Reranked 15 → 5 nodes. Top score: 0.784, Bottom score: 0.632
```

**If metadata weighting isn't working**:
- Check that chunks have metadata stamped (see `ingestion/metadata.py`)
- Verify `health_profile` is populated in state (check logs before retrieve_node)
- If health_profile is empty, metadata scores will all be 0.0 (expected behavior)

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `AttributeError: 'dict' object has no attribute 'metadata'` | Ensure `node.node.metadata` (not `node.metadata`) in loops |
| Metadata scores all 0.0 | Check that user provides health profile in request |
| Scores out of range (> 1.0 or < 0.0) | Verify normalization logic handles edge cases (min==max) |
| Performance degradation | Cross-encoder reranking is the bottleneck (already exists), metadata is negligible |
| Papers not boosted as expected | Check metadata extraction quality in database; use `SELECT metadata FROM data_paper_chunks LIMIT 1` |

---

## Files Modified Summary

| File | Lines | Change |
|------|-------|--------|
| `agents/reranker.py` | +22 | Add `compute_metadata_overlap_score()` |
| `agents/reranker.py` | +80 | Add `rerank_nodes_with_metadata()` |
| `agents/nodes.py` | 1 | Update import line (~15) |
| `agents/nodes.py` | 5-6 | Modify `retrieve_node()` (~68-75) |

**Total additions**: ~102 lines of code (+ comments/docstrings)
**Total modifications**: 2 functions changed in nodes.py (7 lines)

---

## Expected Behavior Changes

### Before Phase 1
Query: "pH 5.2 with itching"
User profile: PCOS diagnosis

Results (pure cross-encoder):
```
1. [0.82] Generic pH levels article
2. [0.79] BV and yeast infections overview
3. [0.76] PCOS symptoms (irregular periods)
4. [0.71] Vaginal health basics
5. [0.68] PCOS + hormonal contraception
```

### After Phase 1
Same query and profile:

Results (70% cross-encoder + 30% metadata):
```
1. [0.78] PCOS + pH + yeast infection (matched 2/3 user attributes)
2. [0.76] PCOS + vulvovaginitis in reproductive age
3. [0.72] Generic pH levels article
4. [0.70] Vaginal health in PCOS patients
5. [0.68] BV and yeast infections overview
```

**Notice**: Papers explicitly discussing PCOS move up because metadata overlap boosts them.

---

## Next Steps After Phase 1

Once Phase 1 is working:
- Run evaluation: `python -m medical_agent.evaluation.run_evaluation --testset testsets/testset_*.csv`
- Compare metrics (NDCG, MRR) with/without metadata weighting
- Tune the 70/30 ratio if needed (currently research-backed, may want to A/B test)
- Move to Phase 2: Reasoning node with confidence validation

---

## Reference Files

- Metadata extraction: `src/medical_agent/ingestion/metadata.py`
- Current reranker: `src/medical_agent/agents/reranker.py` (existing `rerank_nodes()`)
- Retrieve node: `src/medical_agent/agents/nodes.py:34-82`
- State definition: `src/medical_agent/agents/state.py`
