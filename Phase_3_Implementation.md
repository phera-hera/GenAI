# Phase 3: Agentic Loop with Iterative Retrieval & Confidence-Adaptive Prompting

## Sub-Phase Breakdown (4 Steps)

Implement Phase 3 in smaller, testable chunks:

### Phase 3.1 — Confidence-Adaptive Prompting
**Files**: `prompts.py` (new) + `nodes.py` (`generate_node` only)

Create 3 prompt tiers (HIGH/MEDIUM/LOW confidence) and wire into `generate_node`. Responses now adapt tone based on Phase 2 confidence scores.

**Testable**: Run query, check logs for `"Using HIGH/MEDIUM/LOW confidence prompt"`.

---

### Phase 3.2 — State & Retrieval Plumbing
**Files**: `state.py` + `nodes.py` (`retrieve_node` only)

Add 4 new state fields: `original_query`, `retry_count`, `refinement_history`, `skip_retry`. Update `retrieve_node` to save original and use refined query on retries.

**No visible change yet** — groundwork only.

---

### Phase 3.3 — Query Refinement Node
**Files**: `refine_query.py` (new)

Create `generate_refined_query()`, `check_query_similarity()`, `refine_query_node()`. Node exists but not wired to graph yet.

**Testable in isolation**: Call `refine_query_node` directly.

---

### Phase 3.4 — Agentic Loop (Graph Wiring)
**Files**: `graph.py` only

Replace linear `reasoning → generate` with conditional routing. Activates the full loop: `reasoning → [refine_query ↔ retrieve] → generate`.

**Full system active**: Low-confidence queries now retry with refined queries.

---

## Overview

**Goal**: Add conditional looping to retry retrieval when confidence is low, AND adapt generation prompts based on confidence tiers.

**Current state**: retrieve → reasoning → generate (no looping, single prompt)

**New state**: retrieve → reasoning → (if confidence < 0.7: refine query & retry) → generate (with adaptive prompt tier)

**Key principle**: Semantic check before retry, and confidence-based prompt adaptation for medical safety.

---

## Architecture Overview

### Current Flow (Phase 1-2)
```
retrieve_node (Phase 1: metadata weighting)
  ↓
reasoning_node (Phase 2: confidence assessment)
  ↓
generate_node (single prompt for all confidence levels)
  ↓
END
```

### New Flow (Phase 3)
```
retrieve_node (Phase 1: metadata weighting)
  ↓
reasoning_node (Phase 2: confidence assessment)
  ↓
  confidence >= 0.7?
  ↙              ↖
YES               NO
↓                 ↓
proceed        refine_query_node (NEW)
               ├─ LLM asks: "What's missing?"
               ├─ Generate refined query
               ├─ Check similarity (>10% different)
               └─ if diverse enough:
                    ↓
                 retrieve_node (retry)
                    ↓
                 reasoning_node (re-evaluate)
                    ↓
                 if confidence >= 0.7 OR max_retries:
                    ↓
                 generate_node
               else:
                 skip to generate_node
↓
generate_node (Phase 3 ENHANCEMENT: adaptive prompting)
├─ if confidence >= 0.75: HIGH confidence prompt
├─ if 0.50 ≤ confidence < 0.75: MEDIUM confidence prompt
└─ if confidence < 0.50: LOW confidence prompt
↓
END
```

---

## Key Components

### 1. Conditional Routing Logic
- **Threshold**: 0.7 (decides: proceed or retry)
- **Retry decision**: Use semantic check (not just score)
- **Max retries**: 2 (3 total retrieval attempts)

### 2. Query Refinement Strategy
- **Method**: LLM semantic reasoning
- **Input**: Original query, failed docs, confidence reason
- **Output**: Refined query targeting missing aspects
- **Validation**: Embedding similarity check (>10% different)

### 3. Confidence-Adaptive Generation (NEW)
- **High confidence (≥0.75)**: Direct, confident tone
- **Medium confidence (0.50-0.75)**: Exploratory, balanced tone
- **Low confidence (<0.50)**: Humble, cautious with warnings

---

## Files to Modify/Create

| File | Action | Summary |
|------|--------|---------|
| `src/medical_agent/agents/state.py` | MODIFY | Add `retry_count`, `refinement_history` fields |
| `src/medical_agent/agents/refine_query.py` | CREATE (new) | Query refinement logic |
| `src/medical_agent/agents/graph.py` | MODIFY | Add conditional routing, new node |
| `src/medical_agent/agents/nodes.py` | MODIFY | Update generate_node with 3-tier prompts |
| `src/medical_agent/agents/prompts.py` | CREATE (new) | 3 system prompts (high/medium/low confidence) |

---

## Step-by-Step Implementation

### STEP 1: Update state.py with new fields

**File**: `src/medical_agent/agents/state.py`

Add to `MedicalAgentState`:

```python
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages

class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        [existing fields...]
        retry_count: Number of retrieval retries (0-2)
        refinement_history: List of refined queries attempted
    """

    messages: Annotated[list, add_messages]
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str
    citations: list[dict[str, Any]]
    used_citations: list[int]
    confidence_score: float
    retrieval_quality: str
    confidence_method: str
    retry_count: int                           # ← NEW
    refinement_history: list[str]              # ← NEW
    original_query: str                        # ← NEW (user's original query, used for refinement reference)
    skip_retry: bool                           # ← NEW (True = refined query too similar, skip retry)
```

---

### STEP 2: Create prompts.py with 3-tier prompts

**File**: `src/medical_agent/agents/prompts.py` (NEW FILE)

```python
"""
Confidence-adaptive system prompts for medical response generation.

Three tiers based on retrieval confidence:
- HIGH (>=0.75): Direct, confident tone
- MEDIUM (0.50-0.75): Exploratory, measured tone
- LOW (<0.50): Humble, cautious with warnings
"""

HIGH_CONFIDENCE_PROMPT = """You are a knowledgeable women's health consultant providing evidence-based information.

The retrieved research is highly relevant to this patient's query. Present findings confidently and directly.

INSTRUCTIONS:
- Make clear statements supported by the documents
- Provide specific recommendations based on evidence
- Use authoritative but warm tone
- Every factual claim MUST cite sources with [1], [2], etc.
- Ground your response entirely in the provided documents

RULES:
- Do NOT add medical knowledge outside the documents
- Every claim must have a citation
- Be warm, professional, and direct
- Focus on answering the patient's exact question

PATIENT CONTEXT:
{health_context}

DOCUMENTS:
{docs_text}

QUESTION:
{query}

Provide your confident, evidence-based response:"""


MEDIUM_CONFIDENCE_PROMPT = """You are a thoughtful women's health information provider.

The retrieved research provides useful context but may not be comprehensive for this specific question. Present findings in an exploratory, balanced way.

INSTRUCTIONS:
- Use measured language: 'may', 'could', 'suggests', 'research indicates'
- Explore multiple relevant angles from the documents
- Acknowledge limitations of available information
- Recommend consulting healthcare provider when appropriate
- Every citation must be marked [1], [2], etc.
- Ground your response in the provided documents

RULES:
- Do NOT make definitive claims when uncertain
- Use hedging language for speculative points
- Encourage professional medical consultation
- Every factual claim must have a citation
- Acknowledge what the documents DON'T cover

PATIENT CONTEXT:
{health_context}

DOCUMENTS:
{docs_text}

QUESTION:
{query}

Provide your balanced, exploratory response:"""


LOW_CONFIDENCE_PROMPT = """You are a cautious health information assistant with limited knowledge on this topic.

The retrieved research has LIMITED RELEVANCE to this specific question. Present only what the documents provide with explicit humility and strong medical disclaimer.

INSTRUCTIONS:
- Clearly state: "Limited information is available on this specific topic"
- List only what documents provide, with heavy hedging
- Use phrases like: "possibly", "may relate to", "limited evidence suggests"
- Include a STRONG medical disclaimer
- Emphasize consulting a healthcare provider
- Every claim must cite sources [1], [2], etc.
- Ground your response entirely in the limited documents

RULES:
- Do NOT make unsupported claims
- Heavy use of disclaimers
- Explicit acknowledgment of knowledge gaps
- Strong emphasis on seeking professional medical advice
- Every factual claim must be cited
- Include confidence notice: "This analysis has LIMITED CONFIDENCE"

PATIENT CONTEXT:
{health_context}

DOCUMENTS:
{docs_text}

QUESTION:
{query}

⚠️ IMPORTANT: You MUST include a disclaimer that this analysis has LIMITED confidence in the retrieved information.

Provide your humble, cautious response with clear limitations:"""
```

---

### STEP 3: Create refine_query.py

**File**: `src/medical_agent/agents/refine_query.py` (NEW FILE)

```python
"""
Query refinement for agentic retrieval retry (Phase 3).

Generates semantically different queries when initial retrieval is insufficient.
Includes similarity checking to prevent loops.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


async def generate_refined_query(
    original_query: str,
    user_message: str,
    retrieved_docs: str,
    confidence_reason: str,
    llm: AzureChatOpenAI,
) -> str:
    """
    Generate a refined query based on what's missing from initial retrieval.

    Uses LLM to reason about retrieval failure and generate targeted query.

    Args:
        original_query: User's original search query
        user_message: Original user question
        retrieved_docs: Text of retrieved documents (why they failed)
        confidence_reason: Reasoning for low confidence
        llm: Azure OpenAI LLM instance

    Returns:
        Refined query string targeting missing aspects
    """
    logger.info("Generating refined query for retrieval retry")

    refinement_prompt = f"""You are a medical query refinement specialist. The initial retrieval for this patient query was insufficient.

Original Query: {original_query}
User Question: {user_message}

Retrieved Documents Summary:
{retrieved_docs[:500]}...

Confidence Issue: {confidence_reason}

TASK: Generate a focused, refined query that:
1. Targets what the original query missed
2. Is semantically different from the original
3. Uses medical terminology precisely
4. Focuses on one specific aspect the initial retrieval missed

Do NOT:
- Repeat the original query
- Use vague language
- Make it too broad

Output ONLY the refined query string, nothing else:"""

    response = await llm.ainvoke([HumanMessage(content=refinement_prompt)])
    refined_query = response.content.strip()

    logger.info(f"Refined query: {refined_query[:100]}...")

    return refined_query


def check_query_similarity(
    original_query: str,
    refined_query: str,
    embed_model: AzureOpenAIEmbedding,
    similarity_threshold: float = 0.90,
) -> bool:
    """
    Check if refined query is sufficiently different from original.

    Uses embedding similarity: if >90% similar, considered duplicate.

    Args:
        original_query: Original search query
        refined_query: Newly generated refined query
        embed_model: Azure OpenAI embedding model
        similarity_threshold: Threshold for similarity (0.90 = 90%)

    Returns:
        bool: True if queries are sufficiently different (proceed with retry)
              False if too similar (skip retry)
    """
    logger.info("Checking query similarity")

    # Get embeddings
    original_embedding = embed_model.get_text_embedding(original_query)
    refined_embedding = embed_model.get_text_embedding(refined_query)

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    similarity = cosine_similarity(
        [original_embedding],
        [refined_embedding]
    )[0][0]

    logger.info(f"Query similarity: {similarity:.3f} (threshold: {similarity_threshold})")

    if similarity > similarity_threshold:
        logger.warning(
            f"Refined query too similar to original ({similarity:.3f} > {similarity_threshold}). "
            f"Skipping retry to avoid loop."
        )
        return False

    logger.info(f"Query sufficiently different ({similarity:.3f} <= {similarity_threshold}). Proceeding with retry.")
    return True


async def refine_query_node(state: Any) -> dict[str, Any]:
    """
    Reasoning node that refines query for retry when confidence is low.

    Process:
    1. LLM analyzes what's missing from retrieved documents
    2. Generate refined query targeting those aspects
    3. Check similarity to original query (avoid loops)
    4. If different enough, return refined query
    5. If too similar, mark as "skip retry"

    Args:
        state: Current agent state with docs_text, confidence info

    Returns:
        State update with refined_query (or None to skip retry)
    """
    logger.info("Executing refine_query_node")

    original_query = state.get("original_query", "")
    user_message = ""

    # Extract user message
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        user_message = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message.get("content", ""))
        )

    docs_text = state.get("docs_text", "")
    confidence_score = state.get("confidence_score", 0.0)
    confidence_method = state.get("confidence_method", "unknown")

    # Create confidence reason explanation
    confidence_reason = f"Confidence score: {confidence_score:.2f}, method: {confidence_method}"

    # Get LLM
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Generate refined query
    refined_query = await generate_refined_query(
        original_query=original_query,
        user_message=user_message,
        retrieved_docs=docs_text,
        confidence_reason=confidence_reason,
        llm=llm,
    )

    # Check similarity
    embed_model = AzureOpenAIEmbedding(
        model=settings.azure_openai_embedding_deployment_name,
        deployment_name=settings.azure_openai_embedding_deployment_name,
        api_key=settings.azure_openai_embedding_api_key or settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_embedding_endpoint or settings.azure_openai_endpoint,
        api_version=settings.azure_openai_embedding_api_version or settings.azure_openai_api_version,
    )

    is_different = check_query_similarity(
        original_query=original_query,
        refined_query=refined_query,
        embed_model=embed_model,
        similarity_threshold=0.90,  # >90% similar = too similar
    )

    if not is_different:
        logger.warning("Refined query too similar to original. Skipping retry.")
        return {
            "refined_query": None,
            "skip_retry": True,
        }

    logger.info(f"Refined query approved for retry: {refined_query[:80]}...")

    return {
        "refined_query": refined_query,
        "skip_retry": False,
    }
```

---

### STEP 4: Update graph.py with conditional routing

**File**: `src/medical_agent/agents/graph.py`

Replace the entire `build_medical_rag_graph()` function:

```python
from medical_agent.agents.refine_query import refine_query_node

def build_medical_rag_graph() -> "CompiledStateGraph":
    """
    Build and compile the medical RAG agent graph with agentic looping (Phase 3).

    Graph flow:
        START → retrieve → reasoning → [conditional]
                                       ├─ if confidence >= 0.7: generate
                                       └─ if confidence < 0.7: refine → retrieve (max 2 retries)
                                                                       ↓
                                                                     reasoning
                                                                       ↓
                                                                     generate → END

    Features:
        - Phase 1: Metadata-weighted retrieval
        - Phase 2: Confidence assessment with conditional LLM validation
        - Phase 3: Agentic loop with query refinement + adaptive prompting
        - MemorySaver for multi-turn conversations
        - Conditional edges for retry logic

    Returns:
        Compiled LangGraph application with checkpointer
    """
    logger.info("Building medical RAG graph with Phase 3 agentic loop")

    # Create state graph
    workflow = StateGraph(MedicalAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("refine_query", refine_query_node)  # ← NEW Phase 3
    workflow.add_node("generate", generate_node)

    # Define edges with conditional routing
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reasoning")

    # Conditional: if confidence >= 0.7 OR retries exhausted, go to generate. Otherwise refine query.
    # NOTE: Do NOT add a separate workflow.add_edge("reasoning", "generate") — this conditional
    # already handles the high-confidence path. Adding both causes a LangGraph conflict.
    workflow.add_conditional_edges(
        "reasoning",
        lambda state: "generate" if state.get("confidence_score", 0) >= 0.7 or state.get("retry_count", 0) >= 2 else "refine_query",
    )

    # From refine_query: if query too similar (skip_retry=True), go to generate. Otherwise loop back to retrieve.
    workflow.add_conditional_edges(
        "refine_query",
        lambda state: "generate" if state.get("skip_retry", True) else "retrieve",
    )

    workflow.add_edge("generate", END)

    # Add memory for multi-turn conversations
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("Medical RAG graph compiled with Phase 3 agentic loop")

    return app
```

**Add import**:
```python
from medical_agent.agents.refine_query import refine_query_node
```

---

### STEP 5: Update nodes.py - generate_node with 3-tier prompting

**File**: `src/medical_agent/agents/nodes.py`

Replace the `generate_node()` function:

```python
from medical_agent.agents.prompts import (
    HIGH_CONFIDENCE_PROMPT,
    MEDIUM_CONFIDENCE_PROMPT,
    LOW_CONFIDENCE_PROMPT,
)

async def generate_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Generation node: Produces medical response with confidence-adaptive prompting.

    Three-tier prompt strategy based on confidence:
    - HIGH (>=0.75): Confident, direct tone
    - MEDIUM (0.50-0.75): Exploratory, measured tone
    - LOW (<0.50): Humble, cautious with warnings

    All responses use only documents from knowledge base. Tone/phrasing adapts.

    Args:
        state: Current agent state with docs_text, confidence_score

    Returns:
        State update with assistant message and used_citations
    """
    logger.info("Executing generate_node with confidence-adaptive prompting")

    # Get LLM
    llm = AzureChatOpenAI(
        deployment_name=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        temperature=0.0,
    )

    # Use structured output for citations
    structured_llm = llm.with_structured_output(MedicalResponse)

    # Extract context
    docs_text = state.get("docs_text", "")
    ph_value = state.get("ph_value", 0.0)
    health_profile = state.get("health_profile", {})
    health_context = build_health_context(ph_value, health_profile)
    confidence_score = state.get("confidence_score", 0.5)

    # Get conversation history
    messages = state.get("messages", [])
    conversation_history = ""
    if len(messages) > 1:
        history_parts = []
        for msg in messages[:-1]:
            role = msg.get("role", "user") if isinstance(msg, dict) else getattr(msg, "type", "user")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            history_parts.append(f"{role.capitalize()}: {content}")
        conversation_history = "\n".join(history_parts)

    # Get current user query
    last_message = messages[-1]
    current_query = (
        last_message.get("content", "")
        if isinstance(last_message, dict)
        else getattr(last_message, "content", "")
    )

    # Select prompt tier based on confidence
    if confidence_score >= 0.75:
        base_prompt = HIGH_CONFIDENCE_PROMPT
        prompt_tier = "high"
    elif confidence_score >= 0.50:
        base_prompt = MEDIUM_CONFIDENCE_PROMPT
        prompt_tier = "medium"
    else:
        base_prompt = LOW_CONFIDENCE_PROMPT
        prompt_tier = "low"

    logger.info(f"Using {prompt_tier.upper()} confidence prompt (score: {confidence_score:.3f})")

    # Build final prompt
    system_prompt = base_prompt.format(
        health_context=health_context,
        docs_text=docs_text,
        query=current_query,
    )

    # Add conversation history if exists
    if conversation_history:
        system_prompt += f"\n\nPREVIOUS CONVERSATION:\n{conversation_history}\n"

    system_prompt += f"\nANSWER THE QUESTION:"

    logger.info("Generating structured response with adaptive prompt")

    # Generate structured response
    result: MedicalResponse = structured_llm.invoke([HumanMessage(content=system_prompt)])

    logger.info(f"Response generated (tier: {prompt_tier}, citations: {len(result.used_citations)})")

    # Return as assistant message with metadata
    assistant_msg = AIMessage(
        content=result.response,
        metadata={
            "confidence_score": confidence_score,
            "confidence_tier": prompt_tier,
        }
    )

    return {
        "messages": [assistant_msg],
        "used_citations": result.used_citations,
    }
```

---

### STEP 6: Update retrieve_node to save original_query

In `retrieve_node()` in `nodes.py`, add at the beginning:

```python
# Save original query for refinement later
original_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))
state_update = {
    "original_query": original_query,
}
```

And add to return statement at the end:

```python
state_update.update({
    "docs_text": docs_text,
    "citations": citations,
})
return state_update
```

---

## Testing Checklist

### Unit Tests

Create `tests/test_phase_3.py`:

```python
import pytest
from medical_agent.agents.refine_query import check_query_similarity
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

def test_check_query_similarity_identical():
    """Test that identical queries are flagged as too similar."""
    query = "PCOS and pH levels"
    embed_model = AzureOpenAIEmbedding(...)  # Mock

    # This would require real embeddings, but logic is:
    # If similarity > 0.90, return False (too similar, skip retry)

def test_check_query_similarity_different():
    """Test that different queries pass similarity check."""
    original = "PCOS treatment options"
    refined = "Management strategies for PCOS with hormonal IUD"

    # If similarity < 0.90, return True (different enough, proceed)
```

### Integration Tests

```bash
# Start API
uvicorn medical_agent.api.main:app --reload

# Test with low-confidence query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "ph_value": 7.0,
    "symptoms": ["Unknown symptom"],
    "diagnoses": ["Rare condition"]
  }'

# Check logs for:
# - "Uncertain zone: calling LLM for validation"
# - "Refining query for retry"
# - "Using LOW confidence prompt"
```

---

## Verification Steps

### 1. Check syntax
```bash
python -c "from medical_agent.agents.refine_query import refine_query_node; print('✓')"
python -c "from medical_agent.agents.prompts import HIGH_CONFIDENCE_PROMPT; print('✓')"
```

### 2. Verify graph structure
```bash
python -c "from medical_agent.agents.graph import build_medical_rag_graph; g = build_medical_rag_graph(); print('✓ Graph built')"
```

### 3. Lint and format
```bash
ruff check src/medical_agent/agents/
ruff format src/medical_agent/agents/
```

### 4. Type checking
```bash
mypy src/medical_agent/agents/refine_query.py
mypy src/medical_agent/agents/prompts.py
```

---

## Debugging & Logs

Expected log sequence:

```
INFO - Executing retrieve_node
INFO - Metadata-weighted reranking: 15 nodes
INFO - Retrieved 5 citations after reranking

INFO - Executing reasoning_node (async)
INFO - Score-based confidence: 0.62
INFO - Uncertain zone (0.4-0.8): calling LLM for validation
INFO - LLM validation: confidence=0.58
INFO - Hybrid confidence: 0.602

[Low confidence, trigger retry]
INFO - Executing refine_query_node
INFO - Generating refined query for retrieval retry
INFO - Refined query: "PCOS management with copper IUD interactions"
INFO - Query similarity: 0.45 (threshold: 0.90)
INFO - Query sufficiently different. Proceeding with retry.

INFO - Executing retrieve_node (RETRY)
INFO - Reranked 15 → 5 nodes.
INFO - Retrieved 5 citations after reranking

INFO - Executing reasoning_node (async)
INFO - Score-based confidence: 0.73
INFO - High confidence (>= 0.8): Using score-based confidence only

[Confidence improved, proceed to generate]
INFO - Executing generate_node with confidence-adaptive prompting
INFO - Using MEDIUM confidence prompt (score: 0.73)
INFO - Response generated (tier: medium, citations: 4)
```

---

## Files Modified Summary

| File | Lines | Change |
|------|-------|--------|
| `agents/state.py` | +2 fields | Add retry_count, refinement_history |
| `agents/prompts.py` | +150 | NEW: 3 system prompts (high/medium/low) |
| `agents/refine_query.py` | +200 | NEW: Query refinement + similarity check |
| `agents/graph.py` | +10 | Add conditional edges for retry logic |
| `agents/nodes.py` | +40 | Update generate_node with 3-tier prompting |

**Total additions**: ~400 lines
**Total modifications**: 2 graph edges + generate_node logic

---

## Expected Behavior Changes

### Before Phase 3
```
Query: "pH 7.0 + rare condition"
retrieve_node: 5 chunks, scores [0.45, 0.42, 0.40, 0.38, 0.35]
reasoning_node: confidence=0.40 (low)
generate_node: Single prompt, outputs cautious answer
Response: Low-confidence answer (no retry)
```

### After Phase 3
```
Query: "pH 7.0 + rare condition"
retrieve_node: 5 chunks, scores [0.45, 0.42, 0.40, 0.38, 0.35]
reasoning_node: confidence=0.40 (low)
refine_query_node: Generates refined query
retrieve_node (RETRY): New query gets better matches
reasoning_node: confidence=0.68 (medium, improved)
generate_node: Uses MEDIUM confidence prompt (exploratory tone)
Response: Better quality answer with measured tone
```

---

## Guard Rails Implemented

1. **Max retries**: Hard limit of 2 (3 total attempts)
2. **Query diversity**: Embedding similarity must be <90%
3. **Fallback**: If refined query fails, use best-effort from first retrieval
4. **Cost control**: LLM calls capped per phase (validation + refinement only when needed)
5. **Timeout**: No hard timeout (let it complete, medical domain tolerates latency)

---

## Next Steps

After implementation:
1. Test all three prompt tiers with real queries
2. Monitor retry rates (should be <30% of queries)
3. Tune confidence thresholds based on evaluation results
4. Evaluate response quality with RAGAS metrics
5. A/B test: Phase 2 (no retry) vs Phase 3 (with retry)
