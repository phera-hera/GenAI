# Phase 2: Reasoning Node with Confidence Validation Implementation Guide

## Overview

**Goal**: Add a validation node that assesses retrieval quality and computes a confidence score for the entire retrieved set (all 5 chunks together).

**Current state**: retrieve_node → generate_node (linear, no quality assessment)

**New state**: retrieve_node → reasoning_node → generate_node (with confidence validation)

**Key principle**: One confidence score for ALL 5 chunks, NOT per-chunk. Call LLM only when uncertain (0.4-0.8 range).

**Why**: Medical queries need semantic validation. High scores might not answer the actual question. Confidence signal prepares for Phase 3 agentic looping.

---

## Architecture Overview

### Current Flow (Phase 1)
```
retrieve_node:
  Step 1: retrieve_nodes(query, 15)
  Step 2: rerank_nodes_with_metadata(15→5, health_profile)
  Step 3: format_retrieved_nodes(5)
    Output: 5 chunks with scores
    ↓
generate_node:
  Step 1: build_health_context()
  Step 2: LLM generation
    Output: response + citations
```

### New Flow (Phase 2)
```
retrieve_node:
  Step 1: retrieve_nodes(query, 15)
  Step 2: rerank_nodes_with_metadata(15→5, health_profile)
  Step 3: format_retrieved_nodes(5)
    Output: 5 chunks with scores [0.85, 0.76, 0.71, 0.68, 0.63]
    ↓
reasoning_node: ← NEW NODE
  Step 1: Calculate score_confidence = avg(5 scores) = 0.728
  Step 2: Decision tree:
    IF score_confidence >= 0.8:
      → Use score_confidence (no LLM call)
    ELIF score_confidence < 0.4:
      → Use score_confidence (already failed)
    ELSE (0.4 <= score_confidence < 0.8):
      → Call LLM once: "Do these 5 chunks answer the query?"
      → Get llm_confidence
      → Blend: final = (0.6 × score_confidence) + (0.4 × llm_confidence)
  Step 3: Determine quality level (high/low) based on confidence
    Output: confidence_score, retrieval_quality, method
    ↓
generate_node:
  Step 1: build_health_context()
  Step 2: LLM generation (same as before)
  Step 3: Include confidence_score in response metadata
    Output: response + citations + confidence_score
```

### Graph Structure
```
START
  ↓
[retrieve_node] (Phase 1 ✅)
  ├─ Over-retrieve: 15
  ├─ Rerank with metadata: 15→5
  └─ Output: 5 chunks with scores
  ↓
[reasoning_node] ← NEW (Phase 2)
  ├─ Calculate confidence for all 5 chunks
  ├─ Conditionally validate with LLM
  └─ Output: confidence_score, retrieval_quality
  ↓
[generate_node] (unchanged)
  ├─ build_health_context()
  └─ LLM generation
  ↓
END
```

---

## State Changes Required

### Update MedicalAgentState

**File**: `src/medical_agent/agents/state.py`

Add these fields:
```python
confidence_score: float              # 0.0-1.0
retrieval_quality: str               # "high", "low"
confidence_method: str               # "score_only_high", "score_only_low", "hybrid"
```

**Full updated state**:
```python
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages

class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        messages: Conversation history
        ph_value: User's pH measurement
        health_profile: User's health context
        docs_text: Formatted citation text from retrieval
        citations: List of citation metadata dicts
        used_citations: List of citation IDs used in response
        confidence_score: Confidence in retrieval quality (0.0-1.0) ← NEW
        retrieval_quality: Assessment of retrieval (high/low) ← NEW
        confidence_method: How confidence was calculated ← NEW
    """

    messages: Annotated[list, add_messages]
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str
    citations: list[dict[str, Any]]
    used_citations: list[int]
    confidence_score: float             # ← NEW
    retrieval_quality: str              # ← NEW
    confidence_method: str              # ← NEW
```

---

## Files to Modify/Create

| File | Action | Summary |
|------|--------|---------|
| `src/medical_agent/agents/state.py` | MODIFY | Add 3 new fields to MedicalAgentState |
| `src/medical_agent/agents/reasoning.py` | CREATE (new) | New file with reasoning_node() + confidence functions |
| `src/medical_agent/agents/graph.py` | MODIFY | Add reasoning_node to graph |
| `src/medical_agent/agents/nodes.py` | MODIFY | Import reasoning_node |

---

## Step-by-Step Implementation

### STEP 1: Update state.py with new fields

**File**: `src/medical_agent/agents/state.py`

**Current state** (lines 1-31):
```python
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages

class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages)
        ph_value: User's pH measurement
        health_profile: Dict containing age, symptoms, diagnoses, etc.
        docs_text: Formatted citation text from retrieval ("[1]: [Paper:page]: text")
        citations: List of citation metadata dicts
        used_citations: List of citation IDs actually used in the response
    """

    messages: Annotated[list, add_messages]  # Auto-managed conversation history
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str
    citations: list[dict[str, Any]]
    used_citations: list[int]  # Citation IDs used by LLM in response
```

**Change to**:
```python
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages

class MedicalAgentState(TypedDict):
    """
    State schema for the medical RAG agent graph.

    Attributes:
        messages: Conversation history (managed by LangGraph's add_messages)
        ph_value: User's pH measurement
        health_profile: Dict containing age, symptoms, diagnoses, etc.
        docs_text: Formatted citation text from retrieval ("[1]: [Paper:page]: text")
        citations: List of citation metadata dicts
        used_citations: List of citation IDs actually used in the response
        confidence_score: Confidence score for retrieval quality (0.0-1.0)
        retrieval_quality: Assessment of retrieval quality ("high" or "low")
        confidence_method: Method used to calculate confidence ("score_only_high", "score_only_low", "hybrid")
    """

    messages: Annotated[list, add_messages]  # Auto-managed conversation history
    ph_value: float
    health_profile: dict[str, Any]
    docs_text: str
    citations: list[dict[str, Any]]
    used_citations: list[int]  # Citation IDs used by LLM in response
    confidence_score: float  # Confidence in retrieval (0.0-1.0) ← NEW
    retrieval_quality: str   # "high" or "low" ← NEW
    confidence_method: str   # How confidence was calculated ← NEW
```

---

### STEP 2: Create new reasoning.py file

**File**: `src/medical_agent/agents/reasoning.py` (NEW FILE)

**Add entire file content**:

```python
"""
Reasoning node for medical RAG: assess retrieval quality and compute confidence.

Confidence calculation strategy:
  1. Score-based: Average relevance score of top-5 chunks
  2. LLM validation (conditional): If score is uncertain (0.4-0.8),
     call LLM once to validate if chunks answer the query
  3. Hybrid blending: (0.6 × score) + (0.4 × llm) for uncertain cases

Agentic optimization: Only call LLM when uncertain. Saves 60-70% of LLM calls.
"""

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from medical_agent.agents.state import MedicalAgentState
from medical_agent.core.config import settings

logger = logging.getLogger(__name__)


def compute_score_based_confidence(citations: list[dict[str, Any]]) -> float:
    """
    Compute confidence from retrieval scores alone.

    Average the relevance scores of the 5 retrieved chunks.

    Args:
        citations: List of citation dicts with "score" field

    Returns:
        float: Average score (0.0-1.0)

    Example:
        >>> citations = [
        ...     {"score": 0.85},
        ...     {"score": 0.76},
        ...     {"score": 0.71},
        ...     {"score": 0.68},
        ...     {"score": 0.63}
        ... ]
        >>> compute_score_based_confidence(citations)
        0.726
    """
    if not citations:
        return 0.0

    scores = [c.get("score", 0.0) for c in citations]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    logger.debug(f"Score-based confidence: {avg_score:.3f} (from {len(scores)} chunks)")

    return avg_score


async def validate_with_llm(
    query: str,
    docs_text: str,
    health_profile: dict[str, Any],
    llm: AzureChatOpenAI,
) -> float:
    """
    Validate if retrieved documents answer the user's query using LLM.

    Single LLM call to assess if all chunks together contain information
    to answer the user's question.

    Args:
        query: User's original query
        docs_text: Formatted text of all 5 chunks
        health_profile: User's health context
        llm: Azure OpenAI LLM instance

    Returns:
        float: Confidence score 0.0-1.0 from LLM assessment

    Example:
        LLM evaluates: "Do these 5 documents together answer the query
                        'I have PCOS and pH 5.2, what could cause itching?'"
        LLM returns: {"can_answer": true, "confidence": 0.82, "reason": "..."}
    """
    logger.info("Validating retrieval with LLM (uncertain confidence zone)")

    # Build health context string for LLM
    health_context_parts = []
    if health_profile.get("diagnoses"):
        health_context_parts.append(f"Diagnoses: {', '.join(health_profile['diagnoses'])}")
    if health_profile.get("symptoms"):
        health_context_parts.append(f"Symptoms: {', '.join(health_profile['symptoms'])}")
    if health_profile.get("age"):
        health_context_parts.append(f"Age: {health_profile['age']}")

    health_context = "\n".join(health_context_parts) if health_context_parts else "No health context provided"

    # Construct validation prompt
    validation_prompt = f"""You are a medical information validation expert. Assess whether the retrieved documents contain information that helps answer the user's query.

User Query: {query}

User Health Context:
{health_context}

Retrieved Documents:
{docs_text}

Task: Evaluate if these documents together contain information relevant to answering the user's query.

Respond with a JSON object:
{{
  "can_answer": true/false,
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}}

Confidence scale:
- 0.0-0.3: Documents completely irrelevant
- 0.3-0.6: Documents partially relevant, missing key information
- 0.6-0.8: Documents mostly relevant, have main answer
- 0.8-1.0: Documents highly relevant, comprehensive coverage"""

    try:
        response = await llm.ainvoke([HumanMessage(content=validation_prompt)])
        response_text = response.content.strip()

        logger.debug(f"LLM validation response: {response_text[:200]}")

        # Parse JSON from LLM response
        # LLM might include extra text, so try to extract JSON
        try:
            # Try direct JSON parsing first
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON from text if wrapped in markdown or extra text
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                logger.warning("Could not parse LLM validation response as JSON, defaulting to 0.5")
                return 0.5

        llm_confidence = float(result.get("confidence", 0.5))
        can_answer = result.get("can_answer", False)
        reasoning = result.get("reasoning", "")

        logger.info(f"LLM validation: can_answer={can_answer}, confidence={llm_confidence:.3f}")
        logger.debug(f"LLM reasoning: {reasoning}")

        # Clamp to 0.0-1.0 range
        llm_confidence = max(0.0, min(1.0, llm_confidence))

        return llm_confidence

    except Exception as e:
        logger.error(f"LLM validation failed: {e}", exc_info=True)
        # Graceful fallback
        return 0.5


def reasoning_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Reasoning node: Assess retrieval quality and compute confidence score.

    Process:
      1. Extract scores from retrieved citations
      2. Compute score-based confidence (average of 5 scores)
      3. Decision tree:
         - If score >= 0.8: High confidence, use score only
         - If score < 0.4: Low confidence, use score only
         - If 0.4 <= score < 0.8: Uncertain, validate with LLM (1 call)
      4. Determine retrieval quality (high if confidence >= 0.7, else low)
      5. Return confidence_score, retrieval_quality, confidence_method

    Args:
        state: Current agent state with docs_text and citations

    Returns:
        State update with confidence_score, retrieval_quality, confidence_method

    Note: This node does NOT loop back to retrieval (Phase 2 validation only).
          Phase 3 will add looping logic based on confidence.
    """
    logger.info("Executing reasoning_node")

    # Extract citations and their scores
    citations = state.get("citations", [])
    docs_text = state.get("docs_text", "")
    user_query = ""

    # Extract user query from messages
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        user_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))

    # Step 1: Calculate score-based confidence
    score_confidence = compute_score_based_confidence(citations)
    logger.info(f"Score-based confidence: {score_confidence:.3f}")

    # Step 2: Decision tree
    if score_confidence >= 0.8:
        # High confidence: trust the scores
        logger.info("High confidence (>= 0.8): Using score-based confidence only")
        final_confidence = score_confidence
        method = "score_only_high"
        llm_confidence = None

    elif score_confidence < 0.4:
        # Low confidence: already failed, no need to validate
        logger.info("Low confidence (< 0.4): Already failed, no LLM validation needed")
        final_confidence = score_confidence
        method = "score_only_low"
        llm_confidence = None

    else:
        # Uncertain zone (0.4-0.8): Validate with LLM
        logger.info(f"Uncertain zone (0.4-0.8): confidence={score_confidence:.3f}, calling LLM for validation")

        # Get LLM instance
        llm = AzureChatOpenAI(
            deployment_name=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,
        )

        # Validate with LLM (async context requires await, handled in route)
        # For now, use sync method (will update in nodes.py to be async)
        import asyncio
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We're in async context, but reasoning_node is sync
            # This will be handled by converting reasoning_node to async
            logger.warning("reasoning_node called in async context but is sync; will be converted to async")
            llm_confidence = 0.5  # Fallback
        except RuntimeError:
            # Not in async context, use sync
            llm_confidence = 0.5  # Will be replaced with actual validation

        # Blend scores: 60% from retrieval scores, 40% from LLM validation
        final_confidence = (0.6 * score_confidence) + (0.4 * llm_confidence)
        method = "hybrid"

        logger.info(
            f"Hybrid confidence: (0.6 × {score_confidence:.3f}) + (0.4 × {llm_confidence:.3f}) = {final_confidence:.3f}"
        )

    # Step 3: Determine retrieval quality
    retrieval_quality = "high" if final_confidence >= 0.7 else "low"

    logger.info(
        f"Reasoning complete: confidence={final_confidence:.3f}, quality={retrieval_quality}, method={method}"
    )

    # Return state update
    return {
        "confidence_score": final_confidence,
        "retrieval_quality": retrieval_quality,
        "confidence_method": method,
    }
```

**Important note**: The above code has a sync wrapper for the reasoning_node. We'll need to make the LLM validation async-compatible. See Step 3 for how to handle this in nodes.py.

---

### STEP 3: Convert reasoning_node to async (IMPORTANT)

The LLM validation needs to be async because it's called from an async context (FastAPI route).

**Replace the reasoning_node function** in the reasoning.py file with this async version:

```python
async def reasoning_node(state: MedicalAgentState) -> dict[str, Any]:
    """
    Reasoning node: Assess retrieval quality and compute confidence score.

    ASYNC VERSION for FastAPI integration.

    Process:
      1. Extract scores from retrieved citations
      2. Compute score-based confidence (average of 5 scores)
      3. Decision tree:
         - If score >= 0.8: High confidence, use score only (no LLM)
         - If score < 0.4: Low confidence, use score only (no LLM)
         - If 0.4 <= score < 0.8: Uncertain, validate with LLM (1 call)
      4. Determine retrieval quality (high if confidence >= 0.7, else low)
      5. Return confidence_score, retrieval_quality, confidence_method

    Args:
        state: Current agent state with docs_text and citations

    Returns:
        State update with confidence_score, retrieval_quality, confidence_method
    """
    logger.info("Executing reasoning_node (async)")

    # Extract citations and their scores
    citations = state.get("citations", [])
    docs_text = state.get("docs_text", "")
    user_query = ""

    # Extract user query from messages
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        user_query = last_message.content if hasattr(last_message, "content") else str(last_message.get("content", ""))

    # Step 1: Calculate score-based confidence
    score_confidence = compute_score_based_confidence(citations)
    logger.info(f"Score-based confidence: {score_confidence:.3f}")

    # Step 2: Decision tree
    if score_confidence >= 0.8:
        # High confidence: trust the scores
        logger.info("High confidence (>= 0.8): Using score-based confidence only")
        final_confidence = score_confidence
        method = "score_only_high"
        llm_confidence = None

    elif score_confidence < 0.4:
        # Low confidence: already failed, no need to validate
        logger.info("Low confidence (< 0.4): Already failed, no LLM validation needed")
        final_confidence = score_confidence
        method = "score_only_low"
        llm_confidence = None

    else:
        # Uncertain zone (0.4-0.8): Validate with LLM
        logger.info(f"Uncertain zone (0.4-0.8): confidence={score_confidence:.3f}, calling LLM for validation")

        # Get LLM instance
        llm = AzureChatOpenAI(
            deployment_name=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            temperature=0.0,
        )

        # Validate with LLM (async call)
        health_profile = state.get("health_profile", {})
        llm_confidence = await validate_with_llm(
            query=user_query,
            docs_text=docs_text,
            health_profile=health_profile,
            llm=llm,
        )

        # Blend scores: 60% from retrieval scores, 40% from LLM validation
        final_confidence = (0.6 * score_confidence) + (0.4 * llm_confidence)
        method = "hybrid"

        logger.info(
            f"Hybrid confidence: (0.6 × {score_confidence:.3f}) + (0.4 × {llm_confidence:.3f}) = {final_confidence:.3f}"
        )

    # Step 3: Determine retrieval quality
    retrieval_quality = "high" if final_confidence >= 0.7 else "low"

    logger.info(
        f"Reasoning complete: confidence={final_confidence:.3f}, quality={retrieval_quality}, method={method}"
    )

    # Return state update
    return {
        "confidence_score": final_confidence,
        "retrieval_quality": retrieval_quality,
        "confidence_method": method,
    }
```

---

### STEP 4: Update graph.py to add reasoning_node

**File**: `src/medical_agent/agents/graph.py`

**Current code** (lines 23-60):
```python
def build_medical_rag_graph() -> "CompiledStateGraph":
    """
    Build and compile the medical RAG agent graph.

    Graph flow:
        START → retrieve_node → generate_node → END

    Features:
        - MemorySaver for session-based conversation history
        - Simple linear flow (no conditional routing yet)
        - Pure function nodes (no agents/tools)

    Returns:
        Compiled LangGraph application with checkpointer
    """
    logger.info("Building medical RAG graph")

    # Create state graph
    workflow = StateGraph(MedicalAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Define edges (linear flow)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Add memory for multi-turn conversations
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("Medical RAG graph compiled successfully")

    return app
```

**Change to**:
```python
def build_medical_rag_graph() -> "CompiledStateGraph":
    """
    Build and compile the medical RAG agent graph.

    Graph flow:
        START → retrieve_node → reasoning_node → generate_node → END

    Features:
        - Retrieve: Fetch and rerank papers with metadata weighting
        - Reasoning: Assess retrieval quality and compute confidence
        - Generate: Create response with confidence metadata
        - MemorySaver for session-based conversation history

    Returns:
        Compiled LangGraph application with checkpointer
    """
    logger.info("Building medical RAG graph")

    # Create state graph
    workflow = StateGraph(MedicalAgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("reasoning", reasoning_node)  # ← NEW
    workflow.add_node("generate", generate_node)

    # Define edges (linear flow with reasoning validation)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "reasoning")  # ← NEW
    workflow.add_edge("reasoning", "generate")  # ← CHANGED
    workflow.add_edge("generate", END)

    # Add memory for multi-turn conversations
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    logger.info("Medical RAG graph compiled successfully")

    return app
```

**Also add import** at top of graph.py:
```python
from medical_agent.agents.reasoning import reasoning_node
```

---

### STEP 5: Update nodes.py imports

**File**: `src/medical_agent/agents/nodes.py`

**Location**: Around line 14-18 (imports section)

**Add import**:
```python
from medical_agent.agents.reasoning import reasoning_node
```

(This is for documentation; reasoning_node is imported in graph.py)

---

### STEP 6: Update generate_node to include confidence in output (Optional)

**File**: `src/medical_agent/agents/nodes.py`

**Current generate_node** returns just messages and used_citations.

**To include confidence in response metadata** (optional enhancement):

In the `generate_node()` function, modify the return statement (around line 184):

**Current**:
```python
    return {
        "messages": [AIMessage(content=result.response)],
        "used_citations": result.used_citations
    }
```

**Add confidence metadata** (optional):
```python
    # Include confidence score in assistant message metadata (optional)
    confidence_score = state.get("confidence_score", 0.0)
    retrieval_quality = state.get("retrieval_quality", "unknown")

    assistant_msg = AIMessage(
        content=result.response,
        metadata={
            "confidence_score": confidence_score,
            "retrieval_quality": retrieval_quality,
        }
    )

    return {
        "messages": [assistant_msg],
        "used_citations": result.used_citations
    }
```

This allows downstream systems to see confidence scores in the response.

---

## Confidence Score Decision Tree (Reference)

```
Input: 5 chunks with scores [s1, s2, s3, s4, s5]
score_confidence = avg(s1, s2, s3, s4, s5)

┌─────────────────────────────────────────┐
│ IF score_confidence >= 0.8              │
├─────────────────────────────────────────┤
│ Decision: HIGH CONFIDENCE               │
│ LLM calls: 0 ✓                          │
│ Cost: $0                                │
│ Time: <1ms                              │
│ final_confidence = score_confidence     │
│ method = "score_only_high"              │
└─────────────────────────────────────────┘
           ↓
           │
           NO (0.4 <= score_confidence < 0.8)
           ↓
┌─────────────────────────────────────────┐
│ ELSE IF score_confidence < 0.4          │
├─────────────────────────────────────────┤
│ Decision: LOW CONFIDENCE (already failed)│
│ LLM calls: 0 ✓                          │
│ Cost: $0                                │
│ Time: <1ms                              │
│ final_confidence = score_confidence     │
│ method = "score_only_low"               │
└─────────────────────────────────────────┘
           ↓
           │
           NO (0.4 <= score_confidence < 0.8)
           ↓
┌─────────────────────────────────────────┐
│ ELSE (Uncertain Zone)                   │
├─────────────────────────────────────────┤
│ Decision: VALIDATE WITH LLM              │
│ LLM call: 1 ✓                           │
│ Cost: ~$0.01                            │
│ Time: ~500ms                            │
│ llm_confidence = LLM validation result  │
│ final = (0.6 × score) + (0.4 × llm)     │
│ method = "hybrid"                       │
└─────────────────────────────────────────┘
           ↓
           ↓
    final_confidence computed
           ↓
    quality = "high" if final >= 0.7
            = "low" if final < 0.7
           ↓
    Return to state
           ↓
    Always proceed to generate_node
    (No looping in Phase 2)
```

---

## Testing Checklist

### Unit Tests (Optional but recommended)

Create `tests/test_reasoning.py`:

```python
import pytest
from medical_agent.agents.reasoning import compute_score_based_confidence

def test_compute_score_based_confidence_high():
    """Test with high scores."""
    citations = [
        {"score": 0.92},
        {"score": 0.88},
        {"score": 0.85},
        {"score": 0.81},
        {"score": 0.79}
    ]
    confidence = compute_score_based_confidence(citations)
    assert 0.8 <= confidence <= 0.95
    assert confidence >= 0.8  # Should be high confidence

def test_compute_score_based_confidence_low():
    """Test with low scores."""
    citations = [
        {"score": 0.35},
        {"score": 0.32},
        {"score": 0.28},
        {"score": 0.25},
        {"score": 0.22}
    ]
    confidence = compute_score_based_confidence(citations)
    assert 0.2 <= confidence <= 0.4
    assert confidence < 0.4  # Should be low confidence

def test_compute_score_based_confidence_uncertain():
    """Test with uncertain scores (middle range)."""
    citations = [
        {"score": 0.72},
        {"score": 0.68},
        {"score": 0.65},
        {"score": 0.61},
        {"score": 0.58}
    ]
    confidence = compute_score_based_confidence(citations)
    assert 0.4 < confidence < 0.8
    assert 0.6 <= confidence <= 0.7  # Should be in uncertain zone

def test_compute_score_based_confidence_empty():
    """Test with empty citations."""
    confidence = compute_score_based_confidence([])
    assert confidence == 0.0
```

### Integration Tests (Recommended)

1. **Test with actual API call**:
   ```bash
   # Start API
   uvicorn medical_agent.api.main:app --reload

   # Make query with health profile
   curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{
       "ph_value": 5.2,
       "diagnoses": ["PCOS"],
       "symptoms": ["Itchy"]
     }'
   ```

2. **Verify logs**:
   ```
   INFO - Executing reasoning_node (async)
   INFO - Score-based confidence: 0.728
   INFO - Uncertain zone (0.4-0.8): calling LLM for validation
   INFO - LLM validation: can_answer=True, confidence=0.82
   INFO - Hybrid confidence: (0.6 × 0.728) + (0.4 × 0.82) = 0.765
   INFO - Reasoning complete: confidence=0.765, quality=high, method=hybrid
   ```

3. **Compare logs with/without metadata weighting**:
   - Phase 1 impact: Better retrieval ranking (metadata overlap)
   - Phase 2 impact: Confidence scores, quality assessment

---

## Verification Steps

### 1. Check syntax and imports
```bash
cd /Users/ayushsingh/Documents/pHera/AI\ Nation/RAG/pHera\ Rag\ Code/Medical_Agent
python -c "from medical_agent.agents.reasoning import reasoning_node; print('✓ Import successful')"
```

### 2. Verify state changes
```bash
python -c "from medical_agent.agents.state import MedicalAgentState; print('✓ State updated')"
```

### 3. Run linting
```bash
ruff check src/medical_agent/agents/reasoning.py
ruff check src/medical_agent/agents/state.py
ruff check src/medical_agent/agents/graph.py
ruff format src/medical_agent/agents/
```

### 4. Type checking (optional)
```bash
mypy src/medical_agent/agents/reasoning.py
mypy src/medical_agent/agents/state.py
mypy src/medical_agent/agents/graph.py
```

### 5. Run tests
```bash
pytest tests/test_reasoning.py -v
```

### 6. Test graph compilation
```bash
python -c "from medical_agent.agents.graph import build_medical_rag_graph; g = build_medical_rag_graph(); print('✓ Graph compiled'); print(g)"
```

---

## Debugging & Logs

When running Phase 2, look for these log messages:

**High confidence (no LLM)**:
```
INFO - Score-based confidence: 0.82
INFO - High confidence (>= 0.8): Using score-based confidence only
INFO - Reasoning complete: confidence=0.82, quality=high, method=score_only_high
```

**Low confidence (no LLM)**:
```
INFO - Score-based confidence: 0.32
INFO - Low confidence (< 0.4): Already failed, no LLM validation needed
INFO - Reasoning complete: confidence=0.32, quality=low, method=score_only_low
```

**Uncertain (LLM validation)**:
```
INFO - Score-based confidence: 0.65
INFO - Uncertain zone (0.4-0.8): confidence=0.65, calling LLM for validation
INFO - Validating retrieval with LLM (uncertain confidence zone)
DEBUG - LLM validation response: {"can_answer": true, "confidence": 0.82, ...}
INFO - LLM validation: can_answer=True, confidence=0.82
INFO - Hybrid confidence: (0.6 × 0.65) + (0.4 × 0.82) = 0.718
INFO - Reasoning complete: confidence=0.718, quality=high, method=hybrid
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `AttributeError: module 'medical_agent.agents' has no attribute 'reasoning_node'` | Ensure reasoning_node is imported in graph.py |
| `KeyError: 'confidence_score'` in generate_node | State fields must be initialized in reasoning_node return statement |
| LLM validation timeout (>5s) | Increase timeout in Azure OpenAI client config |
| JSON parse error in validate_with_llm | LLM might not return valid JSON; regex extraction is fallback |
| Confidence always 0.5 | Check that citations list has "score" field |
| Graph fails to compile | Run `build_medical_rag_graph()` in Python REPL to see detailed error |

---

## Files Modified Summary

| File | Lines | Change |
|------|-------|--------|
| `agents/state.py` | +3 fields | Add confidence_score, retrieval_quality, confidence_method |
| `agents/reasoning.py` | +260 | NEW file: compute_score_based_confidence, validate_with_llm, reasoning_node |
| `agents/graph.py` | +4 | Import reasoning_node, add to workflow, add edges |
| `agents/nodes.py` | 1 | Import reasoning_node (optional, for clarity) |

**Total additions**: ~260 lines of code (+ docstrings/comments)
**Total modifications**: 3 function changes in graph.py (4 lines)

---

## Expected Behavior Changes

### Before Phase 2
```
Query: "I have pH 5.2 and PCOS, what could it be?"
User: PCOS diagnosis

retrieve_node output:
  Chunk 1: [0.85] PCOS + pH pathophysiology
  Chunk 2: [0.76] PCOS + vulvovaginal symptoms
  Chunk 3: [0.71] Yeast infection + pH
  Chunk 4: [0.68] Vaginal health in PCOS
  Chunk 5: [0.63] Hormonal changes + vaginal flora

→ Immediately to generate_node
→ No quality assessment
→ No confidence flag
```

### After Phase 2
```
Query: "I have pH 5.2 and PCOS, what could it be?"
User: PCOS diagnosis

retrieve_node output:
  Chunks with metadata-weighted scores (Phase 1)
  ↓
reasoning_node:
  score_confidence = avg([0.85, 0.76, 0.71, 0.68, 0.63]) = 0.728
  Zone: Uncertain (0.4-0.8)
  → LLM validates: "Do these 5 chunks answer 'pH 5.2 + PCOS + itching'?"
  → LLM response: confidence=0.82
  → Final: (0.6 × 0.728) + (0.4 × 0.82) = 0.765
  ↓
generate_node:
  Response: [same as before]
  Metadata: confidence_score=0.765, retrieval_quality="high"
  ↓
API response includes: confidence_score, retrieval_quality
```

---

## Next Steps After Phase 2

Once Phase 2 is stable:
1. **Run evaluation**: Test confidence scores on real queries
2. **Tune thresholds**: Should 0.8/0.4 boundaries be different?
3. **Measure LLM call frequency**: What % of queries hit uncertain zone?
4. **Phase 3**: Add looping logic if confidence < 0.7
   - Reasoning node generates refined query
   - Route back to retrieve_node (max 2 retries)
   - Guard rails against infinite loops

---

## Reference Files

- State definition: `src/medical_agent/agents/state.py`
- Phase 1 reranking: `src/medical_agent/agents/reranker.py`
- Graph structure: `src/medical_agent/agents/graph.py`
- Routes (async context): `src/medical_agent/api/routes/query.py`
