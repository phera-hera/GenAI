Subject: Technical Review & Architectural Recommendations for Medical RAG Agent
Hi Team,
I've completed a comprehensive review of our Medical RAG Agent codebase and identified significant architectural issues that are impacting development velocity, maintainability, and cost efficiency. This email outlines the problems observed and proposes a strategic refactoring approach using LlamaIndex's native capabilities.
1. Problems Observed
1.1 Excessive Custom Wrapper Layers
Location: src/medical_agent/ingestion/storage/vector_store.py
We've implemented MedicalPGVectorStore as a 521-line wrapper around LlamaIndex's PGVectorStore. The wrapper attempts to:
Adapt LlamaIndex's vector store to our existing schema
Convert between NodeWithScore objects and custom SearchResult dataclasses
Manually implement hybrid search functionality
Handle async/sync conversions with ThreadPoolExecutor
Why this is problematic:
LlamaIndex's PGVectorStore already supports:
Custom metadata through node.metadata (no schema adaptation needed)
Hybrid search via hybrid_search=True parameter (built-in BM25 + vector fusion)
Direct async operations
Similar issues in:
MedicalRAGIndex (src/medical_agent/rag/index.py - 312 lines) - wraps VectorStoreIndex unnecessarily
MedicalIngestionPipeline (src/medical_agent/ingestion/pipeline.py - 521 lines) - wraps IngestionPipeline with minimal value-add
MedicalPaperRetriever (src/medical_agent/rag/retriever.py - 558 lines) - reimplements retrieval logic that LlamaIndex provides
1.2 Over-Engineered Agent Orchestration
Location: src/medical_agent/agent/
Our LangGraph implementation uses 5 sequential nodes:
query_analyzer.py (349 lines) - Parses pH, extracts symptoms, generates search queries
retriever.py (130+ lines) - Retrieves research chunks
risk_assessor.py (265+ lines) - Assesses risk level
reasoner.py (257+ lines) - Analyzes evidence
response_generator.py (365+ lines) - Formats final response
Problems:
5 separate LLM calls per query (expensive, slow: ~15-30 seconds total)
Manual JSON parsing in every node with fallback logic
No true agentic behavior - fixed pipeline, not iterative refinement
Complex state management - AgentState TypedDict with 20+ fields in state.py (380 lines)
Unnecessary async wrappers - every node has a sync wrapper calling asyncio.run() for LangGraph compatibility (LangGraph supports async natively)
What "Agentic RAG" should mean:
An agent that autonomously decides: "Do I need more information? Should I refine my search? Is this sufficient?" - iterative, not sequential.
What we built:
A rigid pipeline that always executes all 5 nodes regardless of need.
1.3 Manual Prompt Engineering with Brittle Parsing
Location: src/medical_agent/agent/prompts/system_prompts.py
Each node manually constructs prompts, forces JSON output, and parses responses with try-catch blocks:
LLM Response → Extract JSON from markdown → json.loads() → Handle parsing errors → Fallback to defaults
Issues:
No structured output guarantees (OpenAI's function calling not used)
No Pydantic validation
Fallback logic in every node adds complexity
Error-prone (we see multiple "Failed to parse JSON" warnings in logs)
1.4 Redundant Data Structure Conversions
The conversion chain:
Database Row → SearchResult → NodeWithScore → RetrievedChunk → dict → back to different objects
Example flow:
vector_store.py: Database → SearchResult (custom dataclass)
retriever.py: SearchResult → NodeWithScore (LlamaIndex)
Agent nodes: NodeWithScore → RetrievedChunk → dict
Final response: dict → FinalResponse dataclass
Result: Constant serialization overhead and potential data loss at each conversion boundary.
2. Recommended Solution: Simplified Architecture
2.1 Strategic Approach
Phase 1 (MVP - Now): Use LlamaIndex ReActAgent exclusively
Phase 2 (If needed): Wrap ReActAgent as a tool in LangGraph for advanced orchestration
This follows the principle: Start simple, add complexity only when necessary.
2.2 Phase 1: LlamaIndex ReActAgent Implementation
2.2.1 Ingestion Pipeline (Keep Docling, Simplify Storage)
Use native LlamaIndex components:
Document Parsing:
Keep Docling (DoclingReader) - we're already using it correctly in pipeline.py
Keep DoclingNodeParser with HybridChunker - this is working well
Storage:
Remove: MedicalPGVectorStore wrapper (521 lines deleted)
Use: PGVectorStore.from_params() directly with:
  hybrid_search=True  # Built-in BM25 + vector fusion  table_name="paper_chunks"
Metadata:
Store medical metadata directly in node.metadata:
  node.metadata = {      "paper_id": str(paper_id),      "ethnicities": ["Asian", "Caucasian"],      "diagnoses": ["BV"],      "symptoms": ["discharge"],      # ... all 8 medical fields  }
Result: Ingestion pipeline reduces from ~1,500 lines to ~300 lines while maintaining all functionality.
2.2.2 Query Engine with Hybrid Search
Remove:
MedicalRAGIndex (312 lines)
MedicalPaperRetriever (558 lines)
MedicalQueryEngine custom wrapper
Replace with:
Native LlamaIndex query engine setup:1. PGVectorStore (hybrid_search=True)2. VectorStoreIndex.from_vector_store()3. index.as_query_engine() or index.as_chat_engine()
Filtering:
LlamaIndex supports metadata filtering natively:
filters = MetadataFilters(    filters=[        MetadataFilter(key="ethnicities", value="Asian", operator=FilterOperator.CONTAINS),        MetadataFilter(key="ph_category", value="elevated")    ])retriever = index.as_retriever(filters=filters)
This replaces our custom build_metadata_filters() function in metadata_filters.py.
2.2.3 ReActAgent for Agentic RAG
Core concept (from LlamaIndex documentation):
ReActAgent implements the Reasoning-Acting pattern where the LLM:
Reasons about the problem
Acts by calling tools
Observes results
Iterates until satisfied
Implementation:
Step 1: Create Query Engine Tool
Convert our query engine into a tool:- Name: "medical_research_search"- Description: "Search 250+ peer-reviewed medical papers about vaginal pH, symptoms, and women's health"- Function: query_engine.query()
Step 2: Optional Additional Tools
- "ph_risk_calculator": Simple threshold logic (pH > 5.0 = concerning)- "symptom_analyzer": Pattern matching for severe symptoms
Step 3: Create ReActAgent
Agent gets:- Tools list (query engine + optional helpers)- LLM (Azure OpenAI GPT-4o)- System prompt with medical context- Memory (chat history)
Step 4: Single Query Interface
Input: pH value + health profile + optional query textAgent automatically:- Decides if research search is needed- Formulates search queries- Calls tools multiple times if needed- Synthesizes final response- Returns structured answer
Result: 5 nodes + 1,700+ lines of custom orchestration → 1 agent + ~200 lines of configuration
2.2.4 Structured Outputs with Pydantic
Replace manual JSON parsing with Pydantic models:
Define response structure:
class MedicalAnalysisResponse(BaseModel):    risk_level: Literal["NORMAL", "MONITOR", "CONCERNING", "URGENT"]    summary: str    key_findings: List[str]    recommendations: List[str]    citations: List[Citation]    disclaimer: str
Use OpenAI's structured output:
llm = OpenAI(    model="gpt-4o",    response_format=MedicalAnalysisResponse  # Guaranteed structure)
Result: No more JSON parsing errors, guaranteed response structure, type safety.
2.3 Phase 2: When to Add LangGraph (Future)
Only add LangGraph when we need:
Multi-turn conversations - User asks follow-up questions, system remembers context across sessions
Human-in-the-loop approval - "Review this medical advice before sending to user"
Multi-agent coordination - Research agent + Symptom checker agent + Recommendation agent working together
Complex conditional logic - "If high risk AND pregnant, trigger different workflow"
State persistence - Save user sessions, resume conversations days later
Audit trail - Detailed logging of agent decisions for medical compliance
How to integrate:
LangGraph Flow:[Input Validation] → [ReActAgent as Tool] → [Safety Check] → [Format Response]     ↑                      ↑                     ↑                 ↑   LangGraph          LlamaIndex            LangGraph         LangGraph
ReActAgent becomes a single node in LangGraph, wrapped with safety checks and formatting.
3. Benefits of This Approach
3.1 Immediate Benefits
Code Reduction:
Remove: ~3,000+ lines of custom wrappers and orchestration
Add: ~500 lines of ReActAgent configuration
Net: 2,500+ lines deleted (83% reduction in agent code)
Cost Reduction:
Current: 5 LLM calls per query (~$0.15-0.30 per query)
Proposed: 1-2 LLM calls per query (~$0.03-0.06 per query)
Savings: 80% reduction in LLM costs
Latency Improvement:
Current: 15-30 seconds (sequential nodes)
Proposed: 3-8 seconds (single agent with parallel tool calls)
Improvement: 60-75% faster responses
Maintainability:
Remove brittle JSON parsing and fallback logic across 5 files
Use framework-native patterns (easier onboarding)
Fewer custom abstractions to document and maintain
3.2 Future Scalability
Easy additions:
New tools (image analysis, symptom checker) = add to tools list
Multi-agent systems = wrap ReActAgent in LangGraph when needed
Conversation history = built into chat engine
Custom retrieval strategies = modify query engine, not wrapper
4. Migration Path
4.1 Minimal Disruption Strategy
Week 1-2: Ingestion Refactor
Remove MedicalPGVectorStore wrapper
Use PGVectorStore directly with hybrid_search=True
Flatten metadata structure (no nested conversions)
Keep existing database schema (backward compatible)
Test with existing 4 papers
Week 3-4: Agent Refactor
Create QueryEngineTool from existing query engine
Implement ReActAgent with tool
Add Pydantic response models
Test against current system outputs
Keep existing API endpoints (internal implementation change only)
Week 5: Cleanup & Documentation
Delete old agent nodes (5 files, 1,700+ lines)
Delete custom retrievers and wrappers (3 files, 1,400+ lines)
Update architecture documentation
Update API documentation
4.2 Risk Mitigation
Backward Compatibility:
Keep existing API contracts (POST /api/v1/query)
Keep existing database schema
Internal refactor only
Testing:
Use existing test cases in tests/
Compare outputs between old and new systems
Gradual rollout (A/B testing if possible)
Rollback Plan:
Keep old code in feature branch until new system validated
Easy revert if issues discovered
5. Technical References
5.1 LlamaIndex Documentation
ReActAgent:
Official tutorial: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/
With Query Engine tools: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/
Agentic RAG guide: https://llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6
Query Engine Tools:
Tool creation: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/
Agent architecture: https://docs.llamaindex.ai/en/latest/module_guides/deploying/agents/
PGVectorStore:
Setup guide: https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/
Hybrid search documentation (check latest docs for hybrid_search parameter)
5.2 Current Codebase References
Files to remove:
src/medical_agent/ingestion/storage/vector_store.py (521 lines)
src/medical_agent/rag/index.py (312 lines)
src/medical_agent/rag/retriever.py (558 lines)
src/medical_agent/agent/nodes/query_analyzer.py (349 lines)
src/medical_agent/agent/nodes/risk_assessor.py (265 lines)
src/medical_agent/agent/nodes/reasoner.py (257 lines)
src/medical_agent/agent/nodes/response_generator.py (365 lines)
Files to keep/modify:
src/medical_agent/ingestion/pipeline.py - simplify to use native PGVectorStore
src/medical_agent/core/config.py - keep as-is
src/medical_agent/infrastructure/ - keep all client wrappers
src/medical_agent/api/ - keep API layer, update internal calls
6. Conclusion
Our current architecture suffers from premature optimization - we built complex custom orchestration for a problem that LlamaIndex already solves. With only 4 papers and 0 customers, we should prioritize:
Simplicity - Use framework features, not custom code
Speed - Ship faster with less code to maintain
Flexibility - Easy to iterate based on real user feedback
The proposed ReActAgent approach gives us true agentic RAG behavior (iterative refinement, tool-based decisions) while reducing code by 83% and costs by 80%. We can always add LangGraph later when we have concrete requirements for multi-agent coordination or human-in-the-loop workflows.
Recommendation: Proceed with Phase 1 refactor. Focus on shipping a working MVP with ReActAgent, then evaluate Phase 2 (LangGraph) based on actual production needs.
Happy to discuss this further or answer any questions.
Best regards,