# FemTech Medical RAG Agent - Project Understanding

## Executive Summary

This project is a FemTech startup initiative aimed at transforming women's vaginal health monitoring. The platform enables women to measure their vaginal pH levels from home using standard colorimetric test strips and a smartphone camera. A computer vision model (built separately) analyzes test strips with high precision.

**Our scope** begins after receiving the pH value from the CV model. We are building an AI-powered medical research intelligence system that interprets pH readings against curated research papers to provide personalized, evidence-based health insights.

---

## The Problem We're Solving

### Current State
- Women rely on litmus tests and assumptions for vaginal health monitoring
- Medical research is constantly evolving with new studies often contradicting older methods
- Doctors struggle to keep up with the latest research across all domains
- There's a critical gap between cutting-edge research and personalized health guidance

### Our Solution
An AI system that:
1. Ingests and understands curated medical research papers
2. Correlates user-specific health data with research findings
3. Provides personalized, evidence-based health insights
4. Bridges the knowledge gap between latest research and actionable guidance

---

## Target Users

| User Type | Role |
|-----------|------|
| Primary | Women (patients) using the mobile app for self-monitoring |
| Secondary | Doctors/medical professionals reviewing and validating AI outputs |

---

## System Inputs and Outputs

### Inputs

1. **pH Value** (from CV model)
   - Numeric value representing vaginal pH level
   - Provided by the computer vision component

2. **User Health Profile**
   - Age
   - Ethnicity
   - Symptoms (list)
   - Additional health factors (schema to be finalized with medical advisors)

### Outputs

The AI generates a comprehensive response containing:

1. **Short Summary** - Key findings in accessible language
2. **Citations** - Specific papers and passages used for the insights
3. **Risk Level Assessment** - NORMAL | MONITOR | CONCERNING | URGENT
4. **Detailed Explanation** - In-depth analysis of findings
5. **Actionable Steps** - Specific recommendations (e.g., "monitor for X days", "consult a gynecologist")
6. **Medical Disclaimers** - Required legal/informational statements

---

## Knowledge Base

### Source
- PubMed research papers on vaginal health

### Curation Process
- Medical advisors select and curate papers
- Focus on recent, high-quality studies
- Papers with tables and graphs containing critical data

### Volume
- 250-500 papers for MVP
- Continuously updated corpus

### Update Frequency
- Weekly additions
- Manual upload by medical advisors

### Format
- PDF documents
- Structured content (title, abstract, findings, methodology)
- Heavy on tables and graphical data

---

## Regulatory and Compliance Stance

### Classification
- **Purely informational** - NOT a diagnostic tool
- Does not replace professional medical advice

### GDPR Compliance
- Designed to meet strict GDPR standards from day one
- User data privacy is paramount
- Azure services chosen specifically for EU compliance

### Guardrails (Critical Requirements)

The AI must **NEVER**:
- Prescribe medication
- Diagnose specific conditions
- Give advice outside vaginal health scope
- Make claims not grounded in the provided research papers

The AI must **ALWAYS**:
- Base responses only on information from curated research papers
- Include appropriate medical disclaimers
- Cite specific sources for claims
- Recommend professional consultation when appropriate

---

## Risk Assessment Framework

### Risk Levels

| Level | Description | Response Approach |
|-------|-------------|-------------------|
| NORMAL | pH within healthy range, no concerning symptoms | Reassurance, general wellness tips |
| MONITOR | Slight deviation or mild symptoms | Self-monitoring guidance, when to escalate |
| CONCERNING | Significant pH deviation with symptoms | Strong recommendation to consult healthcare provider |
| URGENT | Critical readings indicating potential infection | Immediate professional consultation urged |

### pH Reference Ranges

| pH Range | Symptom Status | Typical Assessment |
|----------|----------------|-------------------|
| 3.8 - 4.5 | None | Normal/Healthy |
| 3.8 - 4.5 | Mild symptoms | Monitor |
| 4.5 - 5.0 | Any | Concerning |
| Above 5.0 | Any | Urgent |

> **Note:** Final thresholds to be validated with medical advisors.

---

## AI System Architecture

### Two Core Pipelines

#### 1. Ingestion Pipeline (Batch Process)
- Runs weekly when new papers are added
- Processes PDFs from cloud storage
- Extracts text and table data
- Chunks content by section
- Generates embeddings
- Stores in vector database

#### 2. Query Pipeline (Real-time)
- Triggered on each user query
- Analyzes user context (pH + profile)
- Retrieves relevant research
- Assesses risk level
- Reasons over evidence
- Generates personalized response

### Agent Workflow Nodes

1. **Query Analyzer**
   - Parses pH value
   - Extracts symptoms from profile
   - Identifies key medical concepts
   - Generates optimized search queries

2. **Retriever**
   - Performs vector similarity search
   - Retrieves top-k relevant paper chunks
   - Includes paper metadata for citations
   - Extracts relevant table data

3. **Risk Assessor**
   - Evaluates pH against normal ranges
   - Cross-references with reported symptoms
   - Considers user profile factors
   - Determines appropriate risk level

4. **Reasoner**
   - Analyzes retrieved evidence
   - Correlates findings with user profile
   - Handles conflicting study results
   - Synthesizes evidence-based insights

5. **Response Generator**
   - Formats summary with proper citations
   - Includes risk level badge/indicator
   - Adds actionable recommendations
   - Appends medical disclaimers
   - Applies final guardrails check

---

## Data Architecture

### Data Entities

1. **Users**
   - Basic user information
   - Account management
   - Placeholder for future authentication

2. **Health Profiles**
   - Linked to users
   - Stores age, ethnicity, symptoms
   - Flexible schema for additional health factors
   - Versioned for query history accuracy

3. **Papers**
   - Research paper metadata
   - Title, authors, journal, publication year, DOI
   - Abstract
   - Cloud storage path to original PDF
   - Parsed content structure

4. **Paper Chunks**
   - Individual sections/tables from papers
   - Chunk type (abstract, introduction, results, discussion, table, conclusion)
   - Content text
   - Vector embedding for similarity search
   - Metadata for filtering

5. **Query Logs**
   - Complete audit trail
   - User ID and timestamp
   - pH value at query time
   - Health profile snapshot
   - Retrieved chunks used
   - Risk level determined
   - Full AI response
   - User feedback (optional)

---

## Paper Processing Strategy

### Document Parsing
- LlamaParser (LlamaCloud) for high-quality extraction
- Special handling for tables (common in medical papers)
- Structured output for consistent chunking

### Chunking Approach

Hierarchical section-based chunking:

**Level 1: Paper Metadata**
- Title, authors, journal, year, DOI
- Abstract (standalone chunk - often sufficient for quick retrieval)

**Level 2: Section Chunks**
- Introduction
- Methods (lower priority for user queries)
- Results (HIGH VALUE - findings and data)
- Discussion (HIGH VALUE - interpretations)
- Conclusions

**Level 3: Tables and Figures**
- Tables extracted as structured data
- Key findings from figures as text descriptions
- Linked to parent paper for context

### Duplicate Handling
- System detects when same paper is uploaded again
- Notifies admin/medical advisor
- User decides: keep new version (if updated) or reject

---

## Handling Conflicting Evidence

Medical research often contains contradictory findings. Strategy for MVP:

1. **Collect and Present** - Show all relevant viewpoints from research
2. **Pattern Recognition** - During testing, identify common conflict patterns
3. **Doctor Review** - Medical advisors help establish prioritization rules
4. **Future Enhancement** - Build weighting system based on recency, sample size, journal impact

---

## User Interaction Model

### MVP Approach
- Single query → Single response
- No conversation history within session
- Complete, comprehensive response for each query

### Future Enhancement
- Conversational follow-ups
- "What about if I also have symptom X?"
- Session-based context memory

---

## Response Time Expectations

- **Target:** 10-15 seconds acceptable
- **Rationale:** Deep reasoning required, infrequent use (monthly per user)
- Users prioritize accuracy over speed

---

## Evaluation Framework

### Golden Set
- 50 question-answer pairs
- Validated by medical doctors
- Covers range of pH values, symptoms, profiles
- Tests edge cases and risk levels

### Metrics to Track
- Retrieval accuracy (right papers found)
- Reasoning quality (correct correlations made)
- Risk assessment accuracy
- Citation correctness
- Guardrail compliance

### Ongoing Validation
- Periodic doctor review of AI outputs
- User feedback collection
- Continuous improvement based on findings

---

## Privacy and Security

### Data Protection
- GDPR compliant by design
- Minimal data collection
- User data encrypted at rest and in transit
- No data sharing with third parties

### Audit Trail
- All queries logged for compliance
- Enables review and improvement
- Supports regulatory requirements

---

## Team and Resources

### Development Team
- Single developer (primary builder)

### Medical Oversight
- Medical advisor actively involved in paper curation
- Periodic review of AI outputs
- Validation of golden set answers

### Budget
- $600 Azure credits (across 3 accounts over 3 months)
- Covers LLM usage, embeddings, document parsing
- Minimal GCP costs for storage and database

---

## Timeline

| Phase | Duration | Focus |
|-------|----------|-------|
| Week 1 | 7 days | Foundation - Project setup, infrastructure, database |
| Week 2 | 7 days | Ingestion Pipeline - PDF parsing, chunking, embeddings |
| Week 3 | 7 days | Agent Workflow - All nodes, prompts, guardrails |
| Week 4 | 7 days | Integration - APIs, evaluation, testing |
| Buffer | 10 days | Polish, optimization, feedback incorporation |

**Total:** ~1 month with 10-day buffer

---

## Key Success Criteria

1. **Accuracy** - Correct retrieval and correlation of research findings
2. **Relevance** - Insights specific to user's profile and pH reading
3. **Safety** - No medical advice violations, proper disclaimers
4. **Reliability** - Consistent, reproducible results
5. **Performance** - Responses within 15 seconds
6. **Traceability** - Clear citations for all claims

---

## Future Roadmap (Post-MVP)

1. Conversational follow-up capability
2. Multi-language support
3. Integration with Zitadel authentication
4. Advanced analytics dashboard
5. Expanded paper corpus
6. Real-time paper ingestion from PubMed
7. Personalized health trend tracking

---

## Open Items (To Be Finalized)

1. Complete user health profile schema (with medical advisor)
2. Golden set 50 Q&A pairs (in progress)
3. Exact response format specification (JSON vs Markdown)
4. Specific medical disclaimer language
5. Prioritization rules for conflicting studies
6. Azure OpenAI model availability confirmation

