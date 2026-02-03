# Medical Agent - AI-Powered Vaginal Health Analysis System
**Technical Documentation**

## 1. Summary & Scope

The Medical Agent is an AI-powered REST API that provides evidence-based analysis of vaginal health based on pH values and optional health profile information. The system retrieves relevant medical research papers and uses Azure OpenAI to generate personalized responses with inline citations.

**Current Development Stage:**
- The system is fully functional for query processing and response generation
- Medical research papers and their embeddings are stored persistently in a database
- **User queries, health profiles, and conversation histories are NOT stored** (processed in-memory only)
- No user authentication is currently implemented
- After testing phase, user data persistence and authentication will be added

The application is stateless from a user perspective—each query is independent, and no user tracking occurs. The system focuses on immediate analysis and evidence-based guidance without retaining user interaction data during development.

---

## 2. End-to-End Data Flow

When a user submits a query via the REST API:

1. **User sends request** via HTTPS POST to `/api/v1/query` with:
   - pH value (required, 0-14 range)
   - Optional: user_message (follow-up question)
   - Optional: session_id (for multi-turn conversation)
   - Optional: health profile (age, diagnoses, symptoms, menstrual status, birth control, hormone therapy, fertility journey)

2. **API server validates input** using request validation and checks data types.

3. **Query is processed in-memory:**
   - System builds enhanced search query combining pH value, health context, and user question
   - No data is written to database at this stage

4. **System searches medical papers database:**
   - Searches PostgreSQL database containing ingested medical research papers
   - Uses hybrid search (keyword matching + semantic similarity)
   - Retrieves top 5 most relevant paper sections from pre-loaded research papers

5. **Azure OpenAI generates response:**
   - Retrieved paper excerpts are sent to Azure OpenAI GPT-4o
   - Health profile and pH context included in prompt
   - AI generates medical analysis with inline citations [1][2][3]
   - Response includes risk assessment and medical disclaimers

6. **Conversation state stored temporarily (in-memory only):**
   - For multi-turn conversations within same session
   - Stored in server memory (RAM) only
   - Cleared immediately when session ends or server restarts

7. **Response returned to user:**
   - pH value, generated analysis, citations, disclaimers, processing time
   - All data sent back to client as JSON

8. **No persistence of user data:**
   - User query is not saved to database
   - Health profile is not saved
   - Generated response is not logged
   - Conversation history exists only in current session memory

**Data Transmission Summary:**
User query and health profile are transmitted to the backend API. Query and health context are forwarded to Azure OpenAI for AI processing. Retrieved medical paper content is sent to Azure OpenAI. **No third-party analytics or tracking services receive user data.** Communication is limited to the FastAPI server, PostgreSQL database (for paper retrieval only), and Azure OpenAI API.

---

## 3. Data Storage, Backups & Retention Policy

### Current Data Persistence

**What IS Stored:**
- Medical research papers (metadata: title, authors, journal, DOI, abstract, publication year)
- Paper content chunks (extracted text sections, max 512 tokens each)
- Vector embeddings (3,072-dimensional mathematical representations for semantic search)
- Original PDF files of research papers (stored in GCP Cloud Storage)
- Medical metadata extracted from papers (diagnoses, symptoms, ethnicities mentioned in research)

**What is NOT Stored:**
- User queries (pH values, questions, health profiles)
- Generated AI responses
- Conversation histories
- User identities (no authentication system)
- Session data (in-memory only, cleared on disconnect)

### Data Storage Architecture

| Data Type | Storage Location | Persistence | Processing Location |
|-----------|------------------|-------------|---------------------|
| Medical papers (metadata) | PostgreSQL database | Permanent | GCP Cloud SQL (EU region) |
| Paper chunks (text) | PostgreSQL database | Permanent | GCP Cloud SQL (EU region) |
| Vector embeddings | PostgreSQL (pgvector extension) | Permanent | GCP Cloud SQL (EU region) |
| PDF files | GCP Cloud Storage | Permanent | European Union (GCP EU region) |
| User queries | Server memory (RAM) only | Session only (~seconds) | Server processing (EU) |
| Health profiles | Server memory (RAM) only | Session only (~seconds) | Server processing (EU) |
| Conversation history | Server memory (RAM) only | Session only | Server processing (EU) |
| Generated responses | Returned to client | Not stored | Azure OpenAI (EU region) |

### Backup Status

**Current (Development):**
- No automated backups configured
- GCP Cloud Storage has default multi-region redundancy within EU (Google-managed)
- PostgreSQL database has no backup retention policy active
- **Rationale:** Development stage with no user data persistence; only research papers stored

**Future (Production):**
- GCP Cloud SQL automated daily backups (30-day retention planned)
- Point-in-time recovery enabled
- Cross-region backup replication within EU for disaster recovery
- Encrypted backups (Google-managed keys)

### Data Processing Jurisdiction

All data processing and storage occurs exclusively within the European Union:
- **Database:** European Union (GCP Cloud SQL, EU region)
- **File Storage:** European Union (GCP Cloud Storage, EU region)
- **AI Processing:** Azure OpenAI (Microsoft Azure, EU region)
- **API Hosting:** European Union (GCP Cloud Run, EU region)

### Retention Timeline

| Data Category | Current Retention | Future Retention (When User Data Persisted) |
|---------------|-------------------|---------------------------------------------|
| Medical papers | Permanent (until manually deleted) | Permanent |
| User queries | 0 seconds (not stored) | Per compliance requirements (TBD: 1-7 years) |
| Health profiles | 0 seconds (not stored) | Until user deletion request |
| Conversation histories | Session only (~minutes) | 90 days (planned) |
| Audit logs | Not implemented | 7 years (compliance standard) |

**Deletion Policy:**
Currently, no user data exists to delete. When user data persistence is implemented, users will have the right to request deletion of all personal data, including query history, health profiles, and conversation logs (GDPR compliance).

---

## 4. Data Types & Governance Classification

| Data Category | Type | Storage Location | Duration | Classification | Handling |
|---------------|------|------------------|----------|----------------|----------|
| **Medical Papers (Metadata)** | Title, authors, journal, DOI, abstract | PostgreSQL `papers` table | Permanent | Public research data | Stored persistently |
| **Paper Content Chunks** | Extracted text sections (512 tokens max) | PostgreSQL `data_paper_chunks` table | Permanent | Public research data | Indexed for search |
| **Vector Embeddings** | 3,072-dimensional vectors | PostgreSQL `data_paper_chunks.embedding` (pgvector) | Permanent | Derived computational data | Used for semantic search |
| **PDF Files** | Original research papers | GCP Cloud Storage (EU) | Permanent | Public research data | Archived for reference |
| **pH Value** | Numeric (0-14 range) | Server memory only | Session only | Sensitive health data | Not persisted (development) |
| **User Questions** | Text input | Server memory only | Session only | Potentially sensitive health data | Not persisted (development) |
| **Health Profile - Age** | Numeric | Server memory only | Session only | PII/demographic data | Not persisted (development) |
| **Health Profile - Diagnoses** | List (e.g., PCOS, endometriosis) | Server memory only | Session only | PHI (Protected Health Information) | Not persisted (development) |
| **Health Profile - Symptoms** | List (e.g., discharge, odor, itching) | Server memory only | Session only | PHI | Not persisted (development) |
| **Health Profile - Menstrual Cycle** | Categorical | Server memory only | Session only | PHI | Not persisted (development) |
| **Health Profile - Birth Control** | List | Server memory only | Session only | PHI | Not persisted (development) |
| **Health Profile - Hormone Therapy** | List | Server memory only | Session only | PHI | Not persisted (development) |
| **Health Profile - Ethnicity** | List | Server memory only | Session only | PII/demographic data | Not persisted (development) |
| **Health Profile - Fertility Journey** | Text/structured data | Server memory only | Session only | PHI | Not persisted (development) |
| **Generated AI Response** | Text with inline citations | Returned to client only | Not stored | Health-related guidance | Not persisted (development) |
| **Conversation History** | Message array | Server memory (RAM) | Session only | Health conversation data | Cleared on disconnect |
| **Session ID** | UUID string | Client-generated, server memory | Session only | Session identifier | Not validated or tracked |
| **Citations** | Paper references with metadata | Returned to client only | Not stored | Research references | Extracted from stored papers |
| **Processing Metadata** | Timestamp, processing time | Returned to client only | Not stored | Operational data | Not persisted (development) |

### Privacy Controls & Constraints

- **Data Minimization:** No user data is collected beyond what is necessary for immediate query processing
- **No Cross-Session Tracking:** Session IDs are client-generated and not validated; no user tracking across sessions
- **No User Accounts:** No authentication or user management system (placeholder code exists but not active)
- **Voluntary Input:** All health profile fields are optional; users control what information they provide
- **No Third-Party Sharing:** User data is never shared with third parties (only Azure OpenAI receives de-identified query context)

### Future Data Governance (When User Data Persistence Added)

When query logging and user profiles are implemented post-testing:

| Future Data Type | Classification | Storage | Governance Requirements |
|------------------|----------------|---------|------------------------|
| **User Accounts** | PII | PostgreSQL `users` table | GDPR compliance, right to deletion |
| **Health Profiles (Persistent)** | PHI | PostgreSQL `health_profiles` table | GDPR-compliant, encryption at rest, access controls |
| **Query Logs** | PHI + operational data | PostgreSQL `query_logs` table | Audit trail, retention policy, anonymization option |
| **Conversation Histories (Persistent)** | PHI | PostgreSQL (TBD table) | Encryption, time-limited retention (90 days planned) |

---

## 5. Architecture, Build & Deployment

**System Components:**
- REST API (FastAPI, Python) deployed on GCP Cloud Run (EU region)
- LangGraph orchestrates retrieval and generation workflow
- LlamaIndex handles hybrid search (keyword + semantic)
- PostgreSQL with pgvector extension on GCP Cloud SQL (EU)
- Azure OpenAI GPT-4o for response generation (EU region)
- GCP Cloud Storage for PDF archival (EU region)

**Deployment:**
- Development: Local Docker or docker-compose
- Production: GCP Cloud Run (EU), Cloud SQL (EU), Cloud Storage (EU)
- Configuration via environment variables in `.env` file
- Database migrations managed via Alembic

---

## 6. Security & Privacy Posture

| Security Layer | Status | Implementation |
|----------------|--------|----------------|
| **Encryption in Transit** | Active | HTTPS/TLS enforced at GCP Cloud Run |
| **Encryption at Rest** | Default | GCP Cloud SQL default encryption (Google-managed keys) |
| **Authentication** | Not Implemented | Placeholder code exists; not enforced |
| **Authorization** | Not Implemented | All endpoints publicly accessible |
| **Rate Limiting** | Not Implemented | No request throttling |
| **Input Validation** | Active | Request validation via Pydantic |

**Current Security Gaps (Acceptable in Development):**
- No user authentication (mitigated by: no user data persistence)
- No rate limiting (mitigated by: development usage only)
- API keys in environment variables (mitigated by: `.env` excluded from version control)

**Privacy Guarantees:**
- No user data persistence (queries and health profiles never saved)
- No cross-session tracking (session IDs not stored)
- No third-party analytics
- Immediate memory clearance on session disconnect

**Before Production:**
- Implement Zitadel OAuth2 authentication
- Enable field-level encryption for health profiles
- Migrate Azure OpenAI keys to GCP Secret Manager
- Implement rate limiting and audit logging

---

## 7. Governance, Traceability & Auditability

**Current Traceability:**
- Git version control for code changes
- Alembic migrations for database schema changes
- Configuration tracked via `.env.example` (actual `.env` excluded from version control)

**Audit Logging:**
- Current: No user interaction audit trail (queries not logged)
- Future: `query_logs` table will track timestamps, user IDs, queries, responses, citations, and processing metadata

**Compliance:**
- Current: GDPR-ready by design (no user data stored)
- Future: Full GDPR compliance framework including right to access, erasure, rectification, and data portability

---

## 8. Current Implementation Status

**Functional Capabilities:**
- REST API for pH analysis (POST `/api/v1/query`)
- Medical paper ingestion (PDF upload, text extraction, embeddings)
- Hybrid search (keyword + semantic similarity)
- AI response generation with inline citations
- Multi-turn conversation (in-memory, same session)
- Health profile contextualization

**Not Yet Implemented:**
- User authentication
- Query logging to database
- Health profile persistence
- Audit logging
- Rate limiting

---

## 9. Assumptions & Constraints for Future Development

**Authentication (Future):**
- Zitadel will provide OAuth2/OpenID Connect
- JWT tokens validated by API
- Authentication required for all `/api/v1/*` endpoints

**User Data Persistence (Future):**
- All queries logged to `query_logs` table
- Health profiles stored in `health_profiles` table
- 90-day retention for conversation histories
- Users can view, export, and delete their data

**Embedding Model:**
- Azure OpenAI `text-embedding-3-large` (3,072 dimensions)
- Model changes require re-embedding all papers

---

## 10. Data Governance Framework for Future Implementation

**Current Compliance (Development):**
- Privacy by design: no user data stored
- Data minimization: only research papers stored
- GDPR requirements not applicable (no PHI/PII persistence)

**Future Compliance Framework:**
- Data Processing Agreements (DPA) with Azure OpenAI and GCP
- GDPR compliance: right to access, erasure, rectification, portability
- Audit logging: all PHI access logged
- Encryption: field-level encryption for health profiles
- Data retention: 1 year for query logs, 90 days for conversations, 7 years for audit logs
- Breach notification procedures (72-hour GDPR requirement)

---

## Conclusion

The Medical Agent is a privacy-focused, AI-powered vaginal health analysis system. Strong privacy is achieved through architectural design: **no user queries, health profiles, or conversation histories are stored during development**. Only medical research papers are persistently stored to enable evidence-based responses. All data processing occurs exclusively within the European Union.

**Current State (Development):**
- Functional REST API with AI-powered pH analysis
- Hybrid search across medical research papers
- In-memory session management (no user data persistence)
- No user authentication or tracking
- Privacy-by-design architecture

**Future State (Post-Testing, Production):**
Any future expansion involving user accounts, query logging, or persistent health profiles will require:
- Comprehensive authentication system (Zitadel integration)
- Database schema activation (`users`, `health_profiles`, `query_logs` tables)
- Field-level encryption for PHI
- Automated daily backups with 30-day retention
- Audit logging for all user actions
- Full GDPR compliance framework
- Data retention and deletion policies
- Security audit and penetration testing

This document establishes the baseline for the current non-persistent, privacy-by-design system and provides a roadmap for future production deployment with full data governance infrastructure.


