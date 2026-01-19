# Streamlit Application Architecture

## Overview

The pHera Medical Agent Streamlit interface is a **role-based multi-page application** designed for two distinct user types:

1. **Users (Scientists/Researchers)**: Non-technical users who need to test and validate the medical agent
2. **Admins (System Administrators)**: Technical users who manage papers and system operations

## Application Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser Opens                         │
│                     http://localhost:8501                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                        Home.py                               │
│                   (Main Entry Point)                         │
│                                                              │
│  ┌────────────────┐              ┌────────────────┐        │
│  │   User Mode    │              │   Admin Mode   │        │
│  │                │              │                │        │
│  │ No password    │              │ Password:      │        │
│  │ required       │              │ admin123       │        │
│  └────────┬───────┘              └───────┬────────┘        │
│           │                              │                  │
└───────────┼──────────────────────────────┼──────────────────┘
            │                              │
            │  Session State:              │
            │  user_role = "user"          │  user_role = "admin"
            │  api_base_url = "..."        │  api_base_url = "..."
            │                              │
            ▼                              ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│     User Navigation       │    │    Admin Navigation       │
│                          │    │                          │
│  📊 Test pH Analysis     │    │  📊 Test pH Analysis     │
│  🏥 API Health           │    │  📚 Paper Management     │
│                          │    │  🏥 API Health           │
└───────────────────────────┘    └───────────────────────────┘
```

## Page Structure

### 1. Home.py (Entry Point)

**Purpose**: Role selection and session initialization

**Features**:
- Role selection (User/Admin)
- Admin password verification
- Session state initialization
- Role switching capability
- API URL configuration (sidebar)

**Session State**:
```python
st.session_state.user_role       # "user" | "admin" | None
st.session_state.api_base_url    # API endpoint URL
```

### 2. pages/1_Test_pH_Analysis.py

**Access**: Both User and Admin

**Purpose**: Test the medical agent with various health scenarios

**Features**:
- Full pH analysis form
- Health profile inputs (age, diagnoses, symptoms, etc.)
- API integration for pH analysis
- Results display with risk levels
- Citations and research references

**User vs Admin Differences**:
- **User Mode**: Clean results only, no debug info
- **Admin Mode**: Includes request/response debug sections

### 3. pages/2_Paper_Management.py

**Access**: Admin Only

**Purpose**: Manage research papers in the database

**Features**:
- **View All Papers**: List all papers with details
- **Delete Paper**: Delete by Paper ID with preview
- **Bulk Operations**: Multi-select deletion

**Technical Requirements**:
- Database connection
- `medical_agent.core.paper_manager` module
- GCP Storage access (optional)

**Safety Features**:
- Confirmation dialogs
- Preview before deletion
- Progress tracking for bulk operations
- Error handling and rollback

### 4. pages/3_API_Health.py

**Access**: Both User and Admin

**Purpose**: Monitor API status and configuration

**Features**:
- Health check endpoint testing
- API information retrieval
- Troubleshooting guides

**Admin Extras**:
- Custom endpoint testing
- OpenAPI schema access
- Advanced diagnostics

## Role-Based Access Control

### Implementation

```python
# Check on each page
if "user_role" not in st.session_state or st.session_state.user_role is None:
    st.warning("Please select your role from the Home page first.")
    st.stop()

# Admin-only pages
if st.session_state.user_role != "admin":
    st.error("Access Denied: This page is only available to administrators.")
    st.stop()
```

### Security Model

- **User Mode**: No authentication, assumes trusted network
- **Admin Mode**: Password-based (stored in `secrets.toml`)
- **Session-based**: Role persists across pages during session
- **No server-side auth**: Streamlit session state only (suitable for internal tools)

**Production Recommendations**:
- Implement proper authentication (OAuth, SAML, etc.)
- Add audit logging for admin operations
- Use environment variables for secrets
- Implement RBAC with multiple admin levels

## Data Flow

### pH Analysis Request Flow

```
User Input (Form)
    ↓
Build Request Payload
    ↓
POST /api/v1/query
    ↓
Medical Agent Processing
    ↓
Response with Analysis
    ↓
Display Results (Risk Level, Insights, Citations)
```

### Paper Deletion Flow

```
Admin Selects Paper
    ↓
Preview Paper Details
    ↓
Confirmation Dialog
    ↓
PaperManager.delete_paper()
    ↓
    ├─→ Delete from Database (chunks + paper record)
    └─→ Delete from GCP Storage (optional)
    ↓
Success/Error Feedback
```

## Technical Architecture

### Technology Stack

- **Frontend**: Streamlit (Python-based web framework)
- **API Client**: Python `requests` library
- **Database**: PostgreSQL (via SQLAlchemy through PaperManager)
- **Cloud Storage**: GCP Storage (for PDFs)
- **Async**: `asyncio` for database operations

### Key Dependencies

```
streamlit>=1.30.0
requests>=2.31.0
```

Plus the main `medical_agent` package for admin features.

### Configuration Management

- **API URL**: Session state (configurable per user)
- **Admin Password**: `.streamlit/secrets.toml` (gitignored)
- **Environment**: Inherits from parent app's `.env`

## Page Navigation

Streamlit automatically creates sidebar navigation from:

1. `Home.py` → "Home" (always visible)
2. `pages/1_*.py` → First navigation item
3. `pages/2_*.py` → Second navigation item
4. `pages/3_*.py` → Third navigation item

**Naming Convention**:
- `1_Test_pH_Analysis.py` → "Test pH Analysis"
- `2_Paper_Management.py` → "Paper Management"
- `3_API_Health.py` → "API Health"

Numbers control order, underscores become spaces.

## Error Handling

### User-Friendly Errors

**User Mode**:
- Generic error messages
- No technical details exposed
- Clear next steps provided

**Admin Mode**:
- Detailed error messages
- Stack traces in expandable sections
- Debug payload inspection

### Common Error Scenarios

1. **API Connection Error**: Clear message to check if API is running
2. **Authentication Error**: Redirect to home for role selection
3. **Database Error**: Admin-only detailed error, user sees generic message
4. **Timeout**: Retry suggestion with context

## Performance Considerations

- **Caching**: Streamlit's `@st.cache_data` for repeated API calls (not implemented yet)
- **Async Operations**: Database queries use `asyncio` to prevent blocking
- **Progress Indicators**: Loading spinners and progress bars for long operations
- **Pagination**: Limit paper list to 100 items (can be extended)

## Security Considerations

### Current Implementation (Development)

✅ Password protection for admin
✅ Gitignored secrets file
✅ Session-based access control
✅ Confirmation dialogs for destructive operations

### Production Recommendations

⚠️ Implement proper authentication system
⚠️ Add SSL/TLS for all connections
⚠️ Implement rate limiting
⚠️ Add audit logging for all admin operations
⚠️ Use encrypted secrets management (not plain text)
⚠️ Implement IP whitelisting for admin access
⚠️ Add CSRF protection
⚠️ Implement proper user management system

## Future Enhancements

### Short Term
- Session history tracking
- Export results to PDF
- Better caching strategy
- Pagination for large paper lists

### Medium Term
- Multi-user authentication system
- Audit logs viewer
- Paper upload interface
- Batch testing scenarios

### Long Term
- Real-time monitoring dashboard
- A/B testing framework
- Analytics and usage metrics
- Integration with CI/CD pipeline

## Deployment Options

### Local Development
```bash
streamlit run Home.py
```

### Production Deployment

**Option 1: Streamlit Cloud**
- Push to GitHub
- Connect to Streamlit Cloud
- Configure secrets via web interface

**Option 2: Docker**
- Include Streamlit in Docker container
- Expose port 8501
- Use environment variables for config

**Option 3: Custom Server**
- Run with production WSGI server
- Reverse proxy (Nginx/Apache)
- Process manager (systemd/supervisor)

## Monitoring and Logging

### Logging Strategy

- **Streamlit Logs**: `~/.streamlit/`
- **Application Logs**: Console output
- **Admin Actions**: Should implement audit log (future)

### Monitoring Points

- API response times
- Error rates by page
- User vs Admin usage patterns
- Paper management operations

## Maintenance

### Regular Tasks

- Update admin password periodically
- Review and cleanup old sessions
- Monitor disk usage for cached data
- Update dependencies

### Backup Considerations

- Secrets file (`secrets.toml`)
- Session state (if persisted)
- Configuration files

---

**Last Updated**: 2026-01-19
**Version**: 1.0.0
**Author**: pHera Development Team
