# pHera Medical Agent - User Guide

## Overview

Multi-page Streamlit application with role-based access for testing the pHera Medical Agent.

## File Structure

```
streamlit/
├── Home.py                          # Entry point with role selection
├── pages/
│   ├── 1_Test_pH_Analysis.py       # pH testing (User + Admin)
│   ├── 2_Paper_Management.py        # Paper CRUD (Admin only)
│   └── 3_API_Health.py              # API monitoring (Admin only)
└── .streamlit/secrets.toml          # Admin password (gitignored)
```

## Features by Role

### User Mode Features
- ✓ Test pH analysis with health profiles
- ✓ View risk levels and insights
- ✓ Access research citations
- ✓ Clean interface, no technical details
- ✗ No paper management
- ✗ No API health monitoring

### Admin Mode Features
- ✓ All User Mode features
- ✓ Paper Management:
  - View all papers in database
  - Delete papers by ID
  - Bulk delete operations
  - GCP Storage integration
- ✓ API Health monitoring and diagnostics
- ✓ Debug information and request/response payloads

## Using the Application

### For Scientists (User Mode)

1. **Start Application**
   ```bash
   streamlit run Home.py
   ```

2. **Select Role**: Click "Continue as User" (no password)

3. **Test pH Analysis**:
   - Enter pH value (required)
   - Add optional health information (age, symptoms, diagnoses, etc.)
   - Click "Analyze pH"
   - Review results

4. **Interpret Results**:
   - **NORMAL** (Green): pH within healthy range
   - **MONITOR** (Yellow): Minor deviation, track over time
   - **CONCERNING** (Orange): Notable deviation, consider consulting doctor
   - **URGENT** (Red): Significant concern, prompt medical consultation

### For Administrators (Admin Mode)

1. **Select Role**: Click "Continue as Admin", enter password (`admin123`)

2. **Paper Management**:
   
   **View All Papers**:
   - Lists all papers with details (title, authors, DOI, chunks)
   - Quick delete option for each paper
   
   **Delete Specific Paper**:
   - Enter Paper ID (UUID)
   - Preview paper details
   - Confirm deletion
   - Optional: Also delete PDF from GCP Storage
   
   **Bulk Operations**:
   - Select multiple papers
   - Confirm bulk deletion
   - Progress tracking

3. **API Health**:
   - Check if API is running
   - View API information
   - Test custom endpoints
   - Advanced diagnostics

## Form Fields (pH Analysis)

### Required
- **pH Value**: 0-14 (typical vaginal pH: 3.8-5.0)

### Optional
- **Age**: User's age
- **Ethnic Background**: One or more backgrounds
- **Diagnoses**: Hormone-related conditions (PCOS, Endometriosis, etc.)
- **Menstrual Cycle**: Regular, Irregular, Perimenopause, etc.
- **Birth Control**: Type of contraception
- **Hormone Therapy**: HRT types
- **Fertility Journey**: Pregnancy status, treatments
- **Symptoms**: Discharge, vulva/vagina, smell, urine symptoms
- **Notes**: Free text (NOT sent to agent, for app context only)

## Security

### Current Implementation
- Password-based admin access
- Session state for role management
- Suitable for internal/trusted networks

### Production Recommendations
- Implement OAuth or SAML authentication
- Individual user accounts with RBAC
- Audit logging for admin operations
- SSL/TLS encryption
- IP whitelisting
- Encrypted secrets management

## Configuration

### Admin Password
Edit `.streamlit/secrets.toml`:
```toml
admin_password = "your_secure_password"
```
⚠️ This file is gitignored and should never be committed.

### API Base URL
Configure in sidebar after selecting role. Default: `http://localhost:8000`

## Troubleshooting

### Connection Errors
**Issue**: "Could not connect to the API"
**Solution**: 
- Verify API server is running: `curl http://localhost:8000/health`
- Check API URL in sidebar
- Ensure no firewall blocking localhost

### Authentication Errors
**Issue**: "Please select your role"
**Solution**: Return to Home page and select User or Admin mode

**Issue**: "Invalid admin password"
**Solution**: Check password in `.streamlit/secrets.toml`

### Paper Management Errors
**Issue**: "Paper Management module not available"
**Solution**: Ensure `medical_agent` package is installed and database is accessible

### Performance Issues
**Issue**: Slow response times
**Solution**:
- Check API server logs for bottlenecks
- Verify database connection is healthy
- Consider increasing timeout values

## Technical Details

### Architecture
- **Frontend**: Streamlit (Python)
- **API**: FastAPI backend at `http://localhost:8000`
- **Database**: PostgreSQL via `medical_agent.core.paper_manager`
- **Storage**: GCP Storage for PDFs
- **Async**: asyncio for database operations

### Session Management
```python
st.session_state.user_role        # "user" | "admin" | None
st.session_state.api_base_url     # API endpoint
```

### Navigation
Streamlit automatically creates sidebar navigation from:
- `Home.py` → "Home"
- `pages/1_*.py` → First item
- `pages/2_*.py` → Second item (hidden for users)
- `pages/3_*.py` → Third item (hidden for users)

## Deployment

### Local Development
```bash
streamlit run Home.py
```

### Production Options

**Streamlit Cloud**:
- Push to GitHub
- Connect repository to Streamlit Cloud
- Configure secrets via web interface

**Docker**:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r streamlit_requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "Home.py"]
```

**Custom Server**:
- Use reverse proxy (Nginx/Apache)
- Process manager (systemd/supervisor)
- Configure SSL/TLS

## Maintenance

### Regular Tasks
- Update admin password periodically
- Monitor application logs
- Update dependencies
- Backup `.streamlit/secrets.toml`

### Monitoring
- API response times
- Error rates by page
- User vs Admin usage patterns
- Paper management operations

## Future Enhancements

**Short Term**:
- Session history tracking
- Export results to PDF
- Result comparison over time

**Long Term**:
- Multi-user authentication system
- Audit logs viewer
- Paper upload interface
- Analytics dashboard

---

**Version**: 1.0.0  
**Last Updated**: 2026-01-19  
**Support**: Contact your administrator for assistance
