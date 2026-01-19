# pHera Medical Agent - Streamlit Interface

> A professional, role-based testing interface for the pHera Medical Agent API

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd Medical_Agent/streamlit
uv pip install -r streamlit_requirements.txt

# 2. Start API (Terminal 1)
cd Medical_Agent
uvicorn src.medical_agent.api.main:app --reload

# 3. Start Streamlit (Terminal 2)
cd Medical_Agent/streamlit
streamlit run Home.py
```

Browser opens automatically at `http://localhost:8501`

## 👥 Two User Modes

### 🧪 User Mode - For Scientists & Researchers
- **No password needed**
- Simple interface to test the medical agent
- Submit pH values and health scenarios
- View analysis results and risk levels
- No technical details exposed

### 🔧 Admin Mode - For System Administrators
- **Password:** `admin123` (change in `.streamlit/secrets.toml`)
- All User features PLUS:
- Manage research papers (view, delete, bulk operations)
- Advanced API diagnostics
- Debug information and logs

## 📁 What's Inside

```
Home.py                    → Main entry point (role selection)
pages/
  ├── 1_Test_pH_Analysis.py   → pH testing interface
  ├── 2_Paper_Management.py   → Admin paper management
  └── 3_API_Health.py         → API monitoring
```

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get running in 5 minutes
- **[STREAMLIT_README.md](STREAMLIT_README.md)** - Complete documentation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture
- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - What was created and why

## 🧪 Test Scenarios

**Normal pH (User Mode):**
```
pH: 4.2, Age: 28, Regular cycle
Expected: NORMAL risk level
```

**Elevated pH (User Mode):**
```
pH: 5.5, Symptoms: Fishy smell
Expected: CONCERNING risk level
```

**Delete Paper (Admin Mode):**
```
1. Go to Paper Management
2. View all papers
3. Select paper to delete
4. Confirm deletion
```

## ⚙️ Configuration

**API URL:** Configurable in sidebar (default: `http://localhost:8000`)

**Admin Password:** Edit `.streamlit/secrets.toml`:
```toml
admin_password = "your_password_here"
```

## 🔒 Security

**Current (Development):**
- Password-protected admin mode
- Session-based access control
- Suitable for trusted networks

**Production Recommendations:**
- Implement proper authentication (OAuth, SAML)
- Use encrypted secrets management
- Add audit logging
- Enable SSL/TLS

## 🆘 Troubleshooting

**"Could not connect to the API"**
→ Start the API server first (see Quick Start step 2)

**"Please select your role"**
→ Go to Home page and select User or Admin mode

**"Access Denied" on Paper Management**
→ You need to be in Admin mode (requires password)

## 📞 Need Help?

1. Check [QUICK_START.md](QUICK_START.md) for common scenarios
2. Review [STREAMLIT_README.md](STREAMLIT_README.md) for details
3. Verify API is running: `curl http://localhost:8000/health`

---

**Version:** 1.0.0  
**Last Updated:** 2026-01-19  
**Built for:** pHera Medical Agent Testing
