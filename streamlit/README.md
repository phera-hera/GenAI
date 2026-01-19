# pHera Medical Agent - Streamlit Interface

Role-based testing interface with two modes:
- **User Mode**: For scientists testing the medical agent (simple, no technical details)
- **Admin Mode**: For administrators managing papers and system (requires password)

## Quick Start

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

Opens automatically at `http://localhost:8501`

## User Modes

### User Mode (Scientists/Researchers)
- **Access**: No password needed
- **Features**: Test pH Analysis only
- **Interface**: Simple, clean, no technical details

### Admin Mode (System Administrators)
- **Access**: Password required (default: `admin123`)
- **Features**: Test pH Analysis + Paper Management + API Health
- **Interface**: Includes debug info and advanced tools

## Configuration

**Change Admin Password**: Edit `.streamlit/secrets.toml`
```toml
admin_password = "your_password_here"
```

**Change API URL**: In sidebar after selecting role (default: `http://localhost:8000`)

## Test Scenarios

**Normal pH:**
```
pH: 4.2, Age: 28, Regular cycle
Expected: NORMAL risk
```

**Elevated pH:**
```
pH: 5.5, Symptoms: Fishy smell, Grey discharge
Expected: CONCERNING/URGENT risk
```

## Troubleshooting

**"Could not connect to API"**
→ Start the API server first (see Quick Start step 2)

**"Access Denied" on Paper Management**
→ Switch to Admin mode (requires password)

## Documentation

See `USER_GUIDE.md` for detailed information on:
- Complete feature list
- Admin operations (paper management)
- Security considerations
- Production deployment

---

**Version**: 1.0.0 | **Contact**: Administrator for issues
