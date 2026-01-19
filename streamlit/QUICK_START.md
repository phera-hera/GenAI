# Quick Start Guide - pHera Medical Agent Streamlit Interface

## For Scientists/Researchers (User Mode)

### 1. Start the Application

```bash
# Terminal 1: Start API
cd Medical_Agent
uvicorn src.medical_agent.api.main:app --reload

# Terminal 2: Start Streamlit
cd Medical_Agent/streamlit
streamlit run Home.py
```

### 2. Access the Interface

- Browser opens automatically at `http://localhost:8501`
- Click **"Continue as User"** (no password needed)

### 3. Test pH Analysis

- Go to **"Test pH Analysis"** in the sidebar
- Enter pH value (required) - try `4.5` for a normal test
- Optionally add symptoms and health information
- Click **"Analyze pH"**
- Review results

### 4. That's It!

You can now test different scenarios without worrying about technical details.

---

## For Administrators (Admin Mode)

### 1. Start the Application

Same as above, but you also need database access configured.

### 2. Access Admin Mode

- Click **"Continue as Admin"**
- Enter password: `admin123` (default)
- You'll see additional "Paper Management" page in sidebar

### 3. Manage Papers

**View Papers:**
- Go to **"Paper Management"** → "View All Papers"
- See all research papers in database

**Delete a Paper:**
- Go to **"Paper Management"** → "Delete Paper"
- Enter Paper ID (from View All Papers)
- Click "Preview Paper" to verify
- Click "DELETE PAPER" and confirm

**Bulk Delete:**
- Go to **"Paper Management"** → "Bulk Operations"
- Select multiple papers
- Confirm deletion

### 4. Change Admin Password

Edit `streamlit/.streamlit/secrets.toml`:

```toml
admin_password = "your_secure_password_here"
```

---

## Common Test Scenarios

### Scenario 1: Normal pH
```
pH Value: 4.2
Age: 28
Menstrual Cycle: Regular
Expected: NORMAL risk level
```

### Scenario 2: Elevated pH with Symptoms
```
pH Value: 5.5
Age: 32
Symptoms: Fishy smell, Grey discharge
Expected: CONCERNING/URGENT risk level
```

### Scenario 3: PCOS Patient
```
pH Value: 5.0
Age: 25
Diagnoses: PCOS
Menstrual Cycle: Irregular
Expected: Personalized insights about PCOS
```

---

## Troubleshooting

**"Could not connect to the API"**
→ Make sure the API server is running (Terminal 1)

**"Please select your role"**
→ Return to Home page and select User or Admin mode

**"Access Denied" on Paper Management**
→ You need to be in Admin mode

---

## Getting Help

- Check `STREAMLIT_README.md` for detailed documentation
- Verify API is running: go to API Health page
- Admin users: check debug information in expandable sections
