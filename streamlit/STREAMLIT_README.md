# Streamlit Interface for pHera Medical Agent

A role-based web interface for the Medical Agent API with two modes:
- **User Mode**: For scientists/researchers testing the application (non-technical)
- **Admin Mode**: For administrators managing research papers and system operations

## Prerequisites

1. **API Server Running**: Make sure the Medical Agent API is running first
2. **Python Environment**: Python 3.10+ with pip or uv
3. **Database Access**: Required for Admin mode paper management

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install Streamlit dependencies
uv pip install -r streamlit_requirements.txt
```

### Option 2: Using pip

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r streamlit_requirements.txt
```

## Running the Interface

### Step 1: Start the API Server

In one terminal:

```bash
# Make sure you're in the project root
cd Medical_Agent

# Start the API server (adjust command based on your setup)
uvicorn src.medical_agent.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Start the Streamlit App

In another terminal:

```bash
# Navigate to the streamlit directory
cd Medical_Agent/streamlit

# Run the Streamlit app
streamlit run Home.py
```

The Streamlit interface will automatically open in your browser at `http://localhost:8501`

## User Roles

### User Mode (For Scientists/Researchers)

**Purpose:** Test the medical agent without technical complexity

**Access:** No password required, just select "User Mode" on the home page

**Features:**
- Test pH Analysis with various scenarios
- View analysis results and risk levels
- Monitor API health status
- Simple, clean interface with no technical details

**Perfect for:**
- Scientists validating medical accuracy
- Researchers testing different scenarios
- Non-technical stakeholders reviewing functionality

### Admin Mode (For System Administrators)

**Purpose:** Manage papers and system operations

**Access:** Requires admin password (default: `admin123`)

**Features:**
- All User Mode features
- Paper Management (view, delete papers)
- Bulk operations on research papers
- Advanced API diagnostics
- Debug information and logs

**Perfect for:**
- System administrators
- DevOps engineers
- Technical staff managing the database

## Usage Guide

### 1. Select Your Role

On the home page, choose:
- **User Mode**: For testing and validation (no password needed)
- **Admin Mode**: For paper management (requires password)

### 2. Configure API URL

- In the sidebar, you'll find the API Base URL setting (default: `http://localhost:8000`)
- This is where the Streamlit app will send requests
- You can change this if your API is running on a different port or server

### 3. Navigate Using Sidebar

The sidebar shows available pages based on your role:

**For User Mode:**
- Test pH Analysis
- API Health

**For Admin Mode:**
- Test pH Analysis
- Paper Management (Admin Only)
- API Health

### 4. Fill in the Form (pH Analysis Page)

**Required:**
- **pH Value**: Enter a pH value between 0-14 (typical vaginal pH: 3.8-5.0)

**Optional (but recommended for better results):**
- **Basic Info**: Age, ethnic background
- **Diagnoses**: Select any hormone-related diagnoses
- **Hormone Status**: Menstrual cycle status, birth control, hormone therapy
- **Fertility Journey**: Pregnancy status, fertility treatments
- **Symptoms**: Discharge, vulva/vagina, smell, urine symptoms
- **Notes**: Free text notes (NOT sent to the agent)

### 5. Submit Analysis

- Click "Analyze pH" button
- Wait for the analysis (typically 5-15 seconds)
- Review the results including:
  - Risk level (NORMAL, MONITOR, CONCERNING, URGENT) with color-coding
  - Detailed analysis
  - Personalized insights
  - Recommended next steps
  - Research citations

### 6. Paper Management (Admin Only)

**View All Papers:**
- See all research papers in the database
- View paper details (title, authors, DOI, chunks)
- Quick delete option for individual papers

**Delete Specific Paper:**
- Enter paper ID (UUID) to delete a specific paper
- Preview paper before deletion
- Option to delete PDF from GCP Storage
- Confirmation required before deletion

**Bulk Operations:**
- Select multiple papers for deletion
- Progress tracking during bulk deletion
- Detailed results for each operation

### 7. API Health Check

- Check if the API is running and accessible
- View API information and configuration
- Admin users get additional diagnostic tools
- Troubleshooting tips included

## Testing Scenarios

Here are some example scenarios to test:

### Scenario 1: Normal pH
- pH Value: 4.2
- Age: 28
- Menstrual Cycle: Regular
- Symptoms: No discharge
- Expected Risk: NORMAL

### Scenario 2: Elevated pH with Symptoms
- pH Value: 5.5
- Age: 32
- Symptoms: Fishy smell, Grey and watery discharge
- Expected Risk: CONCERNING or URGENT

### Scenario 3: Pregnancy
- pH Value: 4.5
- Age: 30
- Fertility Status: I am pregnant
- Expected: Pregnancy-specific guidance

### Scenario 4: PCOS with Irregular Cycle
- pH Value: 5.0
- Age: 25
- Diagnoses: PCOS
- Menstrual Cycle: Irregular
- Expected: PCOS-specific insights

## Troubleshooting

### "Connection Error: Could not connect to the API"

**Solution:**
- Verify the API server is running
- Check the API base URL in the sidebar
- Ensure no firewall is blocking localhost connections

### "API returned status code: 503"

**Solution:**
- Check that Azure OpenAI is configured (see main README.md)
- Verify `.env` file has all required credentials

### "Timeout Error"

**Solution:**
- The query might be complex and taking longer than 60 seconds
- Check API server logs for errors
- Try a simpler query with fewer symptoms

### Port Already in Use

**Solution:**
```bash
# Use a different port for Streamlit
streamlit run Home.py --server.port 8502
```

### Paper Management Not Working

**Solution:**
- Ensure you're in Admin Mode (not User Mode)
- Verify database connection is configured
- Check that `medical_agent` package is properly installed
- Ensure you're running from the correct directory

### "Access Denied" on Paper Management Page

**Solution:**
- You need to be in Admin Mode to access this page
- Return to Home and enter admin password
- Verify the password in `.streamlit/secrets.toml`

## Configuration

### Admin Password

The admin password is stored in `streamlit/.streamlit/secrets.toml`

**To change the admin password:**

1. Edit `streamlit/.streamlit/secrets.toml`
2. Change the `admin_password` value
3. Restart the Streamlit app

**Default password:** `admin123` (⚠️ Change this in production!)

**Important:** The `secrets.toml` file is gitignored and should never be committed to version control.

## Features

### User Mode Features
- ✓ Full pH analysis form based on user_input.pdf requirements
- ✓ Simple, non-technical interface
- ✓ Color-coded risk levels (Green, Yellow, Orange, Red)
- ✓ Citation display with expandable details
- ✓ Medical disclaimers
- ✓ API health monitoring
- ✓ Clean, professional interface
- ✓ No debug information (clean results only)

### Admin Mode Features
- ✓ All User Mode features
- ✓ Paper Management interface
  - View all papers in database
  - Delete individual papers by ID
  - Bulk deletion operations
  - GCP Storage integration
- ✓ Advanced API diagnostics
- ✓ Debug information and request/response payloads
- ✓ Custom endpoint testing
- ✓ System monitoring tools

### Future Enhancements
- Session history tracking
- Export results to PDF
- Comparison of multiple readings
- Graph visualization of pH trends
- Batch testing capabilities
- User authentication system
- Audit logs for admin operations

## Project Structure

```
streamlit/
├── Home.py                          # Main entry point with role selection
├── pages/
│   ├── 1_Test_pH_Analysis.py       # pH testing (User + Admin)
│   ├── 2_Paper_Management.py        # Paper CRUD (Admin only)
│   └── 3_API_Health.py              # API monitoring (User + Admin)
├── .streamlit/
│   └── secrets.toml                 # Admin password (gitignored)
├── .gitignore                       # Git ignore file
├── streamlit_requirements.txt       # Python dependencies
└── STREAMLIT_README.md              # This file
```

### Multi-Page App Structure

Streamlit automatically creates navigation from:
- `Home.py` → Main home page
- `pages/` directory → Sidebar navigation pages

Page naming convention:
- Prefix with number for ordering: `1_`, `2_`, `3_`
- Use underscores, they're converted to spaces in the UI
- Example: `1_Test_pH_Analysis.py` → "Test pH Analysis" in sidebar

## Development Notes

- Built with Streamlit's multi-page app framework
- Role-based access control using session state
- All form fields mirror the mobile app design from `documentation/user_input.pdf`
- Admin users see debug information, regular users see clean results only
- Paper management integrates with `medical_agent.core.paper_manager`
- Notes field is intentionally NOT sent to the medical agent (as per API design)
- Session state persists role and API URL across pages

## Support

For issues or questions:
1. Check API server logs for backend errors
2. Review Streamlit console for frontend errors
3. Use the debugging expanders to inspect request/response payloads
