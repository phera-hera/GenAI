# Streamlit Test Interface for pHera Medical Agent

A simple web-based testing interface for the Medical Agent API.

## Prerequisites

1. **API Server Running**: Make sure the Medical Agent API is running first
2. **Python Environment**: Python 3.10+ with pip or uv

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

## Running the Test Interface

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
# Make sure you're in the project root
cd Medical_Agent

# Run the Streamlit app
streamlit run app.py
```

The Streamlit interface will automatically open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Configure API URL

- At the top of the page, enter your API base URL (default: `http://localhost:8000`)
- This is where the Streamlit app will send requests

### 2. Fill in the Form

**Required:**
- **pH Value**: Enter a pH value between 0-14 (typical vaginal pH: 3.8-5.0)

**Optional (but recommended for better results):**
- **Basic Info**: Age, ethnic background
- **Diagnoses**: Select any hormone-related diagnoses
- **Hormone Status**: Menstrual cycle status, birth control, hormone therapy
- **Fertility Journey**: Pregnancy status, fertility treatments
- **Symptoms**: Discharge, vulva/vagina, smell, urine symptoms
- **Notes**: Free text notes (NOT sent to the agent)

### 3. Submit Analysis

- Click "Analyze pH" button
- Wait for the analysis (typically 5-15 seconds)
- Review the results including:
  - Risk level (NORMAL, MONITOR, CONCERNING, URGENT) with color-coding
  - Detailed analysis
  - Personalized insights
  - Recommended next steps
  - Research citations

### 4. API Health Check

- Use the "API Health Check" tab to verify if the API is running
- Verify API configuration and dependencies
- View API information and status

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
streamlit run app.py --server.port 8502
```

## Features

### Implemented
- Full form based on user_input.pdf requirements
- API health checking
- Request/response debugging
- Color-coded risk levels (Green, Yellow, Orange, Red)
- Citation display with expandable details
- Medical disclaimers
- Responsive layout
- Professional, clean interface
- Integrated quick tips section

### Future Enhancements
- Session history tracking
- Export results to PDF
- Comparison of multiple readings
- Graph visualization of pH trends
- Batch testing capabilities

## Development Notes

- The app is built with Streamlit for simplicity
- All form fields mirror the mobile app design from `documentation/user_input.pdf`
- Request payload can be inspected via expandable sections
- Notes field is intentionally NOT sent to the medical agent (as per API design)

## Support

For issues or questions:
1. Check API server logs for backend errors
2. Review Streamlit console for frontend errors
3. Use the debugging expanders to inspect request/response payloads
