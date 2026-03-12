# User Data Flow - Phera Medical Agent


## The Two Ways People Use Phera

### **Path 1: Anonymous User (Not Logged In)**

Someone visits the app but doesn't create an account.

**What we collect:**
- Their search/question about vaginal health(pH value, symptoms, medical history)
- The analysis we give them back

**Where it goes:**
- Stored in Berlin, Germany (in our database)
- Stored securely with NO name, email, or identity

**Who sees it:**
- Only our technical team (to improve the app)
- Not linked to them personally

**How long we keep it:**
- 6-12 months, then automatically deleted
- No email needed (we don't know who they are)

---

### **Path 2: Registered User (Logged In)**

Someone creates an account using our login system.

**Step 1: Sign Up (Frontend)**
- User provides: email, creates password
- Frontend stores: email, session information
- Location: EU region (Berlin)

**Step 2: User Enters Health Information**
- User fills out form: pH value, symptoms, medical history
- Frontend sends: unique user ID + form data to backend
- Location: Still EU region

**Step 3: Backend Processes the Request**
- Our system receives: user ID + health information (NO email or name)
- System checks: "Have we seen this user before?"
  - If YES → Continue with existing profile
  - If NO → Create new profile
- Everything stored in Berlin, Germany

**Step 4: AI Analysis Happens**
- User's health information → formatted into a medical question
- Question sent to AI system (in EU region)
- AI researches our database of 500+ medical papers
- AI finds relevant information and creates an analysis
- Analysis returned with medical citations

**Step 5: Results Stored**
- Analysis + citations stored in Berlin database
- Linked to user ID (so they can see history)
- Location: Still Berlin, Germany

**Step 6: Follow-up Questions**
- User can ask follow-up questions in same session
- System remembers: previous conversation + health profile
- Everything stored in Berlin database

---

## Where Your Data Actually Goes

### **Our Servers (Berlin, Germany)**
✅ User ID + health information (pH, symptoms, diagnoses, medications)
✅ All queries and responses
✅ Conversation history
✅ NO email, NO name, NO personal details sent here

### **AI Processing (EU Region)**
✅ Only health information is sent to AI
✅ NOT sent: email, name, identity
✅ AI processes it in EU region only
✅ AI results come back to us

### **Medical Paper Database (Berlin)**
✅ Our collection of 500+ medical research papers
✅ Stored in Berlin, Germany
✅ AI searches this for relevant information

---

## Data We DON'T Collect or Share

❌ We never store your name in our system
❌ We never store your email with health data
❌ We never share data with third parties (except AI provider)
❌ We never use your data for marketing
❌ We never sell data

---

## How Long We Keep Your Data

### **For Registered Users:**

| Data Type | How Long | What Happens |
|-----------|----------|-------------|
| Health profile | While account is active | Deleted when you close account |
| Queries & responses | 1 year | At 11 months: we email you |
| Conversation history | 1 year | At 11 months: we email you |

### **At the 1-Year Mark (If Still Stored):**

We email you with 3 options:

1. **"Keep my data"** → We store it for another year (repeat process)
2. **"Delete everything"** → We permanently delete all your data within 7 days
3. **No response** → We automatically delete after 30 days

### **If You Close Your Account Anytime:**
- All your data deleted immediately
- No questions asked

---

## Who Processes Your Data (Third Parties)

### **1. Login System**
- Company: Zitadel
- What we share: Email + login credentials only
- Location: EU region
- Purpose: So you can log in securely

### **2. AI Provider**
- Company: Microsoft Azure OpenAI
- What we share: ONLY health information (no name, no email)
- Location: EU region
- Purpose: Analyze your health information

### **3. AI Evaluation System**
- What we share: AI conversations (no personal details)
- Location: EU region
- Purpose: Make the AI better over time

**Important:** Nobody can identify you from this data. Your name and email are never sent to any third party.

---

## Security & Privacy Guarantees

✅ **Everything in EU region** - Data never leaves Europe
✅ **Encrypted in transit** - Data encrypted when traveling between systems
✅ **Encrypted at rest** - Data encrypted when stored
✅ **Anonymous** - Health data has NO name/email attached
✅ **Right to delete** - You can request deletion anytime
✅ **Right to access** - You can request to see what data we have about you

## What is a "Query Log"?
This is simply a record of:

What the user asked (their health question)
What answer we gave (our analysis)
When it happened (date and time)

Think of it like a note that says: "On March 10, 2026 at 2:30 PM, someone asked about pH 4.5 with mild symptoms. We gave them analysis with 3 medical citations."
That's it — a query log is just a record of the conversation, nothing more.