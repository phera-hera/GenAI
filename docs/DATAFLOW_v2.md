# Phera Medical Agent - User Data Flow (Version 2)

## 1. What is Phera and Current Stage

Phera is an evidence-based health analysis system that helps users understand their vaginal health by analyzing pH readings against 500+ medical research papers. The system is currently in a **testing/development phase** with a commitment to **privacy-by-design**.

All data processing, storage, and AI analysis happens exclusively within the **European Union** (Germany region, EU data centers) under **full GDPR compliance**.

---

## 2. How Phera Works - End-to-End Data Flow

When a user submits a health question, here's what happens:

1. **User submits form** → All optional health information (age, symptoms, diagnoses, etc.) is sent to our backend in Germany (EU region)
2. **Validation** → Our system checks the information is valid (happens in-memory, temporary)
3. **Query generation** → The information is formatted into a medical research question
4. **Paper retrieval** → Our LangGraph agent searches our database of 500+ medical research papers (stored in Berlin data center)
5. **LLM analysis** → The query + relevant papers are sent to Microsoft Azure OpenAI (running in Germany data center, EU region)
6. **Response generation** → Azure OpenAI generates analysis with medical citations (Germany region)
7. **Results returned** → User receives analysis + citations

**Key point:** All data stays in the European Union. Nothing leaves Europe.

---

## 3. What Data is Collected?

Users can choose what information to share. Some fields are required, most are optional.

### **All Users (Anonymous & Logged-In) Collect:**

| Field | Type | Required? | Examples |
|-------|------|-----------|----------|
| **pH Value** | Number (0-14) | ✅ YES | 4.5, 3.8, 5.2 |
| **Symptoms - Discharge** | Multiple choice | ❌ Optional | No discharge, Creamy, Sticky, Egg white, Clumpy white, Grey and watery, Yellow/Green, Red/Brown |
| **Symptoms - Vulva/Vagina** | Multiple choice | ❌ Optional | Dry, Itchy |
| **Symptoms - Smell** | Multiple choice | ❌ Optional | Strong/fishy, Sour, Chemical-like, Very strong/rotten |
| **Symptoms - Urine** | Multiple choice | ❌ Optional | Frequent urination, Burning sensation |
| **Symptoms - Notes** | Free text | ❌ Optional | User's own description of how they're feeling |
| **Age** | Number | ❌ Optional | 18-120 |
| **Ethnic Background** | Multiple choice | ❌ Optional | African/Black, South Asian, East Asian, White/Caucasian, Mixed, etc. |
| **Medical Diagnoses** | Multiple choice | ❌ Optional | PCOS, Endometriosis, Thyroid disorder, Diabetes, Adenomyosis, etc. |
| **Current Medications** | Multiple choice | ❌ Optional | Various hormone-related medications |
| **Menstrual Cycle** | Single choice | ❌ Optional | Regular, Irregular, No period for 12+ months, Perimenopausal, Postmenopausal |
| **Birth Control Methods** | Multiple choice | ❌ Optional | Pill, IUD, Permanent methods, Other |
| **Hormone Therapy** | Multiple choice | ❌ Optional | Estrogen only, Estrogen + Progestin |
| **HRT (Hormone Replacement)** | Multiple choice | ❌ Optional | Testosterone, Estrogen blocker, Puberty blocker |
| **Fertility Journey** | Multiple choice | ❌ Optional | Current status, Fertility treatments |

### **Logged-In Users Additionally Collect:**

| Field | Type | Where Stored | Details |
|-------|------|--------------|---------|
| **User ID** | Unique identifier | PostgreSQL (Berlin) | Links all their queries together |
| **Email** | Email address | Frontend (EU region) | For account management |
| **Session History** | Previous conversations | PostgreSQL (Berlin) | So they can see past analyses |
| **Session Metadata** | Duration, timestamps | PostgreSQL (Berlin) | When they used the app, for how long |

---

## 4. Where is Everything Stored?

All data stays in the **European Union (Germany region, Berlin data center)**. Nothing leaves Europe.

| Data Type | Storage Location | How Long | Encrypted? | GDPR Compliant? |
|-----------|------------------|----------|------------|-----------------|
| **Medical research papers** (content, metadata, searchable index) | GCP Cloud Storage (Germany, EU Berlin data center) | Permanent | ✅ Yes | ✅ Yes |
| **User queries & responses** | PostgreSQL (Germany, EU Berlin data center) | ~1 year (yet to be decided) | ✅ Yes | ✅ Yes |
| **User health profiles** | PostgreSQL (Germany, EU Berlin data center) | ~1 year (yet to be decided) | ✅ Yes | ✅ Yes |
| **Conversation history** | PostgreSQL (Germany, EU Berlin data center) | ~1 year (yet to be decided) | ✅ Yes | ✅ Yes |
| **User email** | Frontend storage (EU region) | [Pending frontend team answer] | [Pending] | ✅ Yes |
| **LLM processing** (temporary) | Azure OpenAI (Germany, EU region) | Minutes only | ✅ Yes | ✅ Yes |

### **Important Note:**
- **GCP = Google Cloud Platform** - Our cloud infrastructure provider (servers in Berlin, Germany)
- **PostgreSQL = Database** - Stores user data, query logs, conversation history (in Berlin)
- **Azure OpenAI = Microsoft's AI** - Processes your health data to generate analysis (runs in Germany, EU region)

---

## 5. Who Else Can See Your Data?

Only specific companies process your data, and they're all bound by GDPR:

| Company | What They Receive | Purpose | Location | Data Protection |
|---------|-------------------|---------|----------|------------------|
| **Microsoft Azure OpenAI** | pH value + health info (NO email, NO name) | Generate analysis | Germany, EU region | ✅ GDPR compliant |
| **LangChain (LangSmith)** | LLM prompts + responses (NO personal details) | Improve AI accuracy | EU region | ✅ GDPR compliant |
| **Zitadel** | Email + login credentials only | Authentication/login | EU region | ✅ GDPR compliant |

**Important:** Your NAME and EMAIL are NEVER sent to AI or analytics services. Only your health information (pH, symptoms, diagnoses) is used for analysis.

### **Data Selling Policy**
We may share **anonymized health data** with research partners to improve women's health knowledge. This is ONLY done with your explicit consent. You can opt-out anytime.

---

## 6. Data Classifications - What Type of Data Is It?

Different types of data require different levels of protection:

| Data | Classification | Why Protected? | Examples |
|------|-----------------|------------------|----------|
| **pH value, symptoms, diagnoses** | **PHI** (Protected Health Information) | Medical/private | High sensitivity |
| **User email, name, ID** | **PII** (Personally Identifiable Information) | Can identify you | High sensitivity |
| **Medical research papers** | **Public Data** | Published research | Low sensitivity |
| **LLM analysis** | **Health Guidance** | Health-related output | Medium sensitivity |

---

## 7. How Long Do We Keep Your Data?

### **Current Retention Timeline: Approximately 1 Year (Yet to be decided with GDPR team)**

| Data Type | Retention Period | What Happens at End |
|-----------|------------------|---------------------|
| Query logs (your questions + our answers) | ~1 year (yet to be decided) | You're notified at 11 months |
| Health profile (symptoms, diagnoses, etc.) | ~1 year (yet to be decided) | You're notified at 11 months |
| Conversation history | ~1 year (yet to be decided) | You're notified at 11 months |
| Email (frontend) | [Pending frontend team] | [Pending frontend team] |
| Audit logs (future) | 7 years (for compliance) | Kept for legal traceability |

### **What Happens After 1 Year?**

We email you with 3 options:

1. ✅ **"Keep my data"** → We store for another year (get a new email in 11 months)
2. ✅ **"Delete everything"** → Permanent deletion within 7 days
3. ⏱️ **No response** → Auto-deleted after 30 days

**You can delete your account anytime** — all data gone immediately, no questions asked.

---

## 8. For Logged-In Users: What Gets Tracked?

When you log in, we track:

✅ **What we DO track:**
- Your user ID (so we know it's you)
- Date/time of each query
- Your health information you enter
- Your previous conversations
- Session duration (how long you used the app)

❌ **What we DON'T track:**
- Your location (IP address)
- Your device type
- Your browsing behavior outside Phera
- Your name/email with health data
- Any cookies or analytics

---

## 9. Data Protection & Privacy Guarantees

| Guarantee | What It Means |
|-----------|---------------|
| ✅ **Everything in EU** | Data never leaves Germany, EU region |
| ✅ **Encrypted in transit** | Data encrypted when traveling between systems |
| ✅ **Encrypted at rest** | Data encrypted when stored |
| ✅ **No tracking** | No cookies, no fingerprinting, no third-party analytics |
| ✅ **Right to delete** | Request deletion anytime, get confirmation |
| ✅ **Right to access** | Request to see what data we have about you |
| ✅ **GDPR compliant** | All services bound by EU data protection laws |

---

## 10. Current State (Development) vs Future State (Production)

### **What We Have NOW:**
- ✅ All data in EU (Berlin data center)
- ✅ Basic encryption
- ✅ No tracking/cookies
- ✅ Health data NOT shared with name/email
- ⏳ No persistent audit logs yet
- ⏳ No field-level encryption yet

### **What We'll Add LATER (Production):**
- ✅ Full field-level encryption
- ✅ 7-year audit trails (for compliance traceability)
- ✅ Formal account deletion process
- ✅ Data portability (export your data anytime)
- ✅ Full GDPR article 17 (right to erasure) implementation
- ✅ Breach notification protocol

---

## 11. Important - Still Being Finalized

❓ **Pending Frontend Team:**
- Where is user email stored exactly? (GCP bucket, database, or elsewhere?)
- How long is session metadata retained?
- Are all frontend systems also in EU region?

❓ **Pending GDPR Team:**
- Exact data retention timeline (we're proposing ~1 year, but this needs legal review)
- Specific Zitadel server location in EU

❓ **Still Being Decided:**
- Exact audit logging requirements
- Data anonymization procedures
- Third-party audit schedule

---

## 12. In Case of a Security Issue

If there's ever a data breach or security incident:

1. 🔔 **You're notified within 72 hours** (required by GDPR)
2. 📋 **We explain exactly what happened** (what data, how many users)
3. 🛡️ **We explain what we're doing to fix it** (steps taken, prevention going forward)
4. 📞 **You can contact our privacy team** for questions

---

## Summary

**Phera is designed with privacy-first in mind:**
- Users decide what to share (all optional except pH)
- Everything stays in EU (Germany, Berlin data center)
- No tracking, no cookies, no third-party analytics
- GDPR compliant from day one
- Users control their data (delete anytime, see anytime)

**When you use Phera, your health information is protected like HIPAA-regulated healthcare data — because your privacy matters.**

---

**Document Version:** 2.0
**Last Updated:** March 2026
**Compliance Status:** GDPR, EU Data Protection Regulations
**Data Centers:** Germany region (EU), Berlin
