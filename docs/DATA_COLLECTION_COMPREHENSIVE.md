# Phera Medical Agent - Complete Data Collection Reference

## All User Input Fields (Anonymous & Logged-In Users)

### **REQUIRED FIELD:**

#### **1. pH Value**
- **Type:** Number
- **Range:** 0 to 14
- **Required:** ✅ YES
- **Purpose:** Vaginal pH reading from test strip
- **Examples:** 3.8, 4.2, 4.5, 5.2, etc.

---

## OPTIONAL FIELDS:

### **2. Symptoms Information**

Users can report symptoms across 5 categories:

#### **2.1 Discharge Type** (Multiple choice - user can select multiple)
- No discharge
- Creamy
- Sticky
- Egg white
- Clumpy white
- Grey and watery
- Yellow/Green
- Red/Brown

#### **2.2 Vulva/Vagina Symptoms** (Multiple choice - user can select multiple)
- Dry
- Itchy

#### **2.3 Smell** (Multiple choice - user can select multiple)
- Strong and unpleasant ("fishy")
- Sour
- Chemical-like
- Very strong or rotten

#### **2.4 Urine-Related Symptoms** (Multiple choice - user can select multiple)
- Frequent urination
- Burning sensation

#### **2.5 Additional Symptom Notes** (Free text)
- User can write anything
- Examples: "Feeling stressed lately", "Pain during intercourse", etc.

---

### **3. Age**
- **Type:** Number
- **Range:** 0 to 120 years
- **Required:** ❌ Optional
- **Purpose:** Age of user
- **Examples:** 18, 25, 35, 45, 55, etc.

---

### **4. Ethnic Background** (Multiple choice - user can select multiple)

- African / Black
- North African
- Arab
- Middle Eastern
- East Asian
- South Asian
- Southeast Asian
- Central Asian / Caucasus
- Latin American / Latina / Latinx / Hispanic
- Sinti / Roma
- White / Caucasian / European
- Mixed / Multiple ancestries
- Other

---

### **5. Medical Diagnoses** (Multiple choice - user can select multiple)

- Adenomyosis
- Amenorrhea
- Cushing's syndrome
- Diabetes
- Endometriosis
- Intersex status
- Thyroid disorder
- Uterine fibroids
- Polycystic ovary syndrome (PCOS)
- Premature ovarian insufficiency (POI)

---

### **6. Current Medications** (Multiple choice - user can select multiple)

This field captures hormone-related medications. Common examples include:
- Metformin (for PCOS)
- Levothyroxine (for thyroid)
- Spironolactone (anti-androgen)
- [Various other hormone-related medications]

---

### **7. Menstrual Cycle Status** (Single choice - user picks ONE)

- Regular
- Irregular
- No period for 12+ months
- Never had a period
- Perimenopausal
- Postmenopausal

---

### **8. Birth Control Methods** (Multiple choice - user can select multiple across categories)

#### **8.1 General Birth Control Status** (Single choice)
- No control
- Stopped in last 3 months
- Emergency contraception in last 7 days

#### **8.2 Pill Type** (Single choice)
- Combined pill
- Progestin-only pill

#### **8.3 IUD Type** (Single choice)
- Hormonal IUD
- Copper IUD

#### **8.4 Other Methods** (Multiple choice - can select multiple)
- Implant
- Injection
- Vaginal ring
- Patch

#### **8.5 Permanent Methods** (Multiple choice - can select multiple)
- Tubal ligation

---

### **9. Hormone Therapy** (Multiple choice - user can select multiple)

- Estrogen only
- Estrogen + Progestin

---

### **10. HRT (Hormone Replacement Therapy)** (Multiple choice - user can select multiple)

- Testosterone
- Estrogen blocker
- Puberty blocker

---

### **11. Fertility Journey**

#### **11.1 Current Status** (Single choice)
- Pregnant
- Had baby (last 12 months)
- Not able to get pregnant
- Trying to conceive

#### **11.2 Fertility Treatments** (Multiple choice - can select multiple)
- Ovulation induction
- IUI (Intrauterine Insemination)
- IVF (In Vitro Fertilization)
- Egg freezing
- Luteal progesterone

---

### **12. Follow-Up Message** (For logged-in users with conversation history)
- **Type:** Free text
- **Purpose:** Follow-up questions on previous analysis
- **Example:** "What does this mean for my fertility?"

---

### **13. Session ID** (System-generated, not user input)
- **Type:** Unique identifier
- **Purpose:** Tracks conversation continuity
- **For:** Logged-in users asking follow-up questions

---

## LOGGED-IN USERS ADDITIONALLY STORE:

### **14. User ID**
- **Type:** Unique identifier
- **Stored In:** PostgreSQL database
- **Purpose:** Links all queries to user account

### **15. Email Address**
- **Type:** Email
- **Stored In:** Frontend storage (EU region)
- **Purpose:** Account management, login

### **16. Session Metadata**
- **Type:** Timestamps, duration
- **Stored In:** PostgreSQL database
- **Purpose:** Usage analytics, conversation history

---

## Summary Table

| Field | Type | Required | Values/Options | Stored Where | Logged-In Only? |
|-------|------|----------|-----------------|--------------|-----------------|
| **pH Value** | Number (0-14) | ✅ YES | Any number 0-14 | PostgreSQL | ❌ Both |
| **Discharge** | Multiple choice | ❌ Optional | 8 options | PostgreSQL | ❌ Both |
| **Vulva/Vagina** | Multiple choice | ❌ Optional | 2 options | PostgreSQL | ❌ Both |
| **Smell** | Multiple choice | ❌ Optional | 4 options | PostgreSQL | ❌ Both |
| **Urine** | Multiple choice | ❌ Optional | 2 options | PostgreSQL | ❌ Both |
| **Symptom Notes** | Free text | ❌ Optional | Any text | PostgreSQL | ❌ Both |
| **Age** | Number (0-120) | ❌ Optional | Any age | PostgreSQL | ❌ Both |
| **Ethnic Background** | Multiple choice | ❌ Optional | 13 options | PostgreSQL | ❌ Both |
| **Diagnoses** | Multiple choice | ❌ Optional | 10 options | PostgreSQL | ❌ Both |
| **Medications** | Multiple choice | ❌ Optional | Various | PostgreSQL | ❌ Both |
| **Menstrual Cycle** | Single choice | ❌ Optional | 6 options | PostgreSQL | ❌ Both |
| **Birth Control** | Multiple choice | ❌ Optional | 12 combinations | PostgreSQL | ❌ Both |
| **Hormone Therapy** | Multiple choice | ❌ Optional | 2 options | PostgreSQL | ❌ Both |
| **HRT** | Multiple choice | ❌ Optional | 3 options | PostgreSQL | ❌ Both |
| **Fertility Journey** | Multiple choice | ❌ Optional | 9 combinations | PostgreSQL | ❌ Both |
| **Follow-Up Message** | Free text | ❌ Optional | Any text | PostgreSQL | ✅ Logged-In |
| **User ID** | ID | N/A | System-generated | PostgreSQL | ✅ Logged-In |
| **Email** | Email | N/A | User's email | Frontend | ✅ Logged-In |
| **Session Metadata** | Timestamps | N/A | Dates/times | PostgreSQL | ✅ Logged-In |

---

## Key Points

- **Total Form Fields:** 15 user-fillable + 4 system fields = 19 total
- **Required:** Only pH value (1 field)
- **Optional:** Everything else (14 fields)
- **Maximum Data Points:**
  - Anonymous user: 1 required + 14 optional = up to 15 data points
  - Logged-in user: Same as above + 4 system fields = up to 19 data points
- **All Data Types:** Health information (PHI) - no personal identifying information except email
- **Storage:** All in Berlin, Germany (EU region)
- **Retention:** ~1 year (yet to be decided)

---

**Document Version:** 1.0
**Created:** March 2026
