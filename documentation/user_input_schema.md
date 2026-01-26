# pHera Medical Questionnaire - User Input Schema

## Overview
This document outlines the user input fields and data structure for the pHera medical questionnaire API. The pHera app is a health assessment tool designed to collect comprehensive health information from users, particularly focused on women's health and reproductive health.

## Required Fields

### pH (Vaginal pH)
- **Field Type**: Decimal Number
- **Required**: Yes ✓
- **Description**: Vaginal pH measurement
- **Range**: 0.0 - 14.0 (typical range 3.8 - 4.5)
- **Decimal Places**: 1
- **Example**: 4.5

---

## Optional Fields

### 1. Demographics

#### Age
- **Field Type**: Numeric Input
- **Required**: No
- **Description**: User's age in years
- **Format**: Integer
- **Example**: 28

### 2. Medical History - Diagnoses Related to Hormones

**Field Type**: Multi-select checkboxes
**Required**: No
**Description**: User can select one or more hormone-related diagnoses
**Options**:
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

### 3. Demographics - Ethnic Background

**Field Type**: Multi-select
**Required**: No
**Description**: User's racial and ethnic background(s)
**Options**:
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
- Mixed / Multiple ancestrie
- Other

### 4. Hormone Status

#### Menstrual Cycle
**Field Type**: Single select
**Required**: No
**Options**:
- Regular
- Irregular
- No period for 12+ months
- Never had a period
- Perimenopausal
- Postmenopausal

### 5. Birth Control

**Field Type**: Multiple selection categories
**Required**: No
**Description**: Current or recent birth control usage

#### Birth Control Status
**Options**:
- No birth control or hormonal birth control
- Stopped birth control in the last 3 months
- Morning after-pill / emergency contraception in the last 7 days

#### Pill
**Options**:
- Combined pill
- Progestin-only pill

#### IUD (Intrauterine Device)
**Options**:
- Hormonal IUD
- Copper IUD

#### Other Hormonal Methods
**Options**:
- Contraceptive implant
- Contraceptive injection
- Vaginal ring
- Patch

#### Permanent Methods
**Options**:
- Tubal ligation

### 6. Hormone Therapy / HRT

**Field Type**: Single select
**Required**: No
**Options**:
- Estrogen only
- Estrogen + progestin

#### Hormone Replacement Therapy (HRT) Options
**Field Type**: Multi-select
**Required**: No
**Options**:
- Testosterone
- Estrogen blocker
- Puberty blocker

### 7. Fertility Journey

#### Current Status
**Field Type**: Single select
**Required**: No
**Options**:
- I am pregnant
- I had a baby (last 12 months)
- I am not able to get pregnant
- I am trying to conceive

#### Fertility Treatments (Last 3 Months)
**Field Type**: Multi-select
**Required**: No
**Options**:
- Ovulation induction
- Intrauterine insemination (IUI)
- In vitro fertilisation (IVF)
- Egg freezing stimulation
- Luteal progesterone

### 8. Symptoms

#### Discharge
**Field Type**: Multi-select
**Required**: No
**Options**:
- No discharge
- Creamy
- Sticky
- Egg white
- Clumpy white
- Grey and watery
- Yellow / Green
- Red / Brown

#### Vulva & Vagina
**Field Type**: Multi-select
**Required**: No
**Options**:
- Dry
- Itchy

#### Smell
**Field Type**: Multi-select
**Required**: No
**Options**:
- Strong and unpleasant ("fishy")
- Sour
- Chemical-like
- Very strong or rotten

#### Urine
**Field Type**: Multi-select
**Required**: No
**Options**:
- Frequent urination
- Burning sensation

#### Notes
**Field Type**: Text Area
**Required**: No
**Description**: User can add additional notes, extra symptoms, or how they've been feeling

---

## Data Structure Example (JSON)

```json
{
  "ph": 4.5,
  "age": 28,
  "diagnoses": ["Endometriosis", "PCOS"],
  "ethnicBackgrounds": ["South Asian", "East Asian"],
  "menstrualCycle": "Irregular",
  "birthControl": {
    "status": "Stopped birth control in the last 3 months",
    "pill": "Combined pill",
    "iud": null,
    "otherHormonal": [],
    "permanent": []
  },
  "hormoneTherapy": {
    "type": "Estrogen + progestin",
    "hrtOptions": []
  },
  "fertilityJourney": {
    "currentStatus": "I am trying to conceive",
    "treatments": ["Ovulation induction"]
  },
  "symptoms": {
    "discharge": ["Creamy", "Sticky"],
    "vulvaVagina": ["Dry"],
    "smell": ["Sour"],
    "urine": ["Frequent urination"],
    "notes": "Experiencing discomfort during menstrual cycle"
  }
}
```

## Minimal Request Example

```json
{
  "ph": 4.2
}
```
