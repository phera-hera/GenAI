---

# AI API Integration Guide

Integration reference for the Phera Medical AI service (pH analysis). Use this for BFF orchestration.

The AI backend serves two separate products from two separate deployed instances. Both call the same endpoint with the same request schema. The difference is in the headers you send and which instance you point to.

---

## Project 1 - Beta (Personalization Test App)

### Overview

Internal tool for Phera's researchers to evaluate AI response quality. No authentication. No user accounts. Anonymous query logs saved for debugging only.

### Headers

| Header | Value |
|---|---|
| `Content-Type` | `application/json` |

Do not send `X-API-Key` or `X-User-Id` for this product.

### Base URL

```
https://phera-rag-beta-52458262724.europe-west10.run.app
```

### Endpoint

```
POST /api/v1/query
```

### Request Body

```json
{
  "ph_value": 4.8,
  "age": 28,
  "diagnoses": ["PCOS", "Endometriosis"],
  "ethnic_backgrounds": ["South Asian"],
  "menstrual_cycle": "Irregular",
  "birth_control": {
    "general": null,
    "pill": "Combined pill",
    "iud": null,
    "other_methods": [],
    "permanent": []
  },
  "hormone_therapy": [],
  "hrt": [],
  "fertility_journey": {
    "current_status": null,
    "fertility_treatments": []
  },
  "symptoms": {
    "discharge": ["Creamy"],
    "vulva_vagina": ["Itchy"],
    "smell": [],
    "urine": [],
    "notes": "Some notes"
  }
}
```

**Field reference:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `ph_value` | float | Yes | Range 0.0-14.0 |
| `age` | int | No | Range 0-120 |
| `diagnoses` | string[] | No | e.g. PCOS, Endometriosis, Diabetes |
| `ethnic_backgrounds` | string[] | No | |
| `menstrual_cycle` | string | No | Regular, Irregular, Perimenopausal, Postmenopausal |
| `birth_control` | object | No | `general`, `pill`, `iud`, `other_methods[]`, `permanent[]` |
| `hormone_therapy` | string[] | No | |
| `hrt` | string[] | No | |
| `fertility_journey` | object | No | `current_status`, `fertility_treatments[]` |
| `symptoms` | object | No | `discharge[]`, `vulva_vagina[]`, `smell[]`, `urine[]`, `notes` |

### Response (200 OK)

```json
{
  "session_id": "9d1be1ac-8f12-4f5e-83f8-936af95a8d4f",
  "ph_value": 4.8,
  "agent_reply": "Your vaginal pH of 4.8 is slightly elevated...",
  "disclaimers": "This analysis is for informational purposes only...",
  "citations": [
    {
      "paper_id": "7d4f22ce-3f9d-4f3e-b36a-617aa0246e07",
      "title": "Vaginal Microbiome and pH Dynamics in Reproductive-Age Women",
      "authors": null,
      "doi": null,
      "relevant_section": "Participants with pH > 4.5 showed lower Lactobacillus dominance..."
    }
  ],
  "processing_time_ms": 1842
}
```

> Display `agent_reply` and `citations[]` in the frontend. The medical disclaimer is embedded within `agent_reply`.

### Data Saved

Anonymous `QueryLog` row only - pH value, generated query, response, citations, timing. `user_id = null`. No user identity attached.

---

## Project 2 - MVP (User-Facing Product)

### Overview

User-facing Phera product. Users can be logged in (full persistence) or anonymous (anonymous query log only). The BFF handles all JWT validation - the AI backend never sees or validates a JWT. The BFF extracts the user ID from the token and passes it directly as a header.

### Authentication Flow

```
User logs in -> Zitadel issues JWT
     |
     v
BFF receives request with JWT
BFF validates JWT against Zitadel
BFF extracts user ID (sub claim)
     |
     v
BFF calls AI Backend with:
  X-API-Key: <secret>
  X-User-Id: <zitadel_user_id>   <- only if user is logged in

AI Backend:
  -> Validates X-API-Key
  -> Reads X-User-Id (present = logged in, absent = anonymous)
  -> Runs analysis
  -> Persists data based on presence of X-User-Id
```

### Headers

| Header | Required | Description |
|---|---|---|
| `Content-Type` | Yes | `application/json` |
| `X-API-Key` | Yes | Service-to-service auth. Provided after deployment. |
| `X-User-Id` | No | Zitadel `sub` claim. Send only when user is logged in. |

### Base URL

```
TBD - provided after deployment
```

### Endpoint

```
POST /api/v1/query
```

### Request Body

Same schema as Project 1. See field reference above.

### Response (200 OK)

Same structure as Project 1.

### Data Persistence

| Scenario | What AI saves |
|---|---|
| `X-User-Id` present (logged in) | QueryLog linked to user + HealthProfile saved/updated |
| `X-User-Id` absent (anonymous) | Anonymous QueryLog only (`user_id = null`) |

> All persistence is handled internally by the AI backend. The BFF does not need to manage or track any of this.

---

## Shared Reference

### Error Responses

| Status | Body |
|---|---|
| `401` | `{"error": "UNAUTHORIZED", "message": "Invalid or missing API key"}` |
| `500` | `{"error": "PROCESSING_ERROR", "message": "Failed to process query: <detail>"}` |
| `503` | `{"error": "SERVICE_NOT_CONFIGURED", "message": "Azure OpenAI is not configured..."}` |

### Health Endpoints

```
GET /health        - basic health check
GET /health/ready  - readiness probe
GET /health/live   - liveness probe
```

### Deployment

| Instance | DEPLOYMENT_MODE | Swagger Docs | Base URL |
|---|---|---|---|
| Beta | `beta` | Disabled (production mode) | https://phera-rag-beta-52458262724.europe-west10.run.app |
| MVP | `mvp` | Disabled in production | TBD |

---
