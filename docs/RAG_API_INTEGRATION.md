# RAG API Integration Guide

Integration spec for the FemTech Medical RAG service (pH analysis). Use this for BFF orchestration.

---

## 1. Variable — Will Change (Update After Deployment)

These values depend on environment and deployment. They will change when we move from local to GCP.

| Item | Local (current) | Post-deployment |
|------|-----------------|-----------------|
| Base URL | `http://localhost:8000` | TBD (e.g. `https://your-service.run.app`) |
| OpenAPI docs | `http://localhost:8000/docs` | TBD (may be restricted) |
| OpenAPI JSON | `http://localhost:8000/openapi.json` | TBD |

**Action:** Use local values for now. We'll share the production base URL and doc URLs once deployment is done (target: next week).

---

## 2. Constant — Unchanged Regardless of Deployment

These remain the same whether we run locally or on GCP.

### Base path

- `/api/v1`

### Main endpoint

- Method: `POST`
- Path: `/api/v1/query`
- Content-Type: `application/json`

### Request schema

**Required:**

| Field | Type | Constraints |
|-------|------|-------------|
| `ph_value` | float | 0.0–14.0 |

**Optional (all fields can be omitted):**

| Field | Type | Constraints / Notes |
|-------|------|----------------------|
| `age` | int | 0–120 |
| `diagnoses` | string[] | e.g. PCOS, Endometriosis, Diabetes |
| `ethnic_backgrounds` | string[] | |
| `menstrual_cycle` | string | Regular, Irregular, Perimenopausal, Postmenopausal, etc. |
| `birth_control` | object | `general`, `pill`, `iud`, `other_methods`, `permanent` |
| `hormone_therapy` | string[] | |
| `hrt` | string[] | |
| `fertility_journey` | object | `current_status`, `fertility_treatments` |
| `symptoms` | object | `discharge`, `vulva_vagina`, `smell`, `urine`, `notes` |

**Note:** Do not use `user_message` or `session_id` for now — follow-up flows are not supported.

### Response schema (200 OK)

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `ph_value` | float | Analyzed pH value |
| `agent_reply` | string | Full analysis text |
| `disclaimers` | string | Medical disclaimer text |
| `citations` | array | See below |
| `processing_time_ms` | int | Processing time in milliseconds |

**Citation object:**

| Field | Type | Notes |
|-------|------|-------|
| `paper_id` | string | Internal ID |
| `title` | string \| null | Paper title |
| `authors` | string \| null | Often null |
| `doi` | string \| null | Often null |
| `relevant_section` | string \| null | Relevant excerpt |

### Error responses

| Status | Body |
|--------|------|
| 500 | `{"error": "PROCESSING_ERROR", "message": "..."}` |
| 503 | `{"error": "SERVICE_NOT_CONFIGURED", "message": "Azure OpenAI is not configured..."}` |

### Health endpoints (constant paths)

- `GET /health` — basic health
- `GET /health/ready` — readiness probe
- `GET /health/live` — liveness probe

---

## 3. Examples

### Minimal request

```json
{ "ph_value": 4.2 }
```

### Full request

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
  "fertility_journey": null,
  "symptoms": {
    "discharge": ["Creamy"],
    "vulva_vagina": ["Itchy"],
    "smell": [],
    "urine": [],
    "notes": "Some notes"
  }
}
```

### Success response (200)

```json
{
  "session_id": "abc-123",
  "ph_value": 4.8,
  "agent_reply": "Your pH reading of 4.8 is slightly elevated...",
  "disclaimers": "This analysis is for informational purposes only...",
  "citations": [
    {
      "paper_id": "node_xyz",
      "title": "Vaginal pH and Microbiome",
      "authors": null,
      "doi": null,
      "relevant_section": "Excerpt..."
    }
  ],
  "processing_time_ms": 1500
}
```

---

## 4. Summary

| Category | Contents |
|----------|----------|
| Variable | Base URL, docs URL — will update after deployment |
| Constant | Endpoint path, request/response schemas, error format, health paths |
