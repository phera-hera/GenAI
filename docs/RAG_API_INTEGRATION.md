# RAG API Integration Guide

Integration spec for the FemTech Medical RAG service (pH analysis). Use this for BFF orchestration.

---

## 1. Two Products, One Backend

The RAG backend serves two separate products. The core pH analysis logic is identical for both — the only difference is what data gets saved afterward.

**Beta** — Internal product used by our team and scientists for testing and evaluation. There is no authentication. No user data is stored. Anonymous query logs are saved for debugging purposes only.

**MVP** — User-facing product. Has authentication via Zitadel. Users can use the app with or without logging in. Data is only saved for users who are logged in.

Both products call the same API endpoint (`POST /api/v1/query`) with the same request body. The backend decides what to save based on which product is calling and whether the user is authenticated.

---

## 2. Data Persistence Rules

| Product | Auth status | What gets saved |
|---------|-------------|-----------------|
| Beta | No auth (always) | Anonymous query logs only (pH value, query, response, citations, timing). No user identity attached. |
| MVP | Not logged in | Nothing saved. |
| MVP | Logged in | Everything saved — query log (linked to user), user record, health profile. |

**Health profile auto-load:** For logged-in MVP users, if they submit health data (age, diagnoses, symptoms, etc.) in a request, it gets saved to their profile. On their next request, if they don't submit health data, the backend automatically loads their previously saved profile and uses it for the analysis. This means the BFF can omit health fields on repeat queries for logged-in users.

---

## 3. Authentication Flow

Authentication only applies to the MVP product. Beta has no authentication at all.

```
MVP flow (logged-in user):

  Mobile App → user logs in via Zitadel → gets JWT token
       │
       ▼
  BFF receives request with JWT token
       │
       ▼
  BFF forwards request to RAG API
  with header: Authorization: Bearer <token>
       │
       ▼
  RAG API validates the token against Zitadel
       │
       ├── Token valid → identify user, run analysis, save data
       └── Token missing/invalid → run analysis, save nothing
```

**How user identification works on the backend:**
1. RAG API validates the JWT and extracts the `sub` claim (Zitadel's unique user ID)
2. Looks up the user in our database by this external ID
3. If the user exists → use their record
4. If the user does not exist → create a new user record (first login)
5. All saved data (query logs, health profile) is linked to this internal user record

**What the BFF needs to do:**
- For MVP: forward the user's Zitadel JWT as `Authorization: Bearer <token>` header when calling the RAG API. If the user is not logged in, simply don't send the header.
- For Beta: don't send any Authorization header.

---

## 4. Deployment

The same Docker image is deployed twice on GCP Cloud Run — one instance for each product. The only difference is configuration.

| Instance | `DEPLOYMENT_MODE` env var | Auth required |
|----------|--------------------------|---------------|
| Beta service | `beta` | No |
| MVP service | `mvp` | Optional (logged-in users get persistence) |

Each instance will have its own base URL. The BFF should point to the correct URL depending on which product it is serving.

### URLs — Will Change After Deployment

| Item | Local (current) | Post-deployment |
|------|-----------------|-----------------|
| Base URL | `http://localhost:8000` | TBD (e.g. `https://phera-beta.run.app`, `https://phera-mvp.run.app`) |
| OpenAPI docs | `http://localhost:8000/docs` | TBD (may be restricted in production) |
| OpenAPI JSON | `http://localhost:8000/openapi.json` | TBD |

**Action:** Use local values for now. Production URLs will be shared once deployment is done (target: next week).

---

## 5. API Reference

These remain the same regardless of deployment or product.

### Base path

- `/api/v1`

### Main endpoint

- Method: `POST`
- Path: `/api/v1/query`
- Content-Type: `application/json`

### Request headers

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | `application/json` |
| `Authorization` | No (MVP only) | `Bearer <zitadel_jwt_token>`. If present and valid, user data is persisted. If absent, API still works but nothing is saved. Not used for Beta. |

### Request body

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

**Note:** `user_message` and `session_id` fields exist in the schema but are not currently used. Do not send them.

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

### Health endpoints

- `GET /health` — basic health
- `GET /health/ready` — readiness probe
- `GET /health/live` — liveness probe

---

## 6. Examples

The request body is identical for both products. The only difference is whether the `Authorization` header is included.

### Request without auth (Beta, or MVP unauthenticated user)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Request with auth (MVP logged-in user)

Same request body, with the `Authorization` header added:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIs..." \
  -d '{
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
  }'
```

### Success response (200)

The response is the same regardless of auth status:

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

## 7. Summary

| Category | Contents |
|----------|----------|
| Products | Beta (internal, no auth, anonymous logs only) and MVP (user-facing, auth-gated persistence) |
| Variable | Base URLs — will update after deployment (two separate URLs, one per product) |
| Constant | Endpoint path, request/response schemas, error format, health endpoints |
| Auth | MVP only. BFF forwards Zitadel JWT as `Authorization: Bearer <token>`. Backend validates and decides persistence. |
| Health profile | Saved on submit for logged-in MVP users. Auto-loaded on next request if health fields are omitted. |

---

## 8. Questions for Johannes — Zitadel Integration

Hi Johannes! We're adding authentication-based data persistence to the RAG backend. When a user is logged in (MVP product), we want to save their query history and health profile. When they're not logged in or on the beta product, we don't save anything (or save only anonymous debug logs).

To wire this up on the backend, we need a few details about how Zitadel is set up:

### Q1. How should we validate tokens?

When the BFF sends the user's Zitadel token to the RAG API, we need to verify it's real and not expired. There are two common ways:

- **Option A — Local validation:** We download Zitadel's public keys (from a JWKS endpoint) and verify the token ourselves. Fast, no network call per request.
- **Option B — Introspection:** We call a Zitadel endpoint on every request and ask "is this token valid?" Slower, but always real-time.

Which approach are you using or expecting us to use?

### Q2. What is the Zitadel issuer URL?

The domain / URL where Zitadel is hosted. Something like `https://our-org.zitadel.cloud` or a self-hosted URL. We need this to verify tokens came from the right place.

### Q3. What is the audience (`aud`) value?

When Zitadel issues a token, it includes an `aud` (audience) field that says which application the token is meant for. What value should we expect here for our backend API?

### Q4. What user information is inside the token?

Specifically:
- Is `sub` (subject) the unique user identifier? Or does Zitadel use a different field?
- Is the user's `email` included in the token?
- Are there any custom claims (like user roles)?

### Q5. Have you already written any token validation code?

If you've already built a middleware, helper function, or validation logic (even in another service or language), could you share it? We want to stay consistent rather than build something separate.

### Q6. How does the BFF currently handle the token?

When a logged-in user makes a request from the mobile app:
- Does the BFF receive the Zitadel JWT in the request?
- Will the BFF forward that token to the RAG API as an `Authorization: Bearer <token>` header?
- Or should the RAG API validate tokens differently?
