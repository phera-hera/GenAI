# Deployment Guide — Project 1 (Beta / RAG Test App)

## Overview

**Project:** Phera Medical RAG Backend (pHera)
**What:** FastAPI + LangGraph RAG service for vaginal health analysis
**Where:** GCP Cloud Run, region `europe-west10` (Berlin) — EU data residency, same region as Cloud SQL
**GCP Project:** `phera-medical-agent`
**Deployment Method:** gcloud CLI directly (no Terraform — infrastructure will be migrated to company GCP later)
**Mode:** `DEPLOYMENT_MODE=beta` (internal test/evaluation app, researchers only)
**Status:** Project 1 only — Project 2 (MVP with CV service) deferred

---

## What This App Does (Context)

The Beta app is an **internal research evaluation tool** (not public-facing).

**Flow:**
1. Researcher submits: pH value + optional health profile (age, diagnoses, symptoms, etc.)
2. FastAPI routes to LangGraph workflow:
   - **Retrieve Node:** Hybrid search (pgvector + BM25) over 500+ medical papers → over-retrieve 15 chunks → cross-encoder reranks to top 5
   - **Generate Node:** GPT-4o produces personalized response with inline citations `[1] [2] [3]`
3. Response returned: `agent_reply` (includes medical disclaimer) + `citations[]`
4. Anonymous QueryLog saved to PostgreSQL + pgvector (user_id = null, for debugging only)

**No authentication needed.** No user accounts. Purely ephemeral analysis.

---

## Infrastructure Already In Place

- **PostgreSQL 16 + pgvector** on GCP Cloud SQL (provisioned, papers ingested)
- **GCP Cloud Storage bucket** `phera_researchpaper` (contains 500+ medical papers, indexed)
- **.env and config.py** ready (will receive `DEPLOYMENT_MODE=beta` at runtime)
- **Dockerfile** ready (Python 3.11-slim, includes Docling + reranker deps, 2Gi recommended)

---

## Prerequisites (Before You Run)

You must have/do these things before deployment:

1. **gcloud CLI** installed and authenticated
   ```bash
   gcloud auth login
   gcloud config set project phera-medical-agent
   ```

2. **Docker** installed and running locally

3. **GCP APIs enabled** (will be auto-enabled in Step 1 if not)
   - Artifact Registry
   - Cloud Run
   - Cloud SQL (already enabled)
   - Secret Manager (optional, but good for later)

4. **GCP credentials** — your account must have:
   - Editor role (or specific: Artifact Registry Editor, Cloud Run Admin, Service Account User)

5. **Azure OpenAI credentials** (you provide):
   - API key (LLM/chat)
   - Endpoint URL (LLM/chat)
   - Deployment names: `gpt-4o`, `gpt-4o-mini`, `text-embedding-3-large`
   - **Note:** Same Azure instance is used for both LLM and embeddings. Separate credentials not needed.

6. **Cloud SQL connection name** (format: `PROJECT:REGION:INSTANCE`)
   - Example: `phera-medical-agent:europe-west10:phera-pg-main`
   - Get it: `gcloud sql instances describe INSTANCE_NAME --format="value(connectionName)"`
   - **Note:** Cloud SQL is already deployed in `europe-west10` (Berlin)

7. **Cloud SQL database password** for user `femtech`

---

## Deployment Steps

### Step 1 — Enable Artifact Registry (First Time Only)

```bash
gcloud services enable artifactregistry.googleapis.com

gcloud artifacts repositories create phera-images \
  --repository-format=docker \
  --location=europe-west10 \
  --description="Phera backend Docker images"
```

This creates a Docker registry in Berlin where you'll push your image.

---

### Step 2 — Authenticate Docker with Artifact Registry

```bash
gcloud auth configure-docker europe-west10-docker.pkg.dev
```

This lets your local Docker daemon push to GCP's Artifact Registry in Berlin.

---

### Step 3 — Build Docker Image Locally

From repo root:

```bash
docker build -t europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:latest .
```

This builds the image based on `Dockerfile`. Will take ~3-5 minutes (first time) due to Python deps + Docling libraries.

---

### Step 4 — Push Image to Artifact Registry

```bash
docker push europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:latest
```

Uploads ~800MB to GCP. Takes ~2-3 minutes depending on network.

---

### Step 5 — Deploy to Cloud Run

**This is the main deployment command.** Replace all `YOUR_*` placeholders with actual values:

```bash
gcloud run deploy phera-rag-beta \
  --image europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:latest \
  --region europe-west10 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 5 \
  --timeout 3600 \
  --add-cloudsql-instances YOUR_CLOUD_SQL_CONNECTION_NAME \
  --set-env-vars "ENVIRONMENT=production,\
DEPLOYMENT_MODE=beta,\
GCP_PROJECT_ID=phera-medical-agent,\
GCP_BUCKET_NAME=phera_researchpaper,\
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY,\
AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT,\
AZURE_OPENAI_API_VERSION=2024-02-15-preview,\
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o,\
AZURE_OPENAI_MINI_DEPLOYMENT_NAME=gpt-4o-mini,\
DATABASE_URL=postgresql+asyncpg://femtech:YOUR_DB_PASSWORD@/femtech_medical?host=/cloudsql/YOUR_CLOUD_SQL_CONNECTION_NAME,\
POSTGRES_PASSWORD=YOUR_DB_PASSWORD"
```

**What each flag means:**
- `--image`: Docker image URL (from Step 4)
- `--region europe-west10`: Deploy to Berlin (same region as Cloud SQL, minimal latency)
- `--allow-unauthenticated`: Public endpoint (no auth required for beta)
- `--memory 2Gi --cpu 2`: Needed for cross-encoder reranker model (~200MB) + concurrent requests
- `--min-instances 1`: Always keep 1 instance warm (no cold starts, consistent response times)
- `--max-instances 5`: Don't scale beyond 5 instances
- `--timeout 3600`: Allow 1 hour for long-running queries (retrieval + generation can take 3-8 seconds per request)
- `--add-cloudsql-instances`: Cloud SQL socket connection (no public IP exposed)
- `--set-env-vars`: All configuration variables

---

### Step 6 — Verify Deployment

Get the Cloud Run service URL:

```bash
gcloud run services describe phera-rag-beta \
  --region europe-west10 \
  --format "value(status.url)"
```

Output will look like: `https://phera-rag-beta-xxxxx-nw.a.run.app`

Test the health endpoint:

```bash
curl https://phera-rag-beta-xxxxx-nw.a.run.app/health
```

Expected response:

```json
{"status": "healthy"}
```

If it returns 200 with `healthy`, deployment succeeded. If 503 or timeout, check logs:

```bash
gcloud run logs read phera-rag-beta --region europe-west10 --limit 50
```

---

### Step 7 — Test the Query Endpoint

Once health check passes, test the actual RAG endpoint:

```bash
curl -X POST https://phera-rag-beta-xxxxx-nw.a.run.app/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "ph_value": 4.8,
    "age": 32,
    "diagnoses": ["BV"],
    "symptoms": {
      "discharge": ["clumpy white"],
      "vulva_vagina": [],
      "smell": ["fishy"],
      "urine": [],
      "notes": ""
    }
  }'
```

Expected response (200 OK):

```json
{
  "session_id": "uuid-xxx",
  "ph_value": 4.8,
  "agent_reply": "Your pH reading of 4.8 is elevated...",
  "disclaimers": "This analysis is for informational purposes only...",
  "citations": [
    {
      "paper_id": "uuid",
      "title": "Bacterial Vaginosis...",
      "authors": null,
      "doi": null,
      "relevant_section": "..."
    }
  ],
  "processing_time_ms": 3847
}
```

---

## Environment Variables Reference

| Variable | Required | Value | Notes |
|----------|----------|-------|-------|
| `ENVIRONMENT` | Yes | `production` | Not `development` (disables Swagger, enables CORS) |
| `DEPLOYMENT_MODE` | Yes | `beta` | Project 1 only (Project 2 = `mvp`) |
| `GCP_PROJECT_ID` | Yes | `phera-medical-agent` | For logging and bucket access |
| `GCP_BUCKET_NAME` | Yes | `phera_researchpaper` | Storage bucket with medical papers |
| `AZURE_OPENAI_API_KEY` | Yes | Your Azure API key | Used for both LLM and embeddings (same instance) |
| `AZURE_OPENAI_ENDPOINT` | Yes | `https://xxx.openai.azure.com/` | Your Azure endpoint |
| `AZURE_OPENAI_API_VERSION` | No | `2024-02-15-preview` | Default is fine |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | No | `gpt-4o` | Deployment name in your Azure account |
| `AZURE_OPENAI_MINI_DEPLOYMENT_NAME` | No | `gpt-4o-mini` | Used for query generation |
| `DATABASE_URL` | Yes | See below | Cloud SQL socket connection |
| `POSTGRES_PASSWORD` | Yes | Your DB password | Password for user `femtech` |

**DATABASE_URL format:**
```
postgresql+asyncpg://femtech:PASSWORD@/femtech_medical?host=/cloudsql/PROJECT:REGION:INSTANCE
```

Example:
```
postgresql+asyncpg://femtech:mypassword123@/femtech_medical?host=/cloudsql/phera-medical-agent:europe-west3:femtech-db-production
```

---

## Post-Deployment

1. **Save the Cloud Run URL** — this is your public endpoint for researchers
2. **Share the URL** with your research team (internal only, no public advertising)
3. **Monitor logs** for errors: `gcloud run logs read phera-rag-beta --region europe-west3 --follow`
4. **Check cost** — Cloud Run charges per request + compute time. Expected: $0.40-2/day for typical research usage

---

## Rollback (If Needed)

If deployment has issues, revert to previous version:

```bash
gcloud run deploy phera-rag-beta \
  --image europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:PREVIOUS_TAG \
  --region europe-west10
```

Or delete and redeploy:

```bash
gcloud run delete phera-rag-beta --region europe-west10 --quiet
# Then re-run Step 5
```

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| 503 Service Unavailable | `gcloud run logs read phera-rag-beta --region europe-west10` — look for Azure OpenAI errors or Cloud SQL connection issues |
| 502 Bad Gateway | Cloud SQL socket connection failing — verify `--add-cloudsql-instances` value and DB password |
| Timeout on /api/v1/query | Normal if retrieval + generation takes >30s. Cloud Run timeout is 3600s, so 8s is fine. Check logs for Azure OpenAI latency |
| 401 Unauthorized | Not applicable for beta (no auth). If implementing later, check `X-API-Key` header |
| High latency (>10s) | LLM inference time or vector search. Check Azure OpenAI instance tier and pgvector query plans |

---

## Notes

- **Same region (Berlin):** Cloud Run and Cloud SQL both in `europe-west10` — minimal latency (~0-5ms), no cross-region data transfer cost
- **Cloud SQL socket connection** (`/cloudsql/CONNECTION_NAME`) is secure and doesn't expose public IPs
- **`--allow-unauthenticated`** is fine for internal beta. No API key auth needed for Project 1.
- **`--min-instances 1`** keeps one instance always warm (no cold starts, ~3-8s response times expected)
- **Azure OpenAI:** Same instance/credentials used for both LLM generation and embedding. Code automatically falls back if separate embedding credentials aren't provided.
- **Cost:** ~$1-3/day with `--min-instances 1` (varies by query volume and inference time). Adjust to `0` if cost is a concern.
- **Next steps after beta:** Project 2 (MVP) requires CV service integration + Zitadel auth + `X-API-Key` + `X-User-Id` header handling

---

---

## Final Deployment Command (With All Placeholders)

Below is the **complete, final deployment command** with all placeholders clearly marked for you to fill in:

```bash
gcloud run deploy phera-rag-beta \
  --image europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:latest \
  --region europe-west10 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 5 \
  --timeout 3600 \
  --add-cloudsql-instances phera-medical-agent:europe-west10:YOUR_INSTANCE_NAME \
  --set-env-vars "ENVIRONMENT=production,\
DEPLOYMENT_MODE=beta,\
GCP_PROJECT_ID=phera-medical-agent,\
GCP_BUCKET_NAME=phera_researchpaper,\
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY,\
AZURE_OPENAI_ENDPOINT=https://YOUR_AZURE_RESOURCE_NAME.openai.azure.com/,\
AZURE_OPENAI_API_VERSION=2024-02-15-preview,\
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o,\
AZURE_OPENAI_MINI_DEPLOYMENT_NAME=gpt-4o-mini,\
DATABASE_URL=postgresql+asyncpg://femtech:YOUR_POSTGRES_PASSWORD@/femtech_medical?host=/cloudsql/phera-medical-agent:europe-west10:YOUR_INSTANCE_NAME,\
POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD"
```

### Placeholders to Replace

| Placeholder | Where to Get | Example |
|---|---|---|
| `YOUR_INSTANCE_NAME` | From `.env`: `GCP_CLOUD_SQL_INSTANCE` or `gcloud sql instances list` | `phera-pg-main` |
| `YOUR_AZURE_OPENAI_API_KEY` | From `.env`: `AZURE_OPENAI_API_KEY` or Azure Portal → Keys | `DwtiHFFPDHLEX9YUGTaCctbB...` |
| `YOUR_AZURE_RESOURCE_NAME` | From `.env` or Azure Portal → Endpoint URL | `YOUR_AZURE_RESOURCE_NAME` (from `https://YOUR_AZURE_RESOURCE_NAME.openai.azure.com/`) |
| `YOUR_POSTGRES_PASSWORD` | From `.env`: `POSTGRES_PASSWORD` or your Cloud SQL password | `YOUR_POSTGRES_PASSWORD` |

### Example (After Filling In)

```bash
gcloud run deploy phera-rag-beta \
  --image europe-west10-docker.pkg.dev/phera-medical-agent/phera-images/rag-backend:latest \
  --region europe-west10 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 5 \
  --timeout 3600 \
  --add-cloudsql-instances phera-medical-agent:europe-west10:phera-pg-main \
  --set-env-vars "ENVIRONMENT=production,\
DEPLOYMENT_MODE=beta,\
GCP_PROJECT_ID=phera-medical-agent,\
GCP_BUCKET_NAME=phera_researchpaper,\
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY,\
AZURE_OPENAI_ENDPOINT=https://YOUR_AZURE_RESOURCE_NAME.openai.azure.com/,\
AZURE_OPENAI_API_VERSION=2024-02-15-preview,\
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o,\
AZURE_OPENAI_MINI_DEPLOYMENT_NAME=gpt-4o-mini,\
DATABASE_URL=postgresql+asyncpg://femtech:YOUR_POSTGRES_PASSWORD@/femtech_medical?host=/cloudsql/phera-medical-agent:europe-west10:phera-pg-main,\
POSTGRES_PASSWORD=YOUR_POSTGRES_PASSWORD"
```

---

## Quick Reference — Common Commands

```bash
# Get service URL
gcloud run services describe phera-rag-beta --region europe-west10 --format "value(status.url)"

# View logs (last 50 lines)
gcloud run logs read phera-rag-beta --region europe-west10 --limit 50

# Stream logs (live)
gcloud run logs read phera-rag-beta --region europe-west10 --follow

# Update environment variables only (without rebuilding)
gcloud run deploy phera-rag-beta \
  --update-env-vars KEY=VALUE \
  --region europe-west10

# Delete service (careful!)
gcloud run delete phera-rag-beta --region europe-west10
```

---
