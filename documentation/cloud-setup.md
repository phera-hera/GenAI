# Cloud Infrastructure Setup Guide

This guide covers setting up all cloud services required for the FemTech Medical RAG Agent.

## Overview

The application uses the following cloud services:

| Service | Provider | Purpose |
|---------|----------|---------|
| Cloud Storage | GCP | Store medical research PDFs |
| Cloud SQL | GCP | PostgreSQL database with pgvector |
| Cloud Run | GCP | Host the FastAPI backend |
| Azure OpenAI | Azure | GPT-4o for reasoning, embeddings |
| Document Intelligence | Azure | PDF parsing and extraction |
| Langfuse | Cloud | Observability and tracing |

## Prerequisites

- GCP account with billing enabled
- Azure account with Azure OpenAI access
- Langfuse account (free tier available)
- Terraform >= 1.0 (for IaC setup)
- gcloud CLI installed and authenticated
- Docker installed (for local development)

---

## 1. GCP Project Setup

### Option A: Using Terraform (Recommended)

```bash
cd infrastructure/gcp

# Initialize Terraform
terraform init

# Create terraform.tfvars from example
cp variables.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Preview changes
terraform plan -var-file="terraform.tfvars"

# Apply configuration
terraform apply -var-file="terraform.tfvars"
```

### Option B: Manual Setup

#### Create a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select an existing one
3. Note your Project ID

#### Enable Required APIs

```bash
gcloud services enable \
  cloudsql.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com
```

#### Create Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://YOUR_PROJECT_ID-femtech-medical-papers

# Enable versioning
gsutil versioning set on gs://YOUR_PROJECT_ID-femtech-medical-papers
```

#### Create Cloud SQL Instance

```bash
# Create instance
gcloud sql instances create femtech-db \
  --database-version=POSTGRES_16 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --root-password=YOUR_SECURE_PASSWORD

# Create database
gcloud sql databases create femtech_medical \
  --instance=femtech-db

# Create user
gcloud sql users create femtech \
  --instance=femtech-db \
  --password=YOUR_USER_PASSWORD
```

#### Enable pgvector Extension

```bash
# Connect to the database
gcloud sql connect femtech-db --user=postgres

# In psql:
\c femtech_medical
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\q
```

#### Create Service Account

```bash
# Create service account
gcloud iam service-accounts create femtech-backend \
  --display-name="FemTech Backend"

# Grant Storage access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:femtech-backend@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Grant Cloud SQL access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:femtech-backend@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

# Create and download key (for local development)
gcloud iam service-accounts keys create ./service-account.json \
  --iam-account=femtech-backend@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

---

## 2. Azure OpenAI Setup

### Create Azure OpenAI Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create resource → "Azure OpenAI"
3. Select a region with GPT-4o availability
4. Choose Standard S0 pricing tier
5. Wait for deployment to complete

### Deploy Models

1. Go to [Azure OpenAI Studio](https://oai.azure.com)
2. Navigate to Deployments
3. Create deployment for **GPT-4o**:
   - Model: gpt-4o
   - Deployment name: `gpt-4o`
   - Tokens per minute: 30K+ recommended
4. Create deployment for **text-embedding-3-small**:
   - Model: text-embedding-3-small
   - Deployment name: `text-embedding-3-small`
   - Tokens per minute: 100K+ recommended

### Get Credentials

1. In Azure Portal, go to your Azure OpenAI resource
2. Navigate to "Keys and Endpoint"
3. Copy:
   - Endpoint URL
   - Key 1

### Environment Variables

```bash
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small
```

---

## 3. Azure Document Intelligence Setup

### Create Resource

1. Go to [Azure Portal](https://portal.azure.com)
2. Create resource → "Azure AI Document Intelligence"
3. Select pricing tier:
   - F0 (Free): 500 pages/month
   - S0 (Standard): Pay per page
4. Wait for deployment

### Get Credentials

1. Go to your Document Intelligence resource
2. Navigate to "Keys and Endpoint"
3. Copy:
   - Endpoint URL
   - Key 1

### Environment Variables

```bash
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key
```

---

## 4. Langfuse Setup

### Create Account

1. Go to [Langfuse Cloud](https://cloud.langfuse.com)
2. Sign up for free tier
3. Create a new project

### Get API Keys

1. Go to Settings → API Keys
2. Create new API key pair
3. Copy:
   - Public Key (pk-lf-...)
   - Secret Key (sk-lf-...)

### Environment Variables

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 5. Local Development Setup

### Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
```

### Start Local Database

```bash
# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Verify it's running
docker-compose ps
```

### Run Migrations

```bash
# Apply database migrations
alembic upgrade head
```

### Verify Setup

```bash
# Run infrastructure check
python scripts/setup_infrastructure.py --check-all
```

---

## 6. Deployment to Cloud Run

### Build and Push Docker Image

```bash
# Set variables
PROJECT_ID=your-project-id
REGION=us-central1

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build image
docker build -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/femtech-images/api:latest .

# Push image
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/femtech-images/api:latest
```

### Deploy to Cloud Run

```bash
# Deploy service
gcloud run deploy femtech-api \
  --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/femtech-images/api:latest \
  --region=${REGION} \
  --platform=managed \
  --add-cloudsql-instances=${PROJECT_ID}:${REGION}:femtech-db \
  --service-account=femtech-backend@${PROJECT_ID}.iam.gserviceaccount.com \
  --set-env-vars="ENVIRONMENT=production" \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID}" \
  --set-secrets="POSTGRES_PASSWORD=femtech-db-password:latest" \
  --allow-unauthenticated
```

---

## Troubleshooting

### GCP Issues

**"Permission denied" errors**
- Verify service account has correct roles
- Check if APIs are enabled
- Ensure credentials file path is correct

**Cloud SQL connection issues**
- Check authorized networks in Cloud SQL settings
- Verify database user and password
- For Cloud Run, ensure Cloud SQL instance is attached

### Azure Issues

**"Deployment not found" errors**
- Verify deployment names match exactly
- Check deployment status in Azure OpenAI Studio
- Ensure region supports the model

**Rate limiting**
- Increase tokens per minute in deployment
- Implement retry logic with exponential backoff

### Langfuse Issues

**"Authentication failed"**
- Verify public and secret keys
- Check host URL is correct
- Ensure keys are from the correct project

---

## Cost Optimization

### Development

- Use `db-f1-micro` for Cloud SQL (~$10/month)
- Use F0 tier for Document Intelligence (free)
- Langfuse hobby tier (free)
- Cloud Run scales to zero

### Production

- Right-size Cloud SQL based on load
- Use committed use discounts
- Enable Cloud SQL Insights for optimization
- Monitor token usage in Azure OpenAI

---

## Security Best Practices

1. **Never commit credentials** - Use environment variables
2. **Use Secret Manager** - Store sensitive values securely
3. **Restrict database access** - Use private IP in production
4. **Enable audit logging** - For compliance requirements
5. **Rotate credentials** - Regularly rotate API keys
6. **Use IAM** - Principle of least privilege

