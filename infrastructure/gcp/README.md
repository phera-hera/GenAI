# GCP Infrastructure Setup

This directory contains Terraform configurations for provisioning GCP resources for the FemTech Medical RAG Agent.

## Resources Created

- **Cloud Storage Bucket**: For storing medical research PDFs
- **Cloud SQL Instance**: PostgreSQL 16 with pgvector support
- **Service Account**: With appropriate IAM permissions
- **Secret Manager Secrets**: For secure credential storage

## Prerequisites

1. Install Terraform (>= 1.0)
2. Install Google Cloud SDK
3. Authenticate with GCP: `gcloud auth application-default login`
4. Create a GCP project and note the Project ID

## Quick Start

### 1. Initialize Terraform

```bash
cd infrastructure/gcp
terraform init
```

### 2. Configure Variables

```bash
cp variables.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### 3. Review the Plan

```bash
terraform plan -var-file="terraform.tfvars"
```

### 4. Apply Configuration

```bash
terraform apply -var-file="terraform.tfvars"
```

### 5. Enable pgvector Extension

After the Cloud SQL instance is created, connect and enable pgvector:

```bash
# Connect via Cloud Shell or gcloud
gcloud sql connect femtech-db-development --user=postgres

# In the psql prompt:
\c femtech_medical
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

## Environment-Specific Configuration

### Development
- Uses `db-f1-micro` (shared CPU, 0.6 GB RAM)
- Public IP enabled with open access
- Deletion protection disabled

### Production
- Use `db-custom-2-4096` or larger
- Use private IP with VPC connector
- Enable deletion protection
- Enable point-in-time recovery

```hcl
# production.tfvars
environment = "production"
db_tier     = "db-custom-2-4096"
```

## Outputs

After applying, Terraform outputs:

| Output | Description |
|--------|-------------|
| `bucket_name` | Cloud Storage bucket name |
| `bucket_url` | Cloud Storage bucket URL |
| `db_instance_name` | Cloud SQL instance name |
| `db_connection_name` | Connection name for Cloud Run |
| `db_public_ip` | Database public IP address |
| `service_account_email` | Backend service account email |
| `database_url` | Database connection URL template |

## Connecting to Cloud SQL

### From Local Machine

1. Get the instance IP:
   ```bash
   terraform output db_public_ip
   ```

2. Add your IP to authorized networks (or use Cloud SQL Proxy)

3. Connect:
   ```bash
   psql -h <IP> -U femtech -d femtech_medical
   ```

### From Cloud Run

Use the Cloud SQL connection name with the Unix socket:
```
postgresql+asyncpg://femtech:PASSWORD@/femtech_medical?host=/cloudsql/PROJECT:REGION:femtech-db-development
```

## Cleanup

To destroy all resources:

```bash
terraform destroy -var-file="terraform.tfvars"
```

⚠️ **Warning**: This will delete all data including the database and stored PDFs.

## Cost Estimation

### Development (db-f1-micro)
- Cloud SQL: ~$10/month
- Cloud Storage: ~$0.02/GB/month
- Total: ~$10-15/month

### Production (db-custom-2-4096)
- Cloud SQL: ~$100/month
- Cloud Storage: ~$0.02/GB/month
- Total: ~$100-150/month

## Security Considerations

1. **Never commit `terraform.tfvars`** - It contains sensitive values
2. Use Secret Manager for credentials in production
3. Restrict database access with private IP in production
4. Rotate credentials regularly
5. Enable audit logging for compliance

