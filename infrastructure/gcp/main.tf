# =============================================================================
# FemTech Medical RAG Agent - GCP Infrastructure
# =============================================================================
# Terraform configuration for GCP resources:
# - Cloud Storage bucket for medical papers
# - Cloud SQL instance with PostgreSQL + pgvector
# - Cloud Run service configuration
# - IAM service accounts and permissions
#
# Usage:
#   cd infrastructure/gcp
#   terraform init
#   terraform plan -var="project_id=your-project-id"
#   terraform apply -var="project_id=your-project-id"
# =============================================================================

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (development, staging, production)"
  type        = string
  default     = "development"
}

variable "db_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-f1-micro"  # Use db-custom-2-4096 or larger for production
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "google" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Enable Required APIs
# =============================================================================

resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudsql.googleapis.com",
    "sqladmin.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
  
  service            = each.key
  disable_on_destroy = false
}

# =============================================================================
# Cloud Storage Bucket for Medical Papers
# =============================================================================

resource "google_storage_bucket" "medical_papers" {
  name     = "${var.project_id}-femtech-medical-papers"
  location = var.region
  
  # Storage class - Standard for frequent access
  storage_class = "STANDARD"
  
  # Uniform bucket-level access (recommended)
  uniform_bucket_level_access = true
  
  # Versioning for data protection
  versioning {
    enabled = true
  }
  
  # Lifecycle rules
  lifecycle_rule {
    condition {
      age = 365  # Move to nearline after 1 year
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  # CORS configuration for signed URLs
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }
  
  labels = {
    environment = var.environment
    service     = "femtech-medical-agent"
  }
  
  depends_on = [google_project_service.required_apis]
}

# =============================================================================
# Cloud SQL Instance (PostgreSQL with pgvector)
# =============================================================================

resource "google_sql_database_instance" "femtech_db" {
  name             = "femtech-db-${var.environment}"
  database_version = "POSTGRES_16"
  region           = var.region
  
  settings {
    tier = var.db_tier
    
    # Storage configuration
    disk_size         = 10  # GB, increase for production
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    
    # Backup configuration
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"  # UTC
      point_in_time_recovery_enabled = var.environment == "production"
      
      backup_retention_settings {
        retained_backups = 7
      }
    }
    
    # IP configuration
    ip_configuration {
      ipv4_enabled    = true
      private_network = null  # Use private IP for production
      
      # Authorized networks for development
      # Remove for production - use Cloud Run connector instead
      dynamic "authorized_networks" {
        for_each = var.environment == "development" ? [1] : []
        content {
          name  = "allow-all-dev"
          value = "0.0.0.0/0"
        }
      }
    }
    
    # Database flags for pgvector
    database_flags {
      name  = "cloudsql.enable_pg_cron"
      value = "on"
    }
    
    # Maintenance window
    maintenance_window {
      day  = 7  # Sunday
      hour = 3  # UTC
    }
    
    user_labels = {
      environment = var.environment
      service     = "femtech-medical-agent"
    }
  }
  
  deletion_protection = var.environment == "production"
  
  depends_on = [google_project_service.required_apis]
}

# Create the application database
resource "google_sql_database" "femtech_medical" {
  name     = "femtech_medical"
  instance = google_sql_database_instance.femtech_db.name
}

# Create the application user
resource "google_sql_user" "femtech_user" {
  name     = "femtech"
  instance = google_sql_database_instance.femtech_db.name
  password = var.db_password
}

# =============================================================================
# Service Account for Backend
# =============================================================================

resource "google_service_account" "backend" {
  account_id   = "femtech-backend-${var.environment}"
  display_name = "FemTech Backend Service Account"
  description  = "Service account for FemTech Medical RAG Agent backend"
}

# Storage Object Admin role
resource "google_storage_bucket_iam_member" "backend_storage" {
  bucket = google_storage_bucket.medical_papers.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.backend.email}"
}

# Cloud SQL Client role
resource "google_project_iam_member" "backend_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# Secret Manager Accessor role
resource "google_project_iam_member" "backend_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.backend.email}"
}

# =============================================================================
# Secrets (Optional - store sensitive values)
# =============================================================================

resource "google_secret_manager_secret" "db_password" {
  secret_id = "femtech-db-password-${var.environment}"
  
  replication {
    auto {}
  }
  
  labels = {
    environment = var.environment
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# =============================================================================
# Outputs
# =============================================================================

output "bucket_name" {
  description = "Name of the Cloud Storage bucket"
  value       = google_storage_bucket.medical_papers.name
}

output "bucket_url" {
  description = "URL of the Cloud Storage bucket"
  value       = google_storage_bucket.medical_papers.url
}

output "db_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.femtech_db.name
}

output "db_connection_name" {
  description = "Cloud SQL connection name for Cloud Run"
  value       = google_sql_database_instance.femtech_db.connection_name
}

output "db_public_ip" {
  description = "Cloud SQL public IP address"
  value       = google_sql_database_instance.femtech_db.public_ip_address
}

output "service_account_email" {
  description = "Backend service account email"
  value       = google_service_account.backend.email
}

output "database_url" {
  description = "Database connection URL for Cloud Run"
  value       = "postgresql+asyncpg://femtech:PASSWORD@/femtech_medical?host=/cloudsql/${google_sql_database_instance.femtech_db.connection_name}"
  sensitive   = true
}

