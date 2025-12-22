# =============================================================================
# Cloud Run Service Configuration
# =============================================================================
# This file defines the Cloud Run service for the FemTech Medical RAG Agent.
# Apply this after the main.tf resources are created and Docker image is pushed.
#
# Prerequisites:
#   - Docker image pushed to Artifact Registry
#   - Cloud SQL instance running
#   - Secrets configured in Secret Manager
# =============================================================================

variable "image_url" {
  description = "Docker image URL in Artifact Registry"
  type        = string
  default     = ""  # Set when deploying
}

variable "min_instances" {
  description = "Minimum number of instances (0 for scale to zero)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

# =============================================================================
# Artifact Registry for Docker Images
# =============================================================================

resource "google_artifact_registry_repository" "femtech" {
  location      = var.region
  repository_id = "femtech-images"
  description   = "Docker images for FemTech Medical RAG Agent"
  format        = "DOCKER"
  
  labels = {
    environment = var.environment
  }
  
  depends_on = [google_project_service.required_apis]
}

# =============================================================================
# Cloud Run Service
# =============================================================================

resource "google_cloud_run_v2_service" "femtech_api" {
  count    = var.image_url != "" ? 1 : 0
  name     = "femtech-api-${var.environment}"
  location = var.region
  
  template {
    service_account = google_service_account.backend.email
    
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }
    
    containers {
      image = var.image_url
      
      ports {
        container_port = 8000
      }
      
      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
      
      # Environment variables
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      
      env {
        name  = "GCP_BUCKET_NAME"
        value = google_storage_bucket.medical_papers.name
      }
      
      # Database URL with Cloud SQL socket
      env {
        name  = "DATABASE_URL"
        value = "postgresql+asyncpg://femtech:PASSWORD@/femtech_medical?host=/cloudsql/${google_sql_database_instance.femtech_db.connection_name}"
      }
      
      # Secrets from Secret Manager
      env {
        name = "POSTGRES_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }
      
      # Startup probe
      startup_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        initial_delay_seconds = 10
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 3
      }
      
      # Liveness probe
      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        timeout_seconds   = 3
        period_seconds    = 30
        failure_threshold = 3
      }
    }
    
    # Cloud SQL connection
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.femtech_db.connection_name]
      }
    }
  }
  
  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  
  labels = {
    environment = var.environment
    service     = "femtech-medical-agent"
  }
}

# =============================================================================
# IAM - Allow unauthenticated access (for public API)
# =============================================================================
# Remove this for production if you want to require authentication

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count    = var.image_url != "" && var.environment == "development" ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.femtech_api[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# =============================================================================
# Outputs
# =============================================================================

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.femtech.repository_id}"
}

output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = var.image_url != "" ? google_cloud_run_v2_service.femtech_api[0].uri : "Not deployed - set image_url variable"
}

