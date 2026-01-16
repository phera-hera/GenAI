"""API route definitions."""

from fastapi import APIRouter

from medical_agent.api.routes.health import router as health_router

# Main API router that includes all sub-routers
router = APIRouter()

# Include health check routes
router.include_router(health_router, tags=["Health"])

__all__ = ["router"]


