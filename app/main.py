"""
FemTech Medical RAG Agent - Main Application Entry Point

A mobile-first diagnostic platform for women's vaginal health using 
RAG-based medical reasoning over curated research papers.

IMPORTANT: This system is purely informational and NOT diagnostic.
All responses include appropriate medical disclaimers.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.core.config import settings
from app.core.exceptions import AppException

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize database connections, load models, warm caches
    - Shutdown: Close connections, cleanup resources
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # TODO: Initialize database connection pool
    # TODO: Initialize Langfuse client if configured
    # TODO: Warm up embedding model connection
    
    if settings.is_azure_openai_configured():
        logger.info("Azure OpenAI is configured")
    else:
        logger.warning("Azure OpenAI is NOT configured - LLM features will be unavailable")
    
    if settings.is_langfuse_configured():
        logger.info("Langfuse observability is configured")
    else:
        logger.warning("Langfuse is NOT configured - tracing will be disabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    # TODO: Close database connections
    # TODO: Flush Langfuse traces
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Application factory for creating the FastAPI instance.
    
    This pattern allows for easier testing and multiple app configurations.
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
## FemTech Medical RAG Agent API

A mobile-first diagnostic platform for women's vaginal health using 
RAG-based medical reasoning over curated research papers.

### ⚠️ Medical Disclaimer

This system is **purely informational** and is **NOT** intended to:
- Diagnose medical conditions
- Prescribe treatments or medications
- Replace professional medical advice

Always consult with a qualified healthcare provider for medical concerns.

### Features

- **Health Profile Management**: Store and update user health information
- **pH-Based Analysis**: Analyze vaginal pH readings with health context
- **Evidence-Based Insights**: Retrieve information from curated medical research
- **Risk Assessment**: Categorize readings into actionable risk levels

### Risk Levels

| Level | Description |
|-------|-------------|
| NORMAL | pH within healthy range, no concerning symptoms |
| MONITOR | Minor deviation or mild symptoms - monitor and track |
| CONCERNING | Notable deviation - consider consulting healthcare provider |
| URGENT | Significant concern - prompt medical consultation recommended |
        """,
        version=settings.app_version,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Include API routes
    app.include_router(router)

    return app


async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """Handle application-specific exceptions."""
    logger.warning(
        f"AppException: {exc.error_code} - {exc.message}",
        extra={"path": request.url.path, "details": exc.details},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        f"Unhandled exception: {exc}",
        extra={"path": request.url.path},
    )
    
    # In production, don't expose internal error details
    error_message = (
        str(exc) if settings.is_development else "An unexpected error occurred"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": error_message,
            }
        },
    )


# Create the application instance
app = create_application()


# Root endpoint
@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "FemTech Medical RAG Agent API",
        "docs": "/docs" if settings.is_development else None,
        "health": "/health",
        "disclaimer": (
            "This system is purely informational and NOT intended to diagnose "
            "medical conditions or replace professional medical advice."
        ),
    }


