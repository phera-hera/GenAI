"""
Langfuse Observability Client

Provides tracing and observability for the RAG agent pipeline.
All LLM calls, retrievals, and agent steps are logged for monitoring.
"""

import logging
from functools import wraps
from typing import Any, Callable

from langfuse import Langfuse

# Try to import decorators (may not be available in all versions)
try:
    from langfuse.decorators import langfuse_context, observe
    _DECORATORS_AVAILABLE = True
except ImportError:
    # Provide stub implementations for older versions
    langfuse_context = None
    observe = None
    _DECORATORS_AVAILABLE = False

from app.core.config import settings
from app.core.exceptions import ObservabilityError

logger = logging.getLogger(__name__)


class LangfuseClient:
    """
    Client for Langfuse observability platform.
    
    Provides:
    - Trace creation and management
    - LLM call logging with latency and token usage
    - RAG retrieval logging
    - Custom event tracking
    """
    
    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        """
        Initialize Langfuse client.
        
        Args:
            public_key: Langfuse public key
            secret_key: Langfuse secret key
            host: Langfuse host URL
        """
        self.public_key = public_key or settings.langfuse_public_key
        self.secret_key = secret_key or settings.langfuse_secret_key
        self.host = host or settings.langfuse_host
        
        self._client: Langfuse | None = None
    
    @property
    def client(self) -> Langfuse:
        """Get or create the Langfuse client."""
        if self._client is None:
            if not self.is_configured():
                raise ObservabilityError("Langfuse is not configured")
            
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
        return self._client
    
    def is_configured(self) -> bool:
        """Check if Langfuse is properly configured."""
        return bool(self.public_key and self.secret_key)
    
    def verify_connection(self) -> bool:
        """
        Verify connection to Langfuse.
        
        Returns:
            True if connection is successful
            
        Raises:
            ObservabilityError: If connection fails
        """
        if not self.is_configured():
            raise ObservabilityError("Langfuse is not configured")
        
        try:
            # Auth check via API
            self.client.auth_check()
            logger.info("Successfully connected to Langfuse")
            return True
        except Exception as e:
            raise ObservabilityError(f"Failed to connect to Langfuse: {e}")
    
    def create_trace(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        """
        Create a new trace for tracking an operation.
        
        Args:
            name: Name of the trace
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional metadata dict
            tags: Optional list of tags
            
        Returns:
            Langfuse trace object
        """
        if not self.is_configured():
            logger.warning("Langfuse not configured, skipping trace creation")
            return None
        
        try:
            trace = self.client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or [],
            )
            return trace
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def log_llm_call(
        self,
        trace,
        name: str,
        model: str,
        input_messages: list[dict],
        output: str,
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log an LLM call to Langfuse.
        
        Args:
            trace: Parent trace object
            name: Name of the generation
            model: Model identifier
            input_messages: Input messages
            output: Generated output
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens)
            latency_ms: Latency in milliseconds
            metadata: Additional metadata
        """
        if trace is None:
            return
        
        try:
            trace.generation(
                name=name,
                model=model,
                input=input_messages,
                output=output,
                usage=usage,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error(f"Failed to log LLM call: {e}")
    
    def log_retrieval(
        self,
        trace,
        name: str,
        query: str,
        documents: list[dict],
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log a retrieval operation to Langfuse.
        
        Args:
            trace: Parent trace object
            name: Name of the retrieval span
            query: Search query
            documents: Retrieved documents
            metadata: Additional metadata
        """
        if trace is None:
            return
        
        try:
            span = trace.span(
                name=name,
                input={"query": query},
                output={"documents": documents, "count": len(documents)},
                metadata=metadata or {},
            )
            return span
        except Exception as e:
            logger.error(f"Failed to log retrieval: {e}")
            return None
    
    def log_event(
        self,
        trace,
        name: str,
        level: str = "DEFAULT",
        message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log a custom event to Langfuse.
        
        Args:
            trace: Parent trace object
            name: Event name
            level: Log level (DEBUG, DEFAULT, WARNING, ERROR)
            message: Optional event message
            metadata: Additional metadata
        """
        if trace is None:
            return
        
        try:
            trace.event(
                name=name,
                level=level,
                metadata={"message": message, **(metadata or {})},
            )
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
    
    def flush(self):
        """Flush all pending events to Langfuse."""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")
    
    def shutdown(self):
        """Shutdown the Langfuse client gracefully."""
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown Langfuse: {e}")


# Global client instance
_langfuse_client: LangfuseClient | None = None


def get_langfuse_client() -> LangfuseClient:
    """Get or create the global Langfuse client."""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = LangfuseClient()
    return _langfuse_client


# Decorator for easy tracing
def trace_operation(
    name: str | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator to automatically trace a function with Langfuse.
    
    Args:
        name: Optional trace name (defaults to function name)
        capture_input: Whether to capture function inputs
        capture_output: Whether to capture function output
    
    Example:
        @trace_operation(name="medical_query")
        def process_query(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            client = get_langfuse_client()
            
            if not client.is_configured():
                # Run without tracing if not configured
                return func(*args, **kwargs)
            
            trace_name = name or func.__name__
            
            # Create trace
            trace = client.create_trace(
                name=trace_name,
                metadata={
                    "function": func.__name__,
                    "module": func.__module__,
                },
            )
            
            try:
                # Log input if requested
                if capture_input and trace:
                    client.log_event(
                        trace,
                        name="input",
                        metadata={"args": str(args), "kwargs": str(kwargs)},
                    )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log output if requested
                if capture_output and trace:
                    client.log_event(
                        trace,
                        name="output",
                        metadata={"result": str(result)[:1000]},  # Truncate
                    )
                
                return result
                
            except Exception as e:
                # Log error
                if trace:
                    client.log_event(
                        trace,
                        name="error",
                        level="ERROR",
                        message=str(e),
                    )
                raise
            finally:
                client.flush()
        
        return wrapper
    return decorator


# Re-export Langfuse decorators for convenience (if available)
__all__ = [
    "LangfuseClient",
    "get_langfuse_client",
    "trace_operation",
]

# Only export decorators if available
if _DECORATORS_AVAILABLE:
    __all__.extend(["observe", "langfuse_context"])

