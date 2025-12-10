import logging
import traceback
from typing import Dict, Any

# Create logger
logger = logging.getLogger("agentic-rag")


def configure_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def log_exception(e: Exception, context: str = ""):
    """Log an exception with context."""
    if context:
        logger.error(f"{context}: {str(e)}", exc_info=True)
    else:
        logger.error(str(e), exc_info=True)


def exception_to_dict(e: Exception) -> Dict[str, Any]:
    """Convert exception to a dictionary for JSON responses (safe for clients)."""
    return {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc()
    }
