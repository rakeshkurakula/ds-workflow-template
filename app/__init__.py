"""FastAPI application package for data science workflow.

This package contains the main FastAPI application with endpoints for
health checking and model predictions.
"""

__version__ = "1.0.0"
__author__ = "DS Workflow Team"

# Import main components for easier access
from .main import app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

__all__ = [
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse"
]
