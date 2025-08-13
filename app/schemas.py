from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    features: List[float] = Field(
        ...,
        description="List of numerical features for prediction",
        example=[1.0, 2.5, -0.3, 4.2]
    )
    model_version: Optional[str] = Field(
        None,
        description="Optional model version to use for prediction",
        example="1.0.0"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.0, 2.5, -0.3, 4.2],
                "model_version": "1.0.0"
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    prediction: str = Field(
        ...,
        description="Predicted class or value",
        example="positive"
    )
    score: float = Field(
        ...,
        description="Prediction confidence score",
        example=0.85
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities",
        example={"negative": 0.15, "positive": 0.85}
    )
    model_name: str = Field(
        ...,
        description="Name of the model used",
        example="dummy_classifier"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used",
        example="1.0.0"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "positive",
                "score": 0.85,
                "probabilities": {
                    "negative": 0.15,
                    "positive": 0.85
                },
                "model_name": "dummy_classifier",
                "model_version": "1.0.0"
            }
        }

class HealthResponse(BaseModel):
    """Response schema for health endpoint"""
    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded",
        example=True
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of the loaded model",
        example="dummy_classifier"
    )
    model_version: Optional[str] = Field(
        None,
        description="Version of the loaded model",
        example="1.0.0"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "dummy_classifier",
                "model_version": "1.0.0"
            }
        }
