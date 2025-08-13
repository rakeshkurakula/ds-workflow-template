from fastapi import FastAPI
from fastapi.responses import JSONResponse
from app.schemas import PredictionRequest, PredictionResponse
import joblib
import numpy as np
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DS Workflow API",
    description="A data science workflow API with health and prediction endpoints",
    version="1.0.0"
)

# Dummy model class for demonstration
class DummyModel:
    """A simple dummy model for demonstration purposes"""
    
    def __init__(self):
        self.model_name = "dummy_classifier"
        self.version = "1.0.0"
    
    def predict(self, features: list) -> float:
        """Make a dummy prediction based on sum of features"""
        return float(np.sum(features) * 0.1)
    
    def predict_proba(self, features: list) -> Dict[str, float]:
        """Return dummy prediction probabilities"""
        score = self.predict(features)
        prob_positive = min(max(score, 0.1), 0.9)  # Keep between 0.1 and 0.9
        return {
            "negative": 1.0 - prob_positive,
            "positive": prob_positive
        }

# Global model instance
model = DummyModel()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_loaded": model is not None,
            "model_name": model.model_name if model else None,
            "model_version": model.version if model else None
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Prediction endpoint"""
    try:
        logger.info(f"Received prediction request with {len(request.features)} features")
        
        # Make prediction using dummy model
        prediction_score = model.predict(request.features)
        probabilities = model.predict_proba(request.features)
        
        # Determine prediction class (binary classification)
        predicted_class = "positive" if prediction_score > 0.5 else "negative"
        
        response = PredictionResponse(
            prediction=predicted_class,
            score=prediction_score,
            probabilities=probabilities,
            model_name=model.model_name,
            model_version=model.version
        )
        
        logger.info(f"Prediction completed: {predicted_class} (score: {prediction_score:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Prediction failed",
                "message": str(e)
            }
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "DS Workflow API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
