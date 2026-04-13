"""
FastAPI routing module.
Handles model loading from MLflow and serves the prediction endpoints.
"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import mlflow.pyfunc
from src.api.schemas import DepressionPredictionInput

api_router = APIRouter()
MODEL = None

def load_mlflow_model():
    """Connects to MLflow and loads the latest registered model."""
    global MODEL
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        model_uri = "models:/Depression_Classifier_Final/latest"
        MODEL = mlflow.pyfunc.load_model(model_uri)
        print(f"Model successfully loaded: {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")

@api_router.post("/predict")
async def predict_depression(data: DepressionPredictionInput):
    """Processes a single record and returns a prediction."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        # Convert to dict using aliases so "Work_Study_Hours" becomes "Work/Study Hours"
        input_dict = data.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        
        # Predict
        prediction = int(MODEL.predict(input_df)[0])
        
        # Access underlying model for probabilities
        base_model = MODEL._model_impl 
        probabilities = base_model.predict_proba(input_df)[0].tolist()
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability_0": round(probabilities[0], 4),
            "probability_1": round(probabilities[1], 4),
            "message": "Depression detected" if prediction == 1 else "No depression detected"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")