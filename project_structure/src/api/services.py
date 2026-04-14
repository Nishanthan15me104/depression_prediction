"""
Service Layer.
Separates ML execution business logic from HTTP routing.
"""
import pandas as pd
import mlflow.pyfunc
import logging
from src.api.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None

    def load_model(self):
        try:
            logger.info("Initializing MLflow connection...")
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self.model = mlflow.pyfunc.load_model(settings.model_uri)
            logger.info(f"Model successfully loaded: {settings.model_uri}")
        except Exception as e:
            logger.error(f"Critical error loading model: {e}")

    def predict(self, input_dict: dict) -> dict:
        if self.model is None:
            raise ValueError("Model is not loaded.")

        input_df = pd.DataFrame([input_dict])
        prediction = int(self.model.predict(input_df)[0])
        
        base_model = self.model._model_impl 
        probabilities = base_model.predict_proba(input_df)[0].tolist()
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability_0": round(probabilities[0], 4),
            "probability_1": round(probabilities[1], 4),
            "message": "Depression detected" if prediction == 1 else "No depression detected"
        }

ml_service = PredictionService()