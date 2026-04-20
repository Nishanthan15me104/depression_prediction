import os
import joblib
import logging
import pandas as pd
from src.api.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        # Determine environment; default to DEV if not set
        self.env = os.getenv("APP_ENV", "DEV")

    def load_model(self):
        """
        Loads the model based on the current environment.
        """
        logger.info(f"DEBUG: Detected Environment: '{self.env}'") 
        try:
            if self.env.strip() == "PROD":
                # PROD: Load the static .pkl file (self-contained)
                model_path = "src/api/static_models/final_model.pkl"
                self.model = joblib.load(model_path)
                logger.info("✅ PROD: Loaded static model successfully from .pkl.")
            else:
                # DEV: Load from MLflow tracking server
                import mlflow.pyfunc
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                self.model = mlflow.pyfunc.load_model(settings.model_uri)
                logger.info("🛠️ DEV: Loaded model from MLflow.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            # We don't raise here to prevent the whole app from crashing 
            # if the model is temporarily unavailable.

    def predict(self, data):
        """
        Handles the transformation of raw data and generates a prediction.
        """
        if self.model is None:
            logger.error("Prediction attempted but model is not loaded.")
            raise RuntimeError("Model is not loaded.")
        
        try:
            # FIX: If data is a dictionary (from your route), 
            # convert it to a DataFrame so it has the .columns attribute.
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Generate prediction
            prediction = self.model.predict(data)
            
            # Return the first result (typically models return an array)
            return prediction[0]
            
        except Exception as e:
            logger.error(f"❌ Prediction Logic Error: {str(e)}")
            raise e

# Instantiate the service singleton
ml_service = PredictionService()