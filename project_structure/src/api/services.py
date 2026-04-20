import os
import joblib
import logging
from src.api.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self.env = os.getenv("APP_ENV", "DEV")

    # This MUST be indented with 4 spaces/1 tab
    def load_model(self):
        logger.info(f"DEBUG: Detected Environment: '{self.env}'") 
        try:
            if self.env.strip() == "PROD":
                model_path = "src/api/static_models/final_model.pkl"
                self.model = joblib.load(model_path)
                logger.info("✅ PROD: Loaded static model successfully.")
            else:
                import mlflow.pyfunc
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                self.model = mlflow.pyfunc.load_model(settings.model_uri)
                logger.info("🛠️ DEV: Loaded model from MLflow.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")

ml_service = PredictionService()