import os
import joblib
import logging
import pandas as pd
from collections import deque  # <--- NEW: Import deque
from src.api.config import settings

logger = logging.getLogger(__name__)

try:
    import mlflow.pyfunc
    HAS_MLFLOW = True
except (ImportError, ModuleNotFoundError):
    mlflow = None
    HAS_MLFLOW = False

class PredictionService:
    def __init__(self):
        self.model = None
        self.env = os.getenv("APP_ENV", "DEV").strip().upper()
        # NEW: Store the last 100 requests in memory
        self.prediction_buffer = deque(maxlen=100) 

    def load_model(self):
        # ... [KEEP YOUR EXISTING load_model LOGIC EXACTLY THE SAME] ...
        pass # Placeholder so you know not to delete your code

    def predict(self, data):
        if self.model is None:
            logger.error("Prediction attempted but model is not loaded.")
            raise RuntimeError("Model is not loaded.")
        
        try:
            if isinstance(data, dict):
                # NEW: Save the incoming data to our buffer before it gets altered
                self.prediction_buffer.append(data.copy())
                
                data = pd.DataFrame([data])
            
            prediction = self.model.predict(data)
            return prediction[0]
            
        except Exception as e:
            logger.error(f"❌ Prediction Logic Error: {str(e)}")
            raise e

ml_service = PredictionService()