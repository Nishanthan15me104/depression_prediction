import os
import joblib
import logging
import pandas as pd
from collections import deque
from src.api.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# --- CONDITIONAL DEPENDENCY SECTION ---
try:
    import mlflow.pyfunc
    HAS_MLFLOW = True
except (ImportError, ModuleNotFoundError):
    mlflow = None
    HAS_MLFLOW = False

class PredictionService:
    """
    Service class to handle machine learning model lifecycle and inference.
    
    This service supports two modes:
    1. PROD: Loads a self-contained static .pkl file (ideal for Cloud/Docker).
    2. DEV: Connects to an MLflow tracking server to pull the latest model.
    """

    def __init__(self):
        """
        Initializes the service and determines the execution environment.
        """
        self.model = None
        # Normalize environment variable to handle whitespace or casing issues
        self.env = os.getenv("APP_ENV", "DEV").strip().upper()
        
        # NEW: In-Memory Buffer for Monitoring
        # Stores the last 100 requests to compare against reference data in Evidently AI
        self.prediction_buffer = deque(maxlen=100) 

    def load_model(self):
        """
        Loads the machine learning model based on the APP_ENV setting.
        """
        current_env = self.env.strip().upper()
        logger.info(f"🔍 Attempting to load model in '{current_env}' mode...") 
        
        try:
            if current_env == "PROD":
                # Path to the exported production model
                model_path = "src/api/static_models/final_model.pkl"
                
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Static model file missing at: {model_path}")

                self.model = joblib.load(model_path)
                logger.info("✅ PROD: Loaded static model successfully from .pkl.")
            
            else:
                # Local Development logic using MLflow
                if not HAS_MLFLOW:
                    raise ImportError(
                        "MLflow library not found. Install it or set APP_ENV=PROD."
                    )
                
                import mlflow.pyfunc
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                self.model = mlflow.pyfunc.load_model(settings.model_uri)
                logger.info("🛠️ DEV: Loaded model from MLflow.")

        except Exception as e:
            logger.error(f"❌ Critical Error loading model: {str(e)}")

    def predict(self, data):
        """
        Generates a prediction for the provided input data and buffers the request.

        Args:
            data (dict or pd.DataFrame): The raw input features for prediction.

        Returns:
            The first element of the model's prediction array.
        """
        if self.model is None:
            logger.error("Prediction attempted but model is not loaded.")
            raise RuntimeError("Model is not loaded. Check startup logs.")
        
        try:
            # NEW: Data Collection for Monitoring
            if isinstance(data, dict):
                # Save a copy to the buffer before converting to DataFrame
                self.prediction_buffer.append(data.copy())
                
                # Convert dictionary input to DataFrame for the model
                data = pd.DataFrame([data])
            
            # Perform inference
            prediction = self.model.predict(data)
            
            # Models typically return an array; return the first value
            return prediction[0]
            
        except Exception as e:
            logger.error(f"❌ Prediction Logic Error: {str(e)}")
            raise e

# Instantiate as a singleton to be imported by the FastAPI routes
ml_service = PredictionService()