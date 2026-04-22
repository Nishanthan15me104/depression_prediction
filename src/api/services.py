import os
import joblib
import logging
import pandas as pd
from src.api.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# --- CONDITIONAL DEPENDENCY SECTION ---
# This allows the app to boot in cloud environments where heavy dev 
# tools like MLflow are intentionally omitted to save space.
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

    def load_model(self):
            """
            Loads the machine learning model based on the APP_ENV setting.
            """
            # Ensure we are checking the cleaned string
            current_env = self.env.strip().upper()
            logger.info(f"DEBUG: Detected Environment: '{current_env}'") 
            
            try:
                if current_env == "PROD":
                    model_path = "src/api/static_models/final_model.pkl"
                    
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Static model file missing at: {model_path}")

                    self.model = joblib.load(model_path)
                    logger.info("✅ PROD: Loaded static model successfully from .pkl.")
                
                else:
                    # Local Development logic
                    # Use the boolean flag we set at the top of the file
                    if not HAS_MLFLOW:
                        raise ImportError(
                            "MLflow library not found. Install it or set APP_ENV=PROD."
                        )
                    
                    # IMPORTANT: Notice we removed 'import mlflow.pyfunc' from here.
                    # We use the 'mlflow' object defined at the top of the file.
                    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                    self.model = mlflow.pyfunc.load_model(settings.model_uri)
                    logger.info("🛠️ DEV: Loaded model from MLflow.")

            except Exception as e:
                logger.error(f"❌ Error loading model: {str(e)}")

    def predict(self, data):
        """
        Generates a prediction for the provided input data.

        Args:
            data (dict or pd.DataFrame): The raw input features for prediction.

        Returns:
            The first element of the model's prediction array.

        Raises:
            RuntimeError: If a prediction is attempted before the model is loaded.
        """
        if self.model is None:
            logger.error("Prediction attempted but model is not loaded.")
            raise RuntimeError("Model is not loaded.")
        
        try:
            # Convert dictionary input to DataFrame to ensure feature names match
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Perform inference
            prediction = self.model.predict(data)
            
            # Models typically return an array/list; return the first value
            return prediction[0]
            
        except Exception as e:
            logger.error(f"❌ Prediction Logic Error: {str(e)}")
            raise e

# Instantiate as a singleton to be imported by the FastAPI routes
ml_service = PredictionService()