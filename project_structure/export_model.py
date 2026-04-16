import mlflow
import joblib
import os
from src.api.config import settings

def promote_model():
    # 1. Setup paths
    export_dir = "src/api/static_models"
    export_path = os.path.join(export_dir, "final_model.pkl")
    
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    print(f"Connecting to MLflow at: {settings.mlflow_tracking_uri}")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        # 2. Load the best model from MLflow
        # Note: We use the URI from your settings
        print(f"Fetching model: {settings.model_uri}")
        model = mlflow.pyfunc.load_model(settings.model_uri)

        # 3. Save as a static pickle file
        # We unwrap the underlying sklearn/xgboost pipeline from the MLflow wrapper
        joblib.dump(model._model_impl, export_path)
        
        print(f"✅ Success! Model promoted to: {export_path}")
        print(f"File size: {os.path.getsize(export_path) / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"❌ Failed to promote model: {e}")

if __name__ == "__main__":
    promote_model()