import mlflow
import joblib
import os
import shutil
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
        # 2. THE FIX: Download the RAW artifact instead of loading it as a pyfunc
        # MLflow stores the native model.pkl inside the 'model' directory of the run.
        print(f"Fetching raw native model from: {settings.model_uri}")
        
        # This gets the actual path to the raw pickle file MLflow stored
        raw_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"{settings.model_uri}/model.pkl"
        )

        # 3. Copy the pure file to your static folder
        # We use shutil.copy because the file MLflow stored is already a pure pickle
        # that was created before the MLflow wrapper was added.
        shutil.copy(raw_model_path, export_path)
        
        print(f"✅ Success! Pure native model promoted to: {export_path}")
        print(f"File size: {os.path.getsize(export_path) / (1024*1024):.2f} MB")
        print("🚀 This file is now 100% independent of MLflow.")

    except Exception as e:
        print(f"❌ Failed to promote model: {e}")
        print("Hint: If 'model.pkl' is not found, check if your model was saved as 'model.joblib' or 'model.xgb'")

if __name__ == "__main__":
    promote_model()