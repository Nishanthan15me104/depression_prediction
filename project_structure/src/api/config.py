"""
Configuration module.
Centralizes all environment variables and application settings.
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Depression Prediction API"
    api_version: str = "v1.0.0"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_uri: str = "models:/Depression_Classifier_Final/latest"
    api_key: str = "dev-secret-key-123"

    class Config:
        env_file = ".env"

settings = Settings()