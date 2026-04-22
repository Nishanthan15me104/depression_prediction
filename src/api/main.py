"""
Main execution file to launch the FastAPI server.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.config import settings
from src.api.routes import api_router
from src.api.services import ml_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI service...")
    ml_service.load_model()

app.include_router(api_router, prefix="/api/v1", tags=["Predictions"])

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "active", "model_loaded": ml_service.model is not None}