import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Try to import the monitoring tool
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

from src.api.config import settings
from src.api.routes import api_router
from src.api.services import ml_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.api_version)

# 1. Standard Middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Isolated Prometheus Monitoring
if HAS_PROMETHEUS:
    try:
        # We initialize it but don't expose yet. 
        # The .instrument() call is what "watches" your routes.
        instrumentator = Instrumentator().instrument(app)
        logger.info("📊 Prometheus instrumentation enabled.")
    except Exception as e:
        logger.warning(f"Could not initialize monitoring: {e}")
        HAS_PROMETHEUS = False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI service...")
    # Load ML Model
    ml_service.load_model()
    
    # Only expose the /metrics endpoint if initialization succeeded
    if HAS_PROMETHEUS:
        instrumentator.expose(app, endpoint="/metrics", tags=["System"])

app.include_router(api_router, prefix="/api/v1", tags=["Predictions"])

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "active", 
        "model_loaded": ml_service.model is not None,
        "monitoring_active": HAS_PROMETHEUS
    }