"""
Production Entry Point for the Depression Prediction API.
"""
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_api")

if __name__ == "__main__":
    logger.info("Initializing Depression Prediction API Launch Sequence...")
    uvicorn.run(
        "src.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )