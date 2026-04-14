"""
FastAPI routing module.
"""
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Depends
from src.api.schemas import DepressionPredictionInput, PredictionOutput
from src.api.services import ml_service
from src.api.security import verify_api_key

logger = logging.getLogger(__name__)
api_router = APIRouter()

@api_router.post("/predict", response_model=PredictionOutput)
async def predict_depression(
    data: DepressionPredictionInput, 
    api_key: str = Depends(verify_api_key)
):
    if ml_service.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        input_dict = data.model_dump(by_alias=True)
        result = await asyncio.to_thread(ml_service.predict, input_dict)
        return result
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail="Data processing error.")