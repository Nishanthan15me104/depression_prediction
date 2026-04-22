"""
FastAPI routing module.
Handles incoming requests and maps model outputs to standardized response schemas.
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
    # Ensure model is ready
    if ml_service.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        # 1. Prepare input data
        input_dict = data.model_dump(by_alias=True)
        
        # 2. Run prediction in a separate thread to keep API responsive
        raw_prediction = await asyncio.to_thread(ml_service.predict, input_dict)
        
        # 3. MANUALLY CONSTRUCT the response to match PredictionOutput schema
        # Since the raw .pkl model only returns a label (0 or 1), 
        # we map it back to the format the API expects.
        response_payload = {
            "status": "success",
            "prediction": int(raw_prediction),
            "probability_0": 0.0,  # Note: Static models often don't provide proba easily
            "probability_1": 0.0,
            "message": "Model inference executed successfully."
        }
        
        return response_payload
        
    except Exception as e:
        logger.error(f"❌ Route Prediction Error: {str(e)}")
        # Provide a more descriptive error if possible
        raise HTTPException(
            status_code=400, 
            detail=f"Data processing error: {str(e)}"
        )