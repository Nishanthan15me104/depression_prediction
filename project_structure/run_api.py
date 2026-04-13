"""
Main execution file to launch the FastAPI server.
Kept outside the main application package to separate execution from definition.
"""
from fastapi import FastAPI
from src.api.routes import api_router, load_mlflow_model
import uvicorn

# Initialize the application
app = FastAPI(
    title="Depression Prediction API",
    description="Real-time inference API backed by MLflow Model Registry",
    version="1.0.0"
)

# Load the model into memory exactly once when the server starts
@app.on_event("startup")
def startup_event():
    """Startup hook to load the MLflow model into memory."""
    load_mlflow_model()

# Include the routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "active", "system": "Depression Prediction API"}

if __name__ == "__main__":
    # Run the server on localhost port 8000
    print("Starting FastAPI server...")
    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=True)