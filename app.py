from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Define the request body structure
class WaterQualityRequest(BaseModel):
    ph: float
    turbidity: float

# Define the response body structure
class WaterQualityResponse(BaseModel):
    quality: str
    confidence: float
    probabilities: list

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the saved model
    model = joblib.load("gbm_model.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# Prediction function
def predict_water_quality(ph: float, turbidity: float) -> WaterQualityResponse:
    try:
        # Create a DataFrame from input values
        user_input = pd.DataFrame({'ph': [ph], 'turbidity': [turbidity]})
        
        # Predict the class probabilities
        probabilities = model.predict_proba(user_input)[0]
        predicted_class = model.predict(user_input)[0]
        
        # Get confidence and quality
        confidence = max(probabilities)
        quality = "Potable" if predicted_class == 1 else "Not Potable"
        
        # Create response
        response = WaterQualityResponse(
            quality=quality,
            confidence=confidence,
            probabilities=probabilities.tolist()
        )
        
        return response
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Define the endpoint
@app.post("/predict", response_model=WaterQualityResponse)
async def predict(request: WaterQualityRequest):
    result = predict_water_quality(request.ph, request.turbidity)
    return result

# Run the application (use this command only for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
