"""
FastAPI app to predict penguin species using a trained XGBoost model.
Includes input validation, manual one-hot encoding, and logging.
"""

import os
import logging
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import joblib
from typing import Dict, Any


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Island(str, Enum):
    """Enum for valid island values."""
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"


class Sex(str, Enum):
    """Enum for valid sex values."""
    Male = "male"
    Female = "female"


class PenguinFeatures(BaseModel):
    """Pydantic model for penguin feature input validation."""
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    predicted_species: str
    confidence_scores: Dict[str, float]


# Initialize FastAPI app
app = FastAPI(
    title="Penguin Species Prediction API",
    description="API to predict penguin species using XGBoost model",
    version="1.0.0"
)

# Global variables for model and preprocessing artifacts
model = None
label_encoder = None
feature_columns = None


def load_model_and_artifacts() -> None:
    """
    Load the trained model, label encoder, and feature columns.
    """
    global model, label_encoder, feature_columns
    
    try:
        # Load the XGBoost model
        model_path = "app/data/model.json"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info("Model loaded successfully from app/data/model.json")
        
        # Load label encoder
        label_encoder_path = "app/data/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            logger.info("Label encoder loaded successfully")
        else:
            logger.warning("Label encoder not found, using default class names")
            # Fallback: create a simple mapping based on known penguin species
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = ['Adelie', 'Chinstrap', 'Gentoo']
        
        # Load feature columns
        feature_columns_path = "app/data/feature_columns.pkl"
        if os.path.exists(feature_columns_path):
            feature_columns = joblib.load(feature_columns_path)
            logger.info("Feature columns loaded successfully")
        else:
            logger.warning("Feature columns not found, using default column order")
            # Fallback: define expected columns based on training
            feature_columns = [
                'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                'body_mass_g', 'year', 'sex_female', 'sex_male', 
                'island_Biscoe', 'island_Dream', 'island_Torgersen'
            ]
        
        logger.info(f"Expected feature columns: {feature_columns}")
        
    except Exception as e:
        logger.error(f"Error loading model and artifacts: {str(e)}")
        raise


def preprocess_input(penguin_data: PenguinFeatures) -> pd.DataFrame:
    """
    Preprocess input data to match the training format with consistent one-hot encoding.
    
    Args:
        penguin_data (PenguinFeatures): Input penguin features
        
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction
    """
    try:
        logger.debug(f"Preprocessing input data: {penguin_data.dict()}")
        
        # Convert to dictionary
        data_dict = penguin_data.dict()
        
        # Create base features
        processed_data = {
            'bill_length_mm': data_dict['bill_length_mm'],
            'bill_depth_mm': data_dict['bill_depth_mm'],
            'flipper_length_mm': data_dict['flipper_length_mm'],
            'body_mass_g': data_dict['body_mass_g'],
            'year': data_dict['year']
        }
        
        # Manual one-hot encoding for sex
        processed_data['sex_female'] = 1 if data_dict['sex'] == 'female' else 0
        processed_data['sex_male'] = 1 if data_dict['sex'] == 'male' else 0
        
        # Manual one-hot encoding for island
        processed_data['island_Biscoe'] = 1 if data_dict['island'] == 'Biscoe' else 0
        processed_data['island_Dream'] = 1 if data_dict['island'] == 'Dream' else 0
        processed_data['island_Torgersen'] = 1 if data_dict['island'] == 'Torgersen' else 0
        
        # Create DataFrame
        df = pd.DataFrame([processed_data])
        
        # Ensure columns are in the same order as training
        if feature_columns is not None:
            # Reorder columns to match training
            df = df.reindex(columns=feature_columns, fill_value=0)
        
        logger.debug(f"Preprocessed data shape: {df.shape}")
        logger.debug(f"Preprocessed data columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing input data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing input: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model and artifacts when the application starts."""
    logger.info("Starting up the application...")
    load_model_and_artifacts()
    logger.info("Application startup complete")


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    logger.info("Root endpoint accessed")
    return {"message": "Penguin Species Prediction API is running!"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_penguin_species(penguin_data: PenguinFeatures):
    """
    Predict penguin species based on input features.
    
    Args:
        penguin_data (PenguinFeatures): Input penguin features
        
    Returns:
        PredictionResponse: Predicted species and confidence scores
    """
    try:
        logger.info(f"Prediction request received: {penguin_data.dict()}")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Validate input values using Pydantic (automatic validation)
        # Additional validation can be added here if needed
        if penguin_data.bill_length_mm <= 0:
            logger.warning("Invalid bill_length_mm value")
            raise HTTPException(status_code=400, detail="bill_length_mm must be positive")
        
        if penguin_data.bill_depth_mm <= 0:
            logger.warning("Invalid bill_depth_mm value")
            raise HTTPException(status_code=400, detail="bill_depth_mm must be positive")
        
        if penguin_data.flipper_length_mm <= 0:
            logger.warning("Invalid flipper_length_mm value")
            raise HTTPException(status_code=400, detail="flipper_length_mm must be positive")
        
        if penguin_data.body_mass_g <= 0:
            logger.warning("Invalid body_mass_g value")
            raise HTTPException(status_code=400, detail="body_mass_g must be positive")
        
        # Preprocess input data
        X_input = preprocess_input(penguin_data)
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]
        
        # Convert prediction to species name
        predicted_species = label_encoder.inverse_transform([prediction])[0]
        
        # Create confidence scores dictionary
        species_names = label_encoder.classes_
        confidence_scores = {
            species: float(prob) for species, prob in zip(species_names, prediction_proba)
        }
        
        logger.info(f"Prediction successful: {predicted_species} with confidence {max(confidence_scores.values()):.4f}")
        
        return PredictionResponse(
            predicted_species=predicted_species,
            confidence_scores=confidence_scores
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """
    Handle validation errors and return a more user-friendly error message.
    """
    logger.warning(f"Validation error: {exc}")
    
    # Extract validation error details
    error_details = []
    for error in exc.detail:
        field = error.get('loc', ['unknown'])[-1]
        message = error.get('msg', 'Invalid value')
        error_details.append(f"{field}: {message}")
    
    return HTTPException(
        status_code=400,
        detail={
            "error": "Input validation failed",
            "details": error_details,
            "valid_sex_values": ["male", "female"],
            "valid_island_values": ["Torgersen", "Biscoe", "Dream"]
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


