"""
FastAPI application for ML model inference on census data.
"""

# built-in libs
import os
import pathlib
import pickle
from contextlib import asynccontextmanager
from typing import Literal


# third party libs
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# own imports
from starter.ml.data import process_data
from starter.ml.model import inference

# Global configuration - not needed for direct uvicorn usage
# API_HOST and API_PORT removed as they will be passed via uvicorn command

# Load model artifacts at startup
model = None
encoder = None
lb = None

# Categorical features for processing
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading model artifacts on startup."""
    global model, encoder, lb
    
    # Define paths using pathlib
    model_dir = pathlib.Path(__file__).parent / "model"
    model_path = model_dir / "model.pkl"
    encoder_path = model_dir / "encoder.pkl"
    lb_path = model_dir / "lb.pkl"
    
    # Load model files if they exist
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
    
    if lb_path.exists():
        with open(lb_path, "rb") as f:
            lb = pickle.load(f)
    
    yield
    
    # Cleanup (if needed)
    model = None
    encoder = None
    lb = None


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting income category based on census data",
    version="0.0.1",
    lifespan=lifespan
)

# Pydantic model for input data with hyphenated column names
class CensusInput(BaseModel):
    age: int = Field(json_schema_extra={"example": 39})
    workclass: Literal[
        "State-gov", "Self-emp-not-inc", "Private", "Federal-gov",
        "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked"
    ] = Field(json_schema_extra={"example": "State-gov"})
    fnlgt: int = Field(json_schema_extra={"example": 77516})
    education: Literal[
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ] = Field(json_schema_extra={"example": "Bachelors"})
    education_num: int = Field(alias="education-num", json_schema_extra={"example": 13})
    marital_status: Literal[
        "Never-married", "Married-civ-spouse", "Divorced",
        "Married-spouse-absent", "Separated", "Married-AF-spouse",
        "Widowed"
    ] = Field(alias="marital-status", json_schema_extra={"example": "Never-married"})
    occupation: Literal[
        "Adm-clerical", "Exec-managerial", "Handlers-cleaners",
        "Prof-specialty", "Other-service", "Sales", "Craft-repair",
        "Transport-moving", "Farming-fishing", "Machine-op-inspct",
        "Tech-support", "Protective-serv", "Armed-Forces",
        "Priv-house-serv"
    ] = Field(json_schema_extra={"example": "Adm-clerical"})
    relationship: Literal[
        "Not-in-family", "Husband", "Wife", "Own-child",
        "Unmarried", "Other-relative"
    ] = Field(json_schema_extra={"example": "Not-in-family"})
    race: Literal[
        "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
        "Other"
    ] = Field(json_schema_extra={"example": "White"})
    sex: Literal["Male", "Female"] = Field(json_schema_extra={"example": "Male"})
    capital_gain: int = Field(alias="capital-gain", json_schema_extra={"example": 2174})
    capital_loss: int = Field(alias="capital-loss", json_schema_extra={"example": 0})
    hours_per_week: int = Field(alias="hours-per-week", json_schema_extra={"example": 40})
    native_country: Literal[
        "United-States", "Cuba", "Jamaica", "India", "Mexico",
        "South", "Puerto-Rico", "Honduras", "England", "Canada",
        "Germany", "Iran", "Philippines", "Italy", "Poland",
        "Columbia", "Cambodia", "Thailand", "Ecuador", "Laos",
        "Taiwan", "Haiti", "Portugal", "Dominican-Republic",
        "El-Salvador", "France", "Guatemala", "China", "Japan",
        "Yugoslavia", "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland",
        "Trinadad&Tobago", "Greece", "Nicaragua", "Vietnam", "Hong",
        "Ireland", "Hungary", "Holand-Netherlands"
    ] = Field(alias="native-country", json_schema_extra={"example": "United-States"})

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }
    )


# Pydantic model for output data
class PredictionResponse(BaseModel):
    prediction: Literal["<=50K", ">50K"] = Field(
        description="Predicted income category"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "<=50K"
            }
        }
    )


@app.get("/")
async def root() -> dict:
    """
    Welcome endpoint for the API that returns a welcome dictionary.
    """
    return { "message": "Welcome to the Census Income Prediction API!"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=200
)
async def predict(input_data: CensusInput) -> PredictionResponse:
    """
    Perform model inference on census data.
    
    Args:
        input_data: Census data input conforming to CensusInput schema
        
    Returns:
        PredictionResponse: Prediction result (<=50K or >50K)
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    
    # Check if model is loaded
    if model is None or encoder is None or lb is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model artifacts exist in the 'model/' directory."
        )
        
    # Create DataFrame
    input_dataframe = pd.DataFrame([{
        "age": input_data.age,
        "workclass": input_data.workclass,
        "fnlgt": input_data.fnlgt,
        "education": input_data.education,
        "education-num": input_data.education_num,
        "marital-status": input_data.marital_status,
        "occupation": input_data.occupation,
        "relationship": input_data.relationship,
        "race": input_data.race,
        "sex": input_data.sex,
        "capital-gain": input_data.capital_gain,
        "capital-loss": input_data.capital_loss,
        "hours-per-week": input_data.hours_per_week,
        "native-country": input_data.native_country,
    }])
    
    # Process the data
    X, _, _, _ = process_data(
        input_dataframe,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    pred = inference(model, X)
    
    # Convert prediction to label
    prediction_label = lb.inverse_transform(pred)[0]
    
    return PredictionResponse(prediction=prediction_label)
