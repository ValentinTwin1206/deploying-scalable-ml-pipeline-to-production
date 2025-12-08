"""
Unit tests for FastAPI application endpoints.
"""

# standard library imports
import pickle
import pathlib 

import pytest
from fastapi.testclient import TestClient

# own imports
from main import app, model, encoder, lb
import main as main_module

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def load_model_artifacts():
    """
    Fixture to load model artifacts before running tests.
    This simulates the startup event that loads the model.
    """
    
    # Define paths using pathlib
    model_dir = pathlib.Path(__file__).parent / "model"
    model_path = model_dir / "model.pkl"
    encoder_path = model_dir / "encoder.pkl"
    lb_path = model_dir / "lb.pkl"
    
    # Load model artifacts if they exist
    if model_path.exists():
        with open(model_path, "rb") as f:
            main_module.model = pickle.load(f)
    
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            main_module.encoder = pickle.load(f)
    
    if lb_path.exists():
        with open(lb_path, "rb") as f:
            main_module.lb = pickle.load(f)
    
    yield
    
    # Cleanup (optional)
    main_module.model = None
    main_module.encoder = None
    main_module.lb = None


def test_get_root():
    """
    Test GET request on the root endpoint.
    Should return a welcome message.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]


def test_post_predict_low_income():
    """
    Test POST request on /predict endpoint with data that should predict <=50K.
    Tests a case with lower education and income indicators.
    """
    # Sample data for someone likely to earn <=50K
    data = {
        "age": 25,
        "workclass": "Private",
        "fnlgt": 226802,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_post_predict_high_income():
    """
    Test POST request on /predict endpoint with data that should predict >50K.
    Tests a case with higher education and income indicators.
    """
    # Sample data for someone likely to earn >50K
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    
    response = client.post("/predict", json=data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["<=50K", ">50K"]
