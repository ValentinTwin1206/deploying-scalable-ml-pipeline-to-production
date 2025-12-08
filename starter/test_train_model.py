"""
Unit tests for training pipeline, data processing, and model functions.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))
from train_model import _load_and_clean_data
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_with_spaces():
    """Fixture to create a temporary CSV file with spaces in data."""
    data = """ age, workclass, education, salary
 39, State-gov, Bachelors, <=50K
 50, Self-emp-not-inc, Bachelors, <=50K
 38, Private, HS-grad, <=50K
 39, State-gov, Bachelors, <=50K
"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_dataframe():
    """Fixture to provide a sample census-like dataframe."""
    data = {
        'age': [39, 50, 38, 53, 28],
        'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Private', 'Private'],
        'education': ['Bachelors', 'Bachelors', 'HS-grad', '11th', 'Bachelors'],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 
                          'Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                      'Handlers-cleaners', 'Prof-specialty'],
        'relationship': ['Not-in-family', 'Husband', 'Not-in-family', 'Husband', 'Wife'],
        'race': ['White', 'White', 'White', 'Black', 'Black'],
        'sex': ['Male', 'Male', 'Male', 'Male', 'Female'],
        'native-country': ['United-States', 'United-States', 'United-States',
                          'United-States', 'Cuba'],
        'salary': ['<=50K', '<=50K', '<=50K', '<=50K', '<=50K']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_model_data():
    """Fixture to provide sample training and test data for model."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    return X_train, y_train, X_test, y_test


@pytest.fixture
def trained_model(sample_model_data):
    """Fixture to provide a trained model."""
    X_train, y_train, _, _ = sample_model_data
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "random_state": 42
    }
    model = train_model(X_train, y_train, params)
    return model


# ============================================================================
# Tests for Data Cleaning (_load_and_clean_data)
# ============================================================================

def test_load_and_clean_data_removes_spaces_from_columns(sample_csv_with_spaces):
    """Test that column names have spaces removed."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    # Check that column names don't have leading/trailing spaces
    for col in df.columns:
        assert col == col.strip()
        assert not col.startswith(' ')
        assert not col.endswith(' ')


def test_load_and_clean_data_removes_spaces_from_values(sample_csv_with_spaces):
    """Test that string values have spaces removed."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    # Check that string values don't have leading/trailing spaces
    for col in df.columns:
        if df[col].dtype == 'object':
            for value in df[col]:
                assert value == value.strip()


def test_load_and_clean_data_removes_duplicates(sample_csv_with_spaces):
    """Test that duplicate rows are removed."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    # The sample CSV has a duplicate row (first and last)
    # After cleaning, we should have 3 unique rows instead of 4
    assert len(df) == 3


def test_load_and_clean_data_returns_dataframe(sample_csv_with_spaces):
    """Test that the function returns a pandas DataFrame."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    assert isinstance(df, pd.DataFrame)


def test_load_and_clean_data_preserves_data_integrity(sample_csv_with_spaces):
    """Test that data values are preserved (only spaces removed)."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    # Check that expected values exist (without spaces)
    assert 'State-gov' in df['workclass'].values
    assert 'Bachelors' in df['education'].values
    assert '<=50K' in df['salary'].values


def test_load_and_clean_data_handles_numeric_columns(sample_csv_with_spaces):
    """Test that numeric columns are preserved correctly."""
    df = _load_and_clean_data(sample_csv_with_spaces)
    
    # Age column should be numeric
    assert df['age'].dtype in ['int64', 'int32', 'float64']
    assert 39 in df['age'].values
    assert 50 in df['age'].values
    assert 38 in df['age'].values


# ============================================================================
# Tests for Data Processing (process_data)
# ============================================================================

def test_process_data_training_returns_correct_types(sample_dataframe):
    """Test that process_data returns correct types in training mode."""
    cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
    
    X, y, encoder, lb = process_data(
        sample_dataframe,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert encoder is not None
    assert lb is not None


def test_process_data_training_shapes(sample_dataframe):
    """Test that process_data returns correct shapes."""
    cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
    
    X, y, encoder, lb = process_data(
        sample_dataframe,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    assert X.shape[0] == len(sample_dataframe)
    assert y.shape[0] == len(sample_dataframe)
    assert X.shape[1] > len(cat_features)  # Should have more features after encoding


def test_process_data_inference_mode(sample_dataframe):
    """Test that process_data works correctly in inference mode."""
    cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
    
    # First, train to get encoder and lb
    X_train, y_train, encoder, lb = process_data(
        sample_dataframe,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    # Then use in inference mode
    X_test, y_test, _, _ = process_data(
        sample_dataframe,
        categorical_features=cat_features,
        label='salary',
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    assert X_test.shape == X_train.shape
    assert y_test.shape == y_train.shape


# ============================================================================
# Tests for Model Training (train_model)
# ============================================================================

def test_train_model_returns_correct_type(sample_model_data):
    """Test that train_model returns a RandomForestClassifier."""
    X_train, y_train, _, _ = sample_model_data
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "random_state": 42
    }
    model = train_model(X_train, y_train, params)
    
    assert isinstance(model, RandomForestClassifier)


def test_train_model_uses_provided_params(sample_model_data):
    """Test that train_model uses the provided hyperparameters."""
    X_train, y_train, _, _ = sample_model_data
    params = {
        "n_estimators": 50,
        "max_depth": 8,
        "random_state": 42
    }
    model = train_model(X_train, y_train, params)
    
    assert model.n_estimators == 50
    assert model.max_depth == 8
    assert model.random_state == 42


def test_train_model_fits_successfully(sample_model_data):
    """Test that train_model successfully fits the model."""
    X_train, y_train, _, _ = sample_model_data
    params = {
        "n_estimators": 10,
        "max_depth": 5,
        "random_state": 42
    }
    model = train_model(X_train, y_train, params)
    
    # Check that model has been fitted (has classes_ attribute)
    assert hasattr(model, 'classes_')
    assert len(model.classes_) == 2  # Binary classification


# ============================================================================
# Tests for Model Inference (inference)
# ============================================================================

def test_inference_returns_predictions(trained_model, sample_model_data):
    """Test that inference returns predictions of correct shape."""
    _, _, X_test, _ = sample_model_data
    predictions = inference(trained_model, X_test)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)
    assert all(pred in [0, 1] for pred in predictions)  # Binary predictions


def test_inference_predictions_are_valid(trained_model, sample_model_data):
    """Test that inference produces valid binary predictions."""
    _, _, X_test, _ = sample_model_data
    predictions = inference(trained_model, X_test)
    
    # All predictions should be either 0 or 1
    assert set(predictions).issubset({0, 1})


# ============================================================================
# Tests for Model Metrics (compute_model_metrics)
# ============================================================================

def test_compute_model_metrics_returns_correct_types():
    """Test that compute_model_metrics returns floats."""
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert isinstance(precision, (float, np.floating))
    assert isinstance(recall, (float, np.floating))
    assert isinstance(fbeta, (float, np.floating))


def test_compute_model_metrics_perfect_prediction():
    """Test compute_model_metrics with perfect predictions."""
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    # Perfect predictions should give scores of 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_compute_model_metrics_values_in_range():
    """Test that compute_model_metrics returns values between 0 and 1."""
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
