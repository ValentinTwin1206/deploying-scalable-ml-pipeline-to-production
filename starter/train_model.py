# Script to train machine learning model.

# standard libs
import pathlib
import pickle
import sys

# third party libs
import pandas as pd

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


def _load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean census data by removing leading/trailing spaces from column names and values,
    and removing duplicate rows.
    
    The raw census.csv file has spaces in column names and string values that need
    to be cleaned for proper data processing. Also removes duplicate records.
    
    Args:
        filepath: Path to the census.csv file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for processing
        
    Example:
        >>> df = load_and_clean_census_data('data/census.csv')
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Clean column names - remove leading and trailing spaces
    df.columns = df.columns.str.strip()
    
    # Clean all string/object columns - remove leading and trailing spaces
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    return df


def compute_performance_on_slices(
    data: pd.DataFrame,
    categorical_features: list,
    label: str,
    model,
    encoder,
    lb
) -> pd.DataFrame:
    """
    Compute model performance on slices of categorical features.
    
    For each categorical feature, computes precision, recall, and F-beta score
    for each unique value in that feature.
    
    Args:
        data: DataFrame containing the test data
        categorical_features: List of categorical feature names
        label: Name of the label column
        model: Trained model
        encoder: Fitted OneHotEncoder
        lb: Fitted LabelBinarizer
        
    Returns:
        pd.DataFrame: DataFrame with columns [feature, value, precision, recall, fbeta, n_samples]
    """
    results = []
    
    for feature in categorical_features:
        # Get unique values for this feature
        unique_values = data[feature].unique()
        
        for value in unique_values:
            # Create a slice of data for this feature value
            slice_data = data[data[feature] == value]
            
            # Skip if slice is too small
            if len(slice_data) < 2:
                continue
            
            # Process the slice
            X_slice, y_slice, _, _ = process_data(
                X=slice_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb
            )
            
            # Make predictions
            preds = inference(model, X_slice)
            
            # Compute metrics
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            
            # Store results
            results.append({
                'feature': feature,
                'value': value,
                'precision': precision,
                'recall': recall,
                'fbeta': fbeta,
                'n_samples': len(slice_data)
            })
    
    return pd.DataFrame(results)


def main() -> int:
    """ Main function to train the model and save artifacts."""

    try:
        # Define paths
        data_path    = pathlib.Path(__file__).parent.parent / "data" / "census.csv"
        output_dir   = pathlib.Path(__file__).parent.parent / "model"
        slice_output_path = output_dir / "slice_output.txt"

        # Define vars
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
        model_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        }

        # clean data
        print("Loading and cleaning data...")
        cleaned_data = _load_and_clean_data(data_path)
        
        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        print("Splitting data into train and test sets...")
        train_data, test_data = train_test_split(cleaned_data, test_size=0.20)

    
        # get 'train' dataset
        print("Processing training data...")
        X_train, y_train, encoder, lb = process_data(
            X=train_data,
            categorical_features=cat_features,
            label="salary",
            training=True
        )

        # get 'test' dataset
        print("Processing test data...")
        X_test, y_test, _, _ = process_data(
            X=test_data, 
            categorical_features=cat_features, 
            label="salary", 
            training=False,
            encoder=encoder,
            lb=lb
        )

        # Train and save the model
        print("Training model...")
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            params=model_params
        )
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        preds = inference(
            model=model,
            X=X_test
        )

        print("Computing overall model performance...")
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        
        print(f"Model Performance:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F-beta: {fbeta:.4f}")
        
        # Compute performance on slices
        print("\nComputing performance on categorical feature slices...")
        slice_performance = compute_performance_on_slices(
            data=test_data,
            categorical_features=cat_features,
            label="salary",
            model=model,
            encoder=encoder,
            lb=lb
        )
        
        # Save slice performance to file
        with open(slice_output_path, "w") as f:
            f.write("Model Performance on Categorical Feature Slices\n")
            f.write("=" * 80 + "\n\n")
            
            for feature in cat_features:
                feature_results = slice_performance[slice_performance['feature'] == feature]
                if len(feature_results) > 0:
                    f.write(f"\nFeature: {feature}\n")
                    f.write("-" * 80 + "\n")
                    for _, row in feature_results.iterrows():
                        f.write(f"  Value: {row['value']:<30} | "
                               f"Precision: {row['precision']:.4f} | "
                               f"Recall: {row['recall']:.4f} | "
                               f"F-beta: {row['fbeta']:.4f} | "
                               f"Samples: {row['n_samples']}\n")
        
        print(f"Slice performance saved to {slice_output_path}")
        
        # Save model artifacts
        model_dir = pathlib.Path(__file__).parent.parent / "model"
        model_dir.mkdir(exist_ok=True)
        
        print(f"Saving model artifacts to {model_dir}...")
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open(model_dir / "encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        
        with open(model_dir / "lb.pkl", "wb") as f:
            pickle.dump(lb, f)
    
        print("Script successfully finished.")
        return 0
    
    except Exception as exp:
        print("[ERRO] An error occurred during model training")
        print(exp)
        return 1

# entry point
if __name__ == "__main__":
    sys.exit(main())