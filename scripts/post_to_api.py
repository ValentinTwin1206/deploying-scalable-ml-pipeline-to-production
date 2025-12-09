"""
Script to interact with the Census Income Prediction API.
Supports both GET and POST requests to the API endpoints.
"""

import argparse
import json
import requests
import sys


def get_root(api_url):
    """
    Send a GET request to the root endpoint.
    
    Args:
        api_url: The base URL of the API
        
    Returns:
        tuple: (status_code, response_json)
    """
    try:
        response = requests.get(api_url)
        status_code = response.status_code
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Failed to parse JSON response", "text": response.text}
        
        return status_code, result
        
    except requests.exceptions.RequestException as e:
        return None, {"error": str(e)}


def post_to_api(api_url):
    """
    Send a POST request to the API with sample census data.
    
    Args:
        api_url: The URL of the API endpoint
        
    Returns:
        tuple: (status_code, response_json)
    """
    
    # Sample data for prediction
    sample_data = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 25000,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "India"
    }
    
    try:
        # Send POST request
        response = requests.post(
            api_url,
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Get status code
        status_code = response.status_code
        
        # Parse response
        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"error": "Failed to parse JSON response", "text": response.text}
        
        return status_code, result
        
    except requests.exceptions.RequestException as e:
        return None, {"error": str(e)}


def main(env: str, method: str) -> int:
    """
    Main function to run API requests.
    
    Args:
        env: Environment ('dev' or 'prod')
        method: HTTP method ('get' or 'post')
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
      
    # dev
    if env == "dev":
        base_url = "http://localhost:8001"
    # prod
    else:  
        base_url = "https://deploying-scalable-ml-pipeline-to.onrender.com"
    
    print(f"Environment: {env.upper()}")
    print(f"Method: {method.upper()}")
    print(f"Base URL: {base_url}\n")
    
    # Execute request based on method
    if method == "get":
        print(f"Sending GET request to '{base_url}/'...")
        status_code, result = get_root(base_url)
    else:  # post
        predict_url = f"{base_url}/predict"
        print(f"Sending POST request to '{predict_url}'...")
        status_code, result = post_to_api(predict_url)
    
    # Display results
    print("=" * 50)
    print("API RESPONSE")
    print("=" * 50)
    print(f"Status Code: {status_code}")
    print(f"\nResult: {json.dumps(result, indent=2)}")
    print("=" * 50)
    
    # Interpret the prediction
    if status_code == 200:
        if method == "get" and "message" in result:
            print(f"\nWelcome Message: {result['message']}")
        elif method == "post" and "prediction" in result:
            prediction = result["prediction"]
            print(f"\nPrediction: {prediction}")
            if prediction == ">50K":
                print("The model predicts this person earns MORE than $50K/year")
            else:
                print("The model predicts this person earns LESS than or equal to $50K/year")
    elif status_code is None:
        print("\nError: Could not connect to the API")
        return 1
    else:
        print(f"\nError: Request failed with status code {status_code}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interact with Census Income Prediction API (GET and POST requests)"
    )
    parser.add_argument(
        "env",
        choices=["dev", "prod"],
        help="Environment to use: 'dev' for localhost, 'prod' for production Render deployment"
    )
    parser.add_argument(
        "method",
        choices=["get", "post"],
        help="HTTP method: 'get' for root endpoint, 'post' for prediction endpoint"
    )
    args = parser.parse_args()

    sys.exit(main(args.env, args.method))
