"""
Prediction Script for Zomato Delivery Time Model
"""

import pickle
import numpy as np
import pandas as pd

class DeliveryTimePredictor:
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl', features_path='feature_names.json'):
        """
        Initialize the predictor with saved model and scaler
        """
        import json

        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load feature names
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

    def predict(self, input_data):
        """
        Predict delivery time for input data

        Parameters:
        input_data: dict or pandas DataFrame with required features

        Returns:
        Predicted delivery time in minutes
        """
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Fill missing with default value

        # Reorder columns to match training data
        input_df = input_df[self.feature_names]

        # Scale the features
        input_scaled = self.scaler.transform(input_df)

        # Make prediction
        prediction = self.model.predict(input_scaled)

        return float(prediction[0])

# Example usage
if __name__ == "__main__":
    # Create predictor instance
    predictor = DeliveryTimePredictor(
        model_path='deployment/xgboost_model.pkl',
        scaler_path='deployment/scaler.pkl',
        features_path='deployment/feature_names.json'
    )

    # Example input (you need to adjust this based on your actual features)
    example_input = {
        "Distance_KM": 5.2,
        "Preparation_Time_Min": 15.0,
        "Order_Hour": 18,
        "Road_traffic_density_Encoded": 2,
        "Bad_Weather": 0,
        "Is_Rush_Hour": 1,
        "Is_Weekend": 0
        # Add all other features with appropriate values
    }

    # Make prediction
    predicted_time = predictor.predict(example_input)
    print(f"Predicted delivery time: {predicted_time:.2f} minutes")
