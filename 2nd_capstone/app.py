"""
Flask API for Zomato Delivery Time Prediction
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

# Global variables for model, scaler, and features
model = None
scaler = None
feature_names = None

def load_artifacts():
    """Load model artifacts"""
    global model, scaler, feature_names

    try:
        # Load model
        with open('deployment/best_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open('deployment/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load feature names
        with open('deployment/feature_names.json', 'r') as f:
            feature_names = json.load(f)

        print("Model artifacts loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        return False

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Zomato Delivery Time Prediction API',
        'status': 'active',
        'endpoints': {
            '/predict': 'POST - Predict delivery time',
            '/health': 'GET - API health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                # Provide default values for missing features
                if 'Distance' in feature:
                    input_df[feature] = 5.0  # Default distance
                elif 'Time' in feature:
                    input_df[feature] = 15.0  # Default time
                elif 'Encoded' in feature or 'Is_' in feature:
                    input_df[feature] = 0  # Default for encoded/binary
                else:
                    input_df[feature] = 0  # Default for others

        # Reorder columns to match training
        input_df = input_df[feature_names]

        # Scale the features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Prepare response
        response = {
            'predicted_delivery_time_minutes': float(prediction[0]),
            'prediction_timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Get list of required features"""
    return jsonify({
        'required_features': feature_names,
        'feature_count': len(feature_names)
    })

if __name__ == '__main__':
    # Load model artifacts
    if load_artifacts():
        print("Starting Flask API...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model artifacts. Exiting.")
