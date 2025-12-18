# Beijing PM2.5 Air Quality Forecasting
## Problem Description

Air pollution, particularly fine particulate matter (PM2.5), poses significant public health risks. Predicting PM2.5 concentration is crucial for issuing timely public advisories and understanding pollution dynamics. This project tackles the challenge of forecasting Beijing's PM2.5 levels by building a machine learning model that predicts the PM2.5 concentration for the next hour based on current and historical meteorological and pollution data.

## Objective

The primary objective of this project is to develop and deploy a reliable machine learning model for hourly PM2.5 concentration forecasting. The project follows a complete end-to-end pipeline:

    - Conduct thorough Exploratory Data Analysis (EDA) to understand data patterns and relationships.

    - Perform extensive feature engineering to create predictive features from the time-series data.

    - Train, compare, and tune multiple regression models to identify the best performer.

    - Containerize the final model into a scalable API using Docker for easy deployment and integration.

## Dataset Information

This project uses the Beijing PM2.5 Data Set from the UCI Machine Learning Repository.

    - Source: UCI Machine Learning Repository (ID: 381)

    - Time Period: January 1st, 2010, to December 31st, 2014.

    - Frequency: Hourly measurements.

    - Instances: 43,824

    - Features: 13

    Target Variable: pm2.5 (PM2.5 concentration in µg/m³)

  **Key Features:** 

       - DEWP: Dew Point (°C)

        - TEMP: Temperature (°C)

        - PRES: Pressure (hPa)

        - cbwd: Combined wind direction (Categorical)

        - Iws: Cumulated wind speed (m/s)

        - Is: Cumulated hours of snow

        - Ir: Cumulated hours of rain

    License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Project Structure

bash
pm25-forecasting/
├── notebooks/               # Jupyter notebooks for analysis and modeling
│   ├── 01_Cleaning_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling_Tuning.ipynb
├── app/                     # FastAPI application for model serving
│   ├── api.py              # Main API application
│   ├── model.py            # Model loading and prediction logic
│   └── schemas.py          # Pydantic models for request/response
├── models/                  # Serialized model artifacts
│   ├── best_pm25_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── tests/                   # Unit tests for the API
├── Dockerfile              # Instructions to build the Docker image
├── docker-compose.yml      # Orchestration for multi-container setup
├── requirements.txt        # Python dependencies
└── README.md               # This file

Setup and Installation Instructions
Prerequisites

    Python 3.9+

    Docker and Docker Compose (for containerized deployment)

## Local Development Setup

  **Clone the repository:**
    bash

git clone <repository-url>
cd pm25-forecasting

**Create and activate a virtual environment:**
bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

**Install the required dependencies:**
bash

pip install -r requirements.txt

Running the API with Docker (Recommended)

The easiest way to run the application is using Docker Compose, which handles all dependencies.

    Ensure you have Docker and Docker Compose installed.

  **From the project root directory, build and start the service:**
    bash

docker-compose up --build

    The API will be available at http://localhost:8000.

    Access the interactive API documentation at http://localhost:8000/docs.

**Using the API**

Once the service is running, you can make predictions by sending a POST request to the /predict endpoint.

**Example using curl:**
bash

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
    "DEWP": -21.0,
    "TEMP": -11.0,
    "PRES": 1021.0,
    "Iws": 1.79,
    "Is": 0,
    "Ir": 0,
    "hour": 0,
    "month": 1,
    "pm2.5_lag_1h": 15.0,
    "pm2.5_roll_mean_6h": 18.5
  }
}'

The response will include the predicted PM2.5 concentration for the next hour and an associated Air Quality Index (AQI) category.
Model Overview

**The modeling process involved:**

    - Feature Engineering: Creating lag features (1h, 3h, 6h, 12h, 24h), rolling statistics (mean, std, min, max), and cyclical time features.

    - Model Comparison: Linear Regression, Random Forest, XGBoost, and LightGBM were evaluated using Time-Series Cross-Validation.

    - Hyperparameter Tuning: RandomizedSearchCV was used to optimize the top-performing models.

    - Final Model: A tuned XGBoost Regressor was selected as the final model for deployment based on its performance in R² score and Mean Absolute Error (MAE).

Citation:

If you use this dataset or code in your work, please cite the original data source:

    Chen, S. (2015). Beijing PM2.5 [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49
