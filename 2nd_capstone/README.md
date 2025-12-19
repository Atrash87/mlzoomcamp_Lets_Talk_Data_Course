# Zomato Delivery Time Prediction

## Problem Description
Accurate delivery time prediction is crucial for food delivery platforms like Zomato. It impacts customer satisfaction, operational efficiency, and resource allocation. Current systems often use simple estimates, but these can be inaccurate due to factors like traffic, weather, distance, and restaurant preparation time.

## Objective
Build a machine learning model to predict food delivery time based on various factors including distance, traffic conditions, weather, time of day, and restaurant characteristics.

## Dataset
The dataset contains delivery information from Zomato including:

- **Size:** 11+ sample rows (full dataset typically contains thousands of rows)  
- **Features:** 20+ columns including GPS coordinates, timestamps, weather conditions, traffic density, vehicle type, and delivery time  
- **Target Variable:** `Time_taken` (min)  

### Key Features
- Delivery person demographics (age, ratings)  
- Restaurant and delivery location coordinates  
- Order timing information  
- Weather conditions  
- Road traffic density  
- Vehicle condition and type  
- Order type and multiple deliveries  
- City type and festival periods  

## Project Structure
```bash
2nd_capstone/
├── data/
│ ├── zomato_delivery_data.csv # Original dataset
│ ├── zomato_delivery_prepared.csv # Data after Phase 1 cleaning
│ └── zomato_delivery_modeling_ready.csv # Data after Phase 2 feature engineering
├── eda_plots/ # Exploratory Data Analysis visualizations
├── model_results/ # Model comparison results and plots
├── deployment/ # Production-ready model artifacts
│ ├── best_model.pkl # Trained model
│ ├── scaler.pkl # Fitted scaler
│ ├── feature_names.json # Feature names
│ ├── model_metadata.json # Model metadata
│ ├── predict.py # Prediction script
│ └── requirements.txt # Python dependencies
├── notebooks/ # Jupyter notebooks (optional)
├── src/ # Source code
│ ├── phase1_data_preparation.py # Phase 1: Data cleaning
│ ├── phase2_feature_engineering.py # Phase 2: Feature engineering
│ └── phase3_model_training.py # Phase 3: Model training
├── app.py # Flask API for deployment
├── Dockerfile # Docker configuration
├── requirements.txt # Main requirements
└── README.md # This file
```

##  Machine Learning Pipeline
The project follows a comprehensive 3-phase approach:

## Phase 1: Data Preparation
- Cleaning date/time columns with inconsistent formats  
- Calculating Haversine distance between locations  
- Handling missing values  
- Encoding categorical variables  

##  Phase 2: Feature Engineering & EDA
- Advanced feature creation (speed metrics, interaction terms)  
- Temporal feature engineering (cyclic encoding, time categories)  
- Spatial feature extraction  
- Exploratory Data Analysis with visualizations  
- Feature transformation and selection  

##  Phase 3: Model Training & Deployment
- Training 9 different algorithms  
- Hyperparameter tuning for best model  
- Feature importance analysis  
- Flask API development  
- Docker containerization  

##  Models Evaluated
- Linear Regression (Baseline)  
- Ridge Regression  
- Lasso Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- LightGBM Regressor  
- K-Neighbors Regressor  

##  Setup and Installation

##  Prerequisites
- Python 3.8 or higher  
- pip package manager  

##  Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/zomato-delivery-prediction.git
cd zomato-delivery-prediction
```
## Install requirements
pip install -r requirements.txt
Running the Pipeline
## Run Phase 1: Data Preparation
python src/phase1_data_preparation.py

## Run Phase 2: Feature Engineering
python src/phase2_feature_engineering.py

## Run Phase 3: Model Training
python src/phase3_model_training.py

##  Running the API
## Start Flask API
python app.py

## The API will be available at http://localhost:5000

**API Endpoints**

- GET / - Home page with API information

- GET /health - Health check endpoint

- GET /features - List of required features

- POST /predict - Predict delivery time

**Example API Request**
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Distance_KM": 5.2,
    "Preparation_Time_Min": 15.0,
    "Order_Hour": 18,
    "Road_traffic_density_Encoded": 2,
    "Bad_Weather": 0,
    "Is_Rush_Hour": 1,
    "Is_Weekend": 0
  }'

##  Docker Deployment
## Build Docker image
docker build -t zomato-delivery-predictor .

## Run container
docker run -p 5000:5000 zomato-delivery-predictor
