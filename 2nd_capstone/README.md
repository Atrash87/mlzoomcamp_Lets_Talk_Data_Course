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

## Key Results

### Model Performance Summary

#### Top 3 Performing Models
1. **XGBoost** – Test MAE: 3.33 minutes (Best)  
   - R²: 0.9097 (Excellent)  
   - Training time: 2.49 seconds  
   - Minor overfitting but best overall  

2. **Gradient Boosting** – Test MAE: 4.72 minutes  
   - R²: 0.8273 (Very Good)  
   - Training time: 0.12 seconds  
   - Good balance of speed and accuracy  

3. **Random Forest** – Test MAE: 6.54 minutes  
   - R²: 0.7599 (Good)  
   - Training time: 0.12 seconds  
   - Most stable (least overfitting)  

#### Worst Performing Models
- **Linear Regression** – Test MAE: 18.57 minutes (worst)  
- **LightGBM** – Test MAE: 14.29 minutes (surprisingly poor)  
- **Ridge Regression** – Test MAE: 12.79 minutes  

---

### Critical Observations

#### Overfitting Issue
- All models show overfitting (negative MAE difference)  
- Most severe: Linear Regression (-18.57 MAE diff)  
- Least severe: Random Forest (-4.78 MAE diff)  

#### Model Selection
- **XGBoost** selected as best despite overfitting  
- Test R² of 0.9097 indicates excellent predictive power  
- 3.33 minutes MAE is acceptable for delivery predictions  

#### Hyperparameter Tuning Results
- Tuning attempted but default parameters performed better  
- Best found parameters: `subsample=0.8`, `n_estimators=50`, `max_depth=3`, `learning_rate=0.1`, `colsample_bytree=0.9`  
- Test MAE increased from 3.33 to 3.89 minutes after tuning  
- Conclusion: XGBoost default settings optimal for this dataset  

---

### Feature Importance Analysis
**Top 5 Most Important Features (XGBoost):**
1. `Delivery_person_Age` – 84.95% importance (Dominant predictor)  
2. `Travel_Time_Min` – 9.51% importance  
3. `Travel_Time_Min_scaled` – 2.97% importance  
4. `Distance_Traffic_Interaction` – 1.07% importance  
5. `Prep_Efficiency` – 0.55% importance  

**Surprising Finding:**  
- `Delivery_person_Age` is overwhelmingly important (85% of importance)  
- Suggests age correlates strongly with delivery efficiency, possibly indicating experience or physical capability  

---

### Deployment Ready

**Artifacts Created:**
- Model file: `xgboost_model.pkl`  
- Scaler: `scaler.pkl`  
- Feature names: `feature_names.json`  
- Metadata: `model_metadata.json`  
- Prediction script: `predict.py`  
- Flask API: `app.py`  
- Docker configuration: `Dockerfile`  

**Performance Interpretation:**
- MAE: 3.33 minutes → Predictions within ±3.33 minutes on average  
- Error rate: 12.3% of average delivery time (27 minutes)  
- R²: 0.9097 → Model explains 91% of delivery time variance  
- Business impact: Could significantly reduce customer complaints  

---

### Limitations & Caveats
- Severe overfitting across all models  
- Feature dominance by `Delivery_person_Age` (potential bias)  
- Lack of validation on unseen data  

---

### Recommendations for Production
- Monitor feature drift (especially `Delivery_person_Age`)  
- Implement ensemble of top 3 models  
- Add real-time features (live traffic, weather updates)  
- A/B test different models in production  


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
