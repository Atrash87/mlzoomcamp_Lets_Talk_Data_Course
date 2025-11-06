# Air Quality Prediction Project

A machine learning project to predict CO concentration using the [Air Quality UCI dataset](https://archive.ics.uci.edu/dataset/360/air+quality).  
It includes data cleaning, feature engineering, model training, and deployment with FastAPI and Docker.

## Dataset Information
The dataset contains 9,358 hourly averaged responses from 5 metal oxide chemical sensors deployed in a polluted urban area in Italy (March 2004 – February 2005).  
It includes CO, Non-Methanic Hydrocarbons, Benzene, NOx, and NO2 concentrations.  
Missing values are marked as `-200`.  
Use is restricted to **research purposes only**.

## Project Structure

air_quality_project/
├── notebook.ipynb
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── feature_columns.pkl
├── air_quality_model.pkl
├── deployment_screenshot.png
└── README.md

## Problem Description
Air pollution poses a serious health and environmental threat in many urban areas.  
This project focuses on analyzing and modeling air quality data to understand the variation of pollutant levels—especially carbon monoxide (CO)—based on time and environmental sensor readings.

## Objective
The main goal is to build a predictive model that can estimate **carbon monoxide (CO) concentration** using sensor data and temporal features.  
Accurate predictions can help improve environmental monitoring systems, enable early warnings for high pollution levels, and support better decision-making for air quality management.



## Setup Instructions

### 1. Clone or Extract the Project
```bash
git clone https://github.com/Atrash87/midtearm_project.git
cd air_quality_project
```
## 2. Install Dependencies

```python
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```
## 3. Train the Model (Optional)
```python
python train.py
```
## This will create:

    air_quality_model.pkl

    feature_columns.pkl

## Run the Web App
## Option A: 
## Run Directly

uvicorn app:app --reload

Then open in your browser:

http://127.0.0.1:8000

http://127.0.0.1:8000/docs

**Example input:**

{
  "hour": 10,
  "day_of_week": 2,
  "month": 5,
  "CO_roll3": 1.1
}

## Option B: 
## Run with Docker via pyhton

```bash
docker build -t air_quality_api .
docker run -d -p 8000:8000 air_quality_api
```
**Then open:**

http://localhost:8000

http://localhost:8000/docs

**Example Input for Prediction (via Python)**
```python
import requests

data = {
    "hour": 10,
    "day_of_week": 2,
    "month": 5,
    "CO_roll3": 1.1
}

res = requests.post("http://localhost:8000/predict", json=data)
print(res.json())
```
**This sends a test request to the running API and returns the predicted CO concentration.**


**Notes**
    - notebook.ipynb includes EDA, feature analysis, and model tuning.      - train.py trains and saves the final model.      - app.py serves predictions through FastAPI.
