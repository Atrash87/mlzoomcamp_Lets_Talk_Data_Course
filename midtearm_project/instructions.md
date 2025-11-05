## Clone or Extract the Project
git clone https://github.com/Atrash8/midtearm_project.git
cd air_quality_project

## Install Dependencies (if running locally without Docker)
python -m venv venv
source venv/bin/activate    # on macOS/Linux
venv\Scripts\activate       # on Windows

pip install -r requirements.txt


## Train the Model (optional)
python train.py

## Run the Web Service (FastAPI) 
## A Directly (without Docker)
uvicorn app:app --reload



## B Using Docker
Build the Docker
image:docker build -t air_quality_api.

Run the container:
docker run -d -p 8000:8000 air_quality_api


## Example prediction:
import requests

data = {
    "hour": 10,
    "day_of_week": 2,
    "month": 5,
    "CO_roll3": 1.1
}

res = requests.post("http://localhost:8000/predict", json=data)
print(res.json())


