import json
import base64
import onnxruntime as ort
from io import BytesIO
from PIL import Image
import numpy as np

# --------------------
# Preprocessing
# --------------------

def prepare_image(img, target_size=(224, 224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
    img = prepare_image(img)

    # Convert to numpy array
    x = np.array(img).astype("float32") / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    # Move channels
    x = np.transpose(x, (2, 0, 1))

    # Add batch dim
    x = np.expand_dims(x, 0)

    return x.astype("float32")


# --------------------
# Model
# --------------------

# The model files EXIST ONLY inside the Docker container
session = ort.InferenceSession("hair_classifier_empty.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


# --------------------
# Lambda handler
# --------------------

def lambda_handler(event, context=None):
    """
    Expects JSON:
      { "image_data": "<base64 string>" }
    """
    image_b64 = event["image_data"]
    image_bytes = base64.b64decode(image_b64)
    img = Image.open(BytesIO(image_bytes))

    x = preprocess(img)

    pred = session.run([output_name], {input_name: x})[0]

    # Convert numpy to Python float
    value = float(pred[0][0])

    return {
        "prediction": value
    }


# --------------------
# Local test
# --------------------

if __name__ == "__main__":
    # Load a test image
    with open("yf.jpeg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    result = lambda_handler({"image_data": b64})
    print("Prediction:", result)
