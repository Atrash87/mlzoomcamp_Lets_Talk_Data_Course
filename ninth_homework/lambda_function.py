import json
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort


# -----------------------------
# Image downloading
# -----------------------------
def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


# -----------------------------
# Preprocessing
# -----------------------------
def prepare_image(img: Image.Image, target_size=(200, 200)) -> np.ndarray:
    # Ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    img = img.resize(target_size)

    # Convert to numpy (0-1)
    img_np = np.array(img).astype("float32") / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    img_np = (img_np - mean) / std

    # HWC → CHW
    img_np = np.transpose(img_np, (2, 0, 1))

    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)

    return img_np


# -----------------------------
# Load ONNX model ONCE
# -----------------------------
session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name


# -----------------------------
# AWS Lambda handler
# -----------------------------
def lambda_handler(event, context):
    """
    event = { "url": "https://some-image.jpg" }
    """
    url = event.get("url")
    if url is None:
        return {"error": "Missing 'url' in request"}

    # Download → preprocess → run model
    img = download_image(url)
    tensor = prepare_image(img)

    output = session.run(None, {input_name: tensor})[0]
    prediction = float(output[0][0])

    return {
        "prediction": prediction
    }
