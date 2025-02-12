from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
import requests
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# URL to download the model if not present
def download_model():
    MODEL_URL = 'https://drive.google.com/uc?export=download&id=11tKJ2cNOPuVX1K4eqSBujYWN0lzt_qJJ'
    MODEL_PATH = 'model/yolo11_model.pt'
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv11 model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # Download in chunks
        response = requests.get(MODEL_URL, stream=True)
        if 'html' in response.headers.get('Content-Type', ''):
            raise ValueError("Error: The URL is returning an HTML page, not the model file.")

        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Model downloaded successfully.")

# Ensure the model is downloaded before loading
download_model()

# Load the YOLOv11 model
model = YOLO(MODEL_PATH)

# Preprocess image to match COCO dataset format
def preprocess_image(image):
    image = cv2.resize(image, (640, 640))  # Resize to YOLOv11 expected size
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Prediction route for file uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the image
    image = preprocess_image(image)

    # Perform inference
    results = model(image)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': int(box.cls),
                'confidence': float(box.conf),
                'box': box.xyxy.tolist()
            })

    return jsonify({'detections': detections})

# New route for Base64-encoded images
@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Extract Base64 image
    image_data = data['image'].split(',')[1]  # Remove data:image/png;base64,
    image_bytes = base64.b64decode(image_data)
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the image
    image = preprocess_image(image)

    # Perform inference
    results = model(image)

    # Process results
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': int(box.cls),
                'confidence': float(box.conf),
                'box': box.xyxy.tolist()
            })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
