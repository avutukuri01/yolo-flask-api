from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the YOLOv11 model
model = YOLO('model/yolo11_model.pt')  # Replace with your actual model path

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
