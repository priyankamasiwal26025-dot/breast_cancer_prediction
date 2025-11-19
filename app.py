"""
Flask app for Breast Cancer Prediction.

Endpoints:
- GET  /           -> serves frontend/index.html
- POST /predict    -> accepts form-data 'image' and returns JSON {label, malignant_probability}
- GET  /sample     -> serves the local sample image path (helpful if hosting environment rewires the local path)
"""

import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# CONFIG
MODEL_PATH = os.environ.get("MODEL_PATH", "model.h5")   # trained Keras model (h5 or tf)
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}
SAMPLE_IMAGE_LOCAL_PATH = "/mnt/data/a1ac1f0b-a6fb-4aa3-819f-b2cd2747bde5.png"  # user-provided local path

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__, static_folder="frontend", static_url_path="")

# Lazy-load model
_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train or upload model.h5")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

# Basic preprocessing - match the train.py image size and scaling
IMG_SIZE = (128, 128)
def preprocess_image(path):
    img = Image.open(path).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    return arr

# Serve frontend index
@app.route("/", methods=["GET"])
def index():
    return send_from_directory('frontend', 'index.html')

# Optional route to return the local sample image directly (useful for testing)
@app.route("/sample", methods=["GET"])
def sample():
    if os.path.exists(SAMPLE_IMAGE_LOCAL_PATH):
        return send_file(SAMPLE_IMAGE_LOCAL_PATH, mimetype='image/png')
    return jsonify({"error": "sample image not found on server"}), 404

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error":"no image provided"}), 400

    f = request.files['image']
    filename = secure_filename(f.filename)
    if filename == '':
        return jsonify({"error":"invalid filename"}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"unsupported file extension '{ext}'"}), 400

    tmp_path = os.path.join(UPLOAD_DIR, filename)
    f.save(tmp_path)

    try:
        img_arr = preprocess_image(tmp_path)
        model = load_model()
        pred = model.predict(np.expand_dims(img_arr, axis=0))[0][0]
        prob = float(pred)           # probability of malignant (assumes sigmoid output)
        label = "malignant" if prob >= 0.5 else "benign"
        return jsonify({"label": label, "malignant_probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Start locally: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
