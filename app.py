import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# CONFIG
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.h5")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
ALLOWED_EXT = {'.jpg', '.jpeg', '.png'}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

# Lazy-load model
_model = None
def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

IMG_SIZE = (128, 128)
def preprocess_image(file_path):
    img = Image.open(file_path).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype='float32') / 255.0
    return arr

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        f = request.files['image']
        filename = secure_filename(f.filename)
        if filename == '':
            return jsonify({"error": "Invalid filename"}), 400

        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"Unsupported file extension '{ext}'"}), 400

        tmp_path = os.path.join(UPLOAD_DIR, filename)
        f.save(tmp_path)

        img_arr = preprocess_image(tmp_path)
        model = load_model()
        pred = model.predict(np.expand_dims(img_arr, axis=0))[0]
        # handle shape: could be scalar or array
        prob = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
