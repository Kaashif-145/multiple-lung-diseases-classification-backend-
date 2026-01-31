# app.py
# Clean Flask backend for lung X-ray prediction using a .keras model

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import traceback

# --------------------------
# Config
# --------------------------
# Correct path — FIXED
MODEL_PATH = r"D:\MAJOR PROJECT\lung_outputs\best_model.keras"

UPLOAD_FOLDER = r"D:\MAJOR PROJECT\uploads"
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

IMG_SIZE = (224, 224)

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------
# Helper functions
# --------------------------
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


def preprocess_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# --------------------------
# Load model
# --------------------------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    print("Model loaded successfully:", MODEL_PATH)

except Exception as e:
    print("[ERROR] Cannot load model:")
    traceback.print_exc()
    model = None


# --------------------------
# CLASS LABELS
# --------------------------
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis", "COVID"]


# --------------------------
# Routes
# --------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running" if model else "model_load_failed",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model failed to load"}), 500

        if "file" not in request.files:
            return jsonify({"error": "Send file using key 'file'"}), 400

        file = request.files["file"]

        filename = secure_filename(file.filename)
        if filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save uploaded file
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Preprocess image
        arr = preprocess_image(save_path)

        # Predict
        preds = model.predict(arr)[0]
        index = int(np.argmax(preds))
        label = CLASS_NAMES[index]

        return jsonify({
            "prediction": label,
            "class_index": index,
            "probabilities": {
                CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
            }
        })

    except Exception as e:
        print("[ERROR] Predict error:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# --------------------------
# Run server
# --------------------------
if __name__ == "__main__":
    print("🚀 Flask server starting on port 5000")
    print("Model path:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=False)
