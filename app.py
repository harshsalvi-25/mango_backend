from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os

app = Flask(__name__)

# ============================
# CONFIG
# ============================

MODEL_PATH = "mango_model.keras"

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# ============================
# LAZY LOAD MODEL
# ============================

model = None

def get_model():
    global model
    if model is None:
        print("🔄 Loading Mango model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Mango model loaded")
    return model

# ============================
# SENSOR STORAGE
# ============================

latest_sensor = {
    "temperature": None,
    "humidity": None,
    "moisture": None
}

# ============================
# SENSOR RISK LOGIC
# ============================

def analyze_risk(temp, humidity, moisture):

    risks = []

    if humidity is None:
        return ["Sensor data not available"]

    if humidity > 80 and temp > 25:
        risks.append("Favorable conditions for Anthracnose")

    if humidity > 70 and temp < 30:
        risks.append("Possible Powdery Mildew risk")

    if humidity > 75 and moisture > 70:
        risks.append("Possible Die Back risk")

    if temp > 28 and humidity > 70:
        risks.append("Possible Gall Midge infestation")

    if len(risks) == 0:
        risks.append("No major disease-favorable conditions detected")

    return risks

# ============================
# PRECAUTIONARY MEASURES
# ============================

PRECAUTIONS = {
    "Anthracnose": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Apply recommended fungicide"
    ],
    "Powdery Mildew": [
        "Improve air circulation",
        "Avoid excess nitrogen fertilizer",
        "Apply sulfur fungicide"
    ],
    "Die Back": [
        "Prune infected branches",
        "Avoid waterlogging",
        "Apply fungicide"
    ],
    "Gall Midge": [
        "Remove affected shoots",
        "Apply recommended insecticide"
    ],
    "Healthy": [
        "Crop is healthy",
        "Maintain proper irrigation",
        "Regular monitoring recommended"
    ]
}

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Mango Backend Running"

# ---------- SENSOR DATA ----------
@app.route("/sensor", methods=["POST"])
def receive_sensor():

    latest_sensor["temperature"] = float(request.form.get("temperature"))
    latest_sensor["humidity"] = float(request.form.get("humidity"))
    latest_sensor["moisture"] = float(request.form.get("moisture"))

    return jsonify({"status": "sensor data received"})

# ---------- IMAGE PREDICTION ----------
@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "image missing"}), 400

    img = image.load_img(BytesIO(image_file.read()), target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    m = get_model()
    pred = m.predict(img_array)

    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])
    label = CLASS_NAMES[idx]

    # Healthy safeguard
    if confidence < 0.60:
        label = "Healthy"

    confidence = round(confidence * 100, 2)

    risk = analyze_risk(
        latest_sensor["temperature"],
        latest_sensor["humidity"],
        latest_sensor["moisture"]
    )

    precautions = PRECAUTIONS.get(label, [])

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor,
        "risk": risk,
        "precautions": precautions
    })

# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
