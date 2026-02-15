from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os

app = Flask(__name__)

# ============================
# LOAD MANGO MODEL
# ============================

MODEL_PATH = "mango_model.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("mango_model.keras not found")

print("🔄 Loading Mango model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Mango model loaded")

# ============================
# CLASS LABELS (EXACT ORDER)
# ============================

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

    if humidity > 80:
        risks.append("Fungal disease risk due to high humidity")

    if moisture > 75:
        risks.append("Root disease risk due to high soil moisture")

    if temp > 30:
        risks.append("Heat stress possible")

    if len(risks) == 0:
        risks.append("No major disease-favorable conditions detected")

    return risks

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

    pred = model.predict(img_array)
    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])
    label = CLASS_NAMES[idx]

    confidence = round(confidence * 100, 2)

    risk = analyze_risk(
        latest_sensor["temperature"],
        latest_sensor["humidity"],
        latest_sensor["moisture"]
    )

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor,
        "risk": risk
    })

# ============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
