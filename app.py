from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# ============================
# CLASS LABELS
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
# GLOBAL TFLITE INTERPRETER
# ============================

interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        print("🔄 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path="mango_model.tflite")
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded")
    return interpreter

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
# PRECAUTIONS
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
    return "Mango Backend Running (TFLite)"

@app.route("/sensor", methods=["POST"])
def receive_sensor():
    latest_sensor["temperature"] = float(request.form.get("temperature"))
    latest_sensor["humidity"] = float(request.form.get("humidity"))
    latest_sensor["moisture"] = float(request.form.get("moisture"))
    return jsonify({"status": "sensor data received"})

@app.route("/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if not image_file:
        return jsonify({"error": "image missing"}), 400

    img = image.load_img(BytesIO(image_file.read()), target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    idx = int(np.argmax(pred))
    confidence = float(pred[0][idx])
    label = CLASS_NAMES[idx]

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
