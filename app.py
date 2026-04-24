from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
import os

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
# TFLITE MODEL
# ============================

interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        print("🔄 Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path="mango_model.tflite")
        interpreter.allocate_tensors()
        print("✅ Model loaded")
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
# RISK LOGIC
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

    if not risks:
        risks.append("No major disease-favorable conditions detected")

    return risks

# ============================
# PRECAUTIONS (CLEAN)
# ============================

PRECAUTIONS = {
    "Anthracnose": [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Maintain spacing",
        "Ensure air circulation",
        "Monitor crop regularly"
    ],
    "Powdery Mildew": [
        "Improve air circulation",
        "Avoid high humidity",
        "Maintain spacing",
        "Monitor crop regularly"
    ],
    "Die Back": [
        "Prune infected branches",
        "Avoid waterlogging",
        "Maintain plant health",
        "Monitor crop regularly"
    ],
    "Gall Midge": [
        "Remove affected shoots",
        "Maintain hygiene",
        "Monitor crop regularly"
    ],
    "Bacterial Canker": [
        "Use healthy planting material",
        "Remove infected parts",
        "Maintain orchard hygiene",
        "Monitor crop regularly"
    ],
    "Cutting Weevil": [
        "Remove damaged parts",
        "Maintain cleanliness",
        "Monitor crop regularly"
    ],
    "Sooty Mould": [
        "Control insect infestation",
        "Clean leaves",
        "Improve air circulation",
        "Monitor crop regularly"
    ],
    "Healthy": [
        "Crop is healthy",
        "Maintain irrigation",
        "Ensure nutrition",
        "Regular monitoring"
    ]
}

# ============================
# TREATMENT (WITH IMAGES)
# ============================

TREATMENT = {

    "Anthracnose": {
        "pesticide": [
            {
                "name": "Mancozeb Fungicide (Mancozeb 75% WP)",
                "image": "https://5.imimg.com/data5/SELLER/Default/2021/7/QW/GH/SA/6616513/mancozeb-75-wp-contact-fungicide.jpg"
            }
        ],
        "fertilizer": [
            {
                "name": "Micronutrient Fertilizer Spray (Zn + B mixture)",
                "image": "https://ariesagro.com/wp-content/uploads/2022/11/zincbor.png"
            }
        ]
    },

    "Powdery Mildew": {
        "pesticide": [
            {
                "name": "Sulfur Fungicide (Sulfur 80% WP)",
                "image": "https://mankindag.com/wp-content/uploads/2024/08/IMG_0110_Mankind-Sulfur.png"
            }
        ],
        "fertilizer": [
            {
                "name": "Foliar Spray (NPK 19:19:19)",
                "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRk4yTRk1n56z2eUysubxOdSlhRm-E8n9pPxA&s"
            }
        ]
    },

    "Die Back": {
        "pesticide": [
            {
                "name": "Carbendazim Fungicide (Carbendazim 50% WP)",
                "image": "https://5.imimg.com/data5/SELLER/Default/2022/4/ZO/RV/TB/8542708/whatsapp-image-2022-03-09-at-1-24-37-pm.jpeg"
            }
        ],
        "fertilizer": [
            {
                "name": "Bio Fertilizer (Growth Promoter)",
                "image": "https://organicbazar.net/cdn/shop/files/PlantGrowthPromoterNew.jpg?v=1703743756"
            }
        ]
    },

    "Gall Midge": {
        "pesticide": [
            {
                "name": "Imidacloprid Insecticide (Imidacloprid 17.8% SL)",
                "image": "https://www.gujaratpesticides.com/wp-content/uploads/2022/08/MOLDOR-200-copy.png"
            }
        ],
        "fertilizer": [
            {
                "name": "Micronutrient Fertilizer (Zn based)",
                "image": "https://mahadhan.co.in/wp-content/uploads/2017/05/Chelatedzn1-kg.jpg"
            }
        ]
    },

    "Bacterial Canker": {
        "pesticide": [
            {
                "name": "Copper Oxychloride Fungicide (Copper Oxychloride 50% WP)",
                "image": "https://easy2agri.in/cdn/shop/files/1.jpg?v=1685599192"
            }
        ],
        "fertilizer": [
            {
                "name": "Micronutrient Spray (Boron + Zinc)",
                "image": "https://5.imimg.com/data5/SELLER/Default/2022/12/EN/UZ/UU/5280580/zinbo-liquid-1-ltr-500x500.jpg"
            }
        ]
    },

    "Cutting Weevil": {
        "pesticide": [
            {
                "name": "Chlorpyrifos Insecticide (Chlorpyrifos 20% EC)",
                "image": "https://dujjhct8zer0r.cloudfront.net/media/prod_image/2740249481743498805.webp"
            }
        ],
        "fertilizer": [
            {
                "name": "Foliar Spray (NPK 19:19:19)",
                "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRk4yTRk1n56z2eUysubxOdSlhRm-E8n9pPxA&s"
            }
        ]
    },

    "Sooty Mould": {
        "pesticide": [
            {
                "name": "Neem Oil Spray (Azadirachtin based)",
                "image": "https://m.media-amazon.com/images/I/71kjm8ZSN6L.jpg"
            }
        ],
        "fertilizer": [
            {
                "name": "Micronutrient Fertilizer Spray",
                "image": "https://mahadhan.co.in/wp-content/uploads/2017/05/mircronutrients2-300x413.jpg"
            }
        ]
    },

    "Healthy": {
        "pesticide": [
            {
                "name": "Not Required",
                "image": "https://img.freepik.com/premium-vector/no-action-required-red-rubber-stamp-with-text-white-background_545399-3715.jpg"
            }
        ],
        "fertilizer": [
            {
                "name": "Micronutrient Fertilizer",
                "image": "https://mahadhan.co.in/wp-content/uploads/2017/05/mircronutrients2-300x413.jpg"
            }
        ]
    }
}

# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "Mango Backend Running"

@app.route("/sensor", methods=["POST"])
def sensor():
    latest_sensor["temperature"] = float(request.form.get("temperature"))
    latest_sensor["humidity"] = float(request.form.get("humidity"))
    latest_sensor["moisture"] = float(request.form.get("moisture"))
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if not file:
        return jsonify({"error": "image missing"}), 400

    img = image.load_img(BytesIO(file.read()), target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter = get_interpreter()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
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
    treatment = TREATMENT.get(label, {"pesticide": [], "fertilizer": []})

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "sensor": latest_sensor,
        "risk": risk,
        "precautions": precautions,
        "treatment": treatment
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
