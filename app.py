from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# ✅ Safe model loading
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("❌ model.pkl not found! Run train_model.py first")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route('/')
def home():
    return "🚗 Accident Prediction API Running!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get data safely
        data = request.json.get('features')

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ✅ Check feature count (VERY IMPORTANT)
        if len(data) != 5:
            return jsonify({"error": "Expected 5 features"}), 400

        # ✅ Model check
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # ✅ Prediction
        prediction = model.predict([data])

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# 🔥 Optional: Fix your /messages error (dummy route)
@app.route('/messages', methods=['POST'])
def messages():
    return jsonify({
        "reply": "AI module coming soon 🚀"
    })

if __name__ == "__main__":
    app.run()