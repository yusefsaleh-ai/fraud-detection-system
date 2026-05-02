from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from blockchain.simple_blockchain import Blockchain

app = Flask(__name__)
CORS(app)

model = joblib.load("model/random_forest_model.pkl")
blockchain = Blockchain()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "amount_gbp",
            "hour_of_day",
            "merchant_category",
            "payment_method",
            "transaction_velocity",
            "location_risk",
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        response = {
            "prediction": int(prediction),
            "label": "fraud" if prediction == 1 else "not fraud",
            "confidence_not_fraud": float(probabilities[0]),
            "confidence_fraud": float(probabilities[1]),
        }

        # Log transaction + result to blockchain ledger
        blockchain.add_block({
            "transaction": data,
            "result": response
        })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running"})


if __name__ == "__main__":
    app.run(debug=True)