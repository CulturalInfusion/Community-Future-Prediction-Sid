import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load all trained LSTM models for different communities
model_dir = os.path.join(os.path.dirname(__file__), "models")
models = {
    "african": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_african.keras")),
    "asian": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_asian.keras")),
    "european": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_european.keras")),
    "hispanic": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_hispanic.keras")),
    "indigenous": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_indigenous.keras")),
    "middle eastern": load_model(os.path.join(model_dir, "lstm_model_social_media_representation_middle_eastern.keras"))
}

app = Flask(__name__, template_folder="templates")

@app.route("/")
def homepage():
    return render_template("Flask_UI.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = data.get("input")
        community = data.get("community", "middle eastern").strip().lower()

        if community not in models:
            return jsonify({"error": f"Community '{community}' is not supported."}), 400

        if not input_data or len(input_data) != 20:
            return jsonify({"error": "Please provide exactly 20 numeric values."}), 400

        try:
            input_data = [float(x) for x in input_data]
        except ValueError:
            return jsonify({"error": "All input values must be numeric."}), 400

        model = models[community]
        current_input = np.array(input_data).reshape(1, 20, 1)

        # Predict next 11 years
        predictions = []
        for _ in range(11):
            next_val = model.predict(current_input)[0, 0]
            predictions.append(float(next_val))
            current_input = np.append(current_input[:, 1:, :], [[[next_val]]], axis=1)

        start_year = 2025
        forecast = {str(start_year + i): round(pred, 4) for i, pred in enumerate(predictions)}

        return jsonify({
            "community": community,
            "predictions": forecast
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
