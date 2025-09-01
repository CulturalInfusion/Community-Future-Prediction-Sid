import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(f"Current working directory: {os.getcwd()}")

# Check models directory
model_dir = "models"
if not os.path.exists(model_dir):
    print(f"❌ Models directory not found!")
    exit()

print(f"✅ Found models directory")
print("Available model files:", os.listdir(model_dir))

# Try to load models - only load ones that exist
models = {}
model_files = {
    "african": "lstm_model_social_media_representation_african.keras",
    "asian": "lstm_model_social_media_representation_asian.keras",
    "european": "lstm_model_social_media_representation_european.keras", 
    "hispanic": "lstm_model_social_media_representation_hispanic.keras",
    "indigenous": "lstm_model_social_media_representation_indigenous.keras",
    "middle eastern": "lstm_model_social_media_representation_middle_eastern.keras"
}

# Load only existing models
for community, filename in model_files.items():
    model_path = os.path.join(model_dir, filename)
    if os.path.exists(model_path):
        try:
            models[community] = load_model(model_path)
            print(f"✅ Successfully loaded {community} model")
        except Exception as e:
            print(f"❌ Error loading {community} model: {e}")
    else:
        print(f"⚠️ Model file not found: {filename}")

if not models:
    print("❌ No models were loaded! Please check your model files.")
    exit()

print(f"🎉 Successfully loaded {len(models)} models: {list(models.keys())}")

app = Flask(__name__, template_folder="templates")

@app.route("/")
def homepage():
    return render_template("Flask_UI.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = data.get("input")
        community = data.get("community", "").strip().lower()

        # Check if requested community model exists
        if community not in models:
            available_communities = list(models.keys())
            return jsonify({
                "error": f"Community '{community}' model not available.", 
                "available_communities": available_communities
            }), 400

        if not input_data or len(input_data) != 20:
            return jsonify({"error": "Please provide exactly 20 numeric values."}), 400

        try:
            input_data = [float(x) for x in input_data]
        except ValueError:
            return jsonify({"error": "All input values must be numeric."}), 400

        model = models[community]
        current_input = np.array(input_data).reshape(1, 20, 1)

        # Predict next 11 years (ORIGINAL VERSION - NO MODIFICATIONS)
        predictions = []
        for _ in range(11):
            next_val = model.predict(current_input)[0, 0]
            predictions.append(float(next_val))
            current_input = np.append(current_input[:, 1:, :], [[[next_val]]], axis=1)

        start_year = 2025
        forecast = {str(start_year + i): round(pred, 4) for i, pred in enumerate(predictions)}

        return jsonify({
            "community": community,
            "predictions": forecast,
            "available_communities": list(models.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
