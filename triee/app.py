import logging
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from model.minimal_model import GCN_TCN_CapsNet  # Import your model class

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Store latest prediction for frontend display
latest_prediction = {"predicted_activity": "Waiting for data..."}

# Load the model once at startup
logger.info("Loading model...")
model = GCN_TCN_CapsNet(input_dim=3, num_classes=6, num_nodes=128)
model.load_state_dict(torch.load('model/saved_model_vvip.pth', map_location=torch.device('cpu')))
model.eval()
logger.info("Model loaded successfully.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json(force=True)
        logger.debug("=== 📥 RAW DATA RECEIVED ===")
        logger.debug(data)

        if not data:
            logger.warning("No data received")
            return jsonify({"error": "No data received"}), 400

        raw_series = data.get("payload", [])

        logger.debug(f"📊 Payload entries: {len(raw_series)}")
        acc_count = sum(1 for entry in raw_series if 'accelerometer' in entry['name'])
        logger.debug(f"📈 Accelerometer readings: {acc_count}")

        # Preprocess the input data
        input_tensor = preprocess_input(raw_series)

        logger.debug(f"✅ Preprocessed tensor shape: {input_tensor.shape}")

        # Predict the class of activity
        predicted_class = load_model_and_predict(model, input_tensor)

        # Update the latest prediction
        latest_prediction["predicted_activity"] = predicted_class
        logger.info(f"Predicted activity: {predicted_class}")
        return jsonify({"predicted_activity": predicted_class})

    except Exception as e:
        logger.error(f"❌ Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/latest", methods=["GET"])
def get_latest_prediction():
    return jsonify(latest_prediction)

def preprocess_input(data_series):
    accelerometer_data = []

    for entry in data_series:
        name = entry['name']
        values = entry['values']
        if 'accelerometer' in name:
            accelerometer_data.append([values.get('x', 0), values.get('y', 0), values.get('z', 0)])

    # Ensure exactly 128 samples (padding or trimming)
    if len(accelerometer_data) < 128:
        accelerometer_data.extend([[0, 0, 0]] * (128 - len(accelerometer_data)))
    elif len(accelerometer_data) > 128:
        accelerometer_data = accelerometer_data[:128]

    accelerometer_data = np.array(accelerometer_data, dtype=np.float32)
    accelerometer_data = (accelerometer_data - accelerometer_data.mean(axis=0)) / (accelerometer_data.std(axis=0) + 1e-6)

    input_tensor = torch.tensor(accelerometer_data, dtype=torch.float32).unsqueeze(0)  # (1, 128, 3)
    logger.debug(f"Preprocessed tensor shape before passing to model: {input_tensor.shape}")
    return input_tensor



def load_model_and_predict(model, input_tensor):
    """
    Use the trained model to make a prediction based on the input tensor.
    """
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    activity_labels = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']
    return activity_labels[predicted_class]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
