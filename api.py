from flask import Flask, request, jsonify
import train_model # Import file kodingan tadi

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    if not data or not data.get("dataset") or len(data.get("dataset")) < 10:
        return jsonify({"status": "error", "message": "Dataset minimal 10 baris"}), 400

    try:
        # Panggil fungsi dari train_model.py
        result = train_model.process_train_logic(data.get("dataset"), data.get("training", {}))
        result["status"] = "success"
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    data = request.get_json()
    try:
        # Panggil fungsi dari train_model.py
        result = train_model.process_predict_manual_logic(data)
        result["status"] = "success"
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 API Produksi Telur v2 - Logic Refactored"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)