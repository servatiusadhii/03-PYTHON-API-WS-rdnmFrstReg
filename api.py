from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

from train_model import train_model  # ⬅️ PENTING

app = Flask(__name__)

MODEL_PATH = "model_telur.pkl"


# =========================
# Helper: load model
# =========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# =========================
# TRAIN MODEL (DARI LARAVEL)
# =========================
@app.route("/train", methods=["POST"])
def train():
    req = request.get_json()

    if not req:
        return jsonify({"error": "Request JSON kosong"}), 400

    dataset = req.get("dataset")
    training = req.get("training")

    if not dataset or not training:
        return jsonify({"error": "Dataset atau training parameter tidak lengkap"}), 400

    try:
        result = train_model(dataset, training)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =========================
# PREDICT (MODEL TERBARU)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return jsonify({"error": "Model belum ditraining"}), 400

    data = request.get_json()

    try:
        fitur = np.array([[
            int(data["jumlah_ayam"]),
            float(data["pakan_total_kg"]),
            int(data["kematian"]),
            int(data["afkir"])
        ]])
    except Exception:
        return jsonify({"error": "Parameter tidak valid"}), 400

    prediksi = model.predict(fitur)

    return jsonify({
        "prediksi_telur_kg": round(float(prediksi[0]), 2)
    })

@app.route("/traind", methods=["POST"])
def traind():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Request harus JSON"}), 400

    dataset = data.get("dataset")
    training = data.get("training")

    # cek dataset
    if not dataset or len(dataset) < 2:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 2 baris, sekarang: " + str(len(dataset) if dataset else 0)
        }), 400

    try:
        df = pd.DataFrame(dataset)

        required_cols = ["pakan_total_kg", "kematian", "afkir", "telur_kg"]
        for c in required_cols:
            if c not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom {c} tidak ada"}), 400

        df["pakan_total_kg"] = df["pakan_total_kg"].astype(float)
        df["kematian"] = df["kematian"].astype(int)
        df["afkir"] = df["afkir"].astype(int)
        df["telur_kg"] = df["telur_kg"].astype(float)

        X = df[["pakan_total_kg", "kematian", "afkir"]]
        y = df["telur_kg"]

        n_estimators = int(training.get("n_estimators", 100))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth")
        max_depth = int(max_depth) if max_depth else None

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth
        )

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))   # FIX buat sklearn lama
        r2 = r2_score(y_test, pred)

        return jsonify({
            "status": "success",
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================
# HEALTH CHECK
# =========================
@app.route("/", methods=["GET"])
def home():
    return "🚀 API Random Forest Telur Aktif"


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=True
    )
