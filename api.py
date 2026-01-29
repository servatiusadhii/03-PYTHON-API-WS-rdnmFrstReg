from flask import Flask, request, jsonify
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Request JSON kosong"}), 400

    dataset = data.get("dataset")
    training = data.get("training")
    input_prediksi_harian = data.get("input_prediksi_harian")
    input_prediksi_bulanan = data.get("input_prediksi_bulanan")

    if not dataset or not training:
        return jsonify({"status": "error", "message": "Dataset atau training parameter tidak lengkap"}), 400

    if len(dataset) < 2:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 2 baris, sekarang: " + str(len(dataset))
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        # Prediksi harian berdasarkan input
        pred_harian = None
        if input_prediksi_harian:
            pred_harian = model.predict([[
                float(input_prediksi_harian["pakan_total_kg"]),
                int(input_prediksi_harian["kematian"]),
                int(input_prediksi_harian["afkir"])
            ]])[0]

        # Prediksi bulanan berdasarkan input (sum bulanan)
        pred_bulanan = None
        if input_prediksi_bulanan:
            pred_bulanan = model.predict([[
                float(input_prediksi_bulanan["pakan_total_kg"]),
                int(input_prediksi_bulanan["kematian"]),
                int(input_prediksi_bulanan["afkir"])
            ]])[0]

        return jsonify({
            "status": "success",
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 2),
            "prediksi_harian_telur_kg": round(float(pred_harian), 2) if pred_harian else None,
            "prediksi_bulanan_telur_kg": round(float(pred_bulanan), 2) if pred_bulanan else None
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "🚀 API Random Forest Telur Aktif"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
