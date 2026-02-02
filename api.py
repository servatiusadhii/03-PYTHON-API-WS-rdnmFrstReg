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

    if not dataset or not training:
        return jsonify({"status": "error", "message": "Dataset atau training parameter tidak lengkap"}), 400

    if len(dataset) < 7:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 7 hari untuk prediksi presisi"
        }), 400

    try:
        df = pd.DataFrame(dataset)

        required_cols = [
            "tanggal", "jumlah_ayam", "pakan_total_kg",
            "kematian", "afkir", "telur_kg"
        ]
        for c in required_cols:
            if c not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom {c} tidak ada"}), 400

        # ===========================
        # PREPROCESSING
        # ===========================
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df = df.sort_values("tanggal")

        df["jumlah_ayam"] = df["jumlah_ayam"].astype(int)
        df["kematian"] = df["kematian"].astype(int)
        df["afkir"] = df["afkir"].astype(int)
        df["pakan_total_kg"] = df["pakan_total_kg"].astype(float)
        df["telur_kg"] = df["telur_kg"].astype(float)

        # ===========================
        # FEATURE ENGINEERING
        # ===========================
        df["ayam_aktif"] = df["jumlah_ayam"] - df["kematian"] - df["afkir"]
        df["ayam_aktif"] = df["ayam_aktif"].clip(lower=1)

        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["ayam_aktif"]
        df["telur_per_ayam"] = df["telur_kg"] / df["ayam_aktif"]

        # fitur historis (lag)
        df["telur_lag_1"] = df["telur_per_ayam"].shift(1)
        df["telur_lag_3"] = df["telur_per_ayam"].rolling(3).mean()

        df = df.dropna()

        # ===========================
        # HITUNG BERAT RATA-RATA PER BUTIR
        # ===========================
        if "jumlah_telur_butir" in df.columns:
            df["berat_per_butir"] = df["telur_kg"] / df["jumlah_telur_butir"]
        else:
            df["berat_per_butir"] = df["telur_per_ayam"] / 1  # estimasi sederhana

        avg_berat_per_butir = df["berat_per_butir"].mean()

        # ===========================
        # MODEL INPUT
        # ===========================
        X = df[[
            "ayam_aktif",
            "pakan_per_ayam",
            "kematian",
            "afkir",
            "telur_lag_1",
            "telur_lag_3"
        ]]
        y = df["telur_per_ayam"]

        model = RandomForestRegressor(
            n_estimators=int(training.get("n_estimators", 200)),
            random_state=int(training.get("random_state", 42)),
            max_depth=training.get("max_depth")
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred_test)
        mse = mean_squared_error(y_test, pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred_test)

        # ===========================
        # PREDIKSI HARIAN
        # ===========================
        last = df.tail(1)
        X_last = X.tail(1)

        pred_per_ayam = model.predict(X_last)[0]
        pred_harian_kg = pred_per_ayam * float(last["ayam_aktif"])

        # prediksi per butir
        pred_harian_butir = pred_harian_kg / avg_berat_per_butir
        pred_bulanan_butir = pred_harian_butir * 30
        pred_bulanan_kg = pred_harian_kg * 30

        return jsonify({
            "status": "success",
            "akurasi": {
                "MAE_per_ayam": round(float(mae), 4),
                "MSE_per_ayam": round(float(mse), 4),
                "RMSE_per_ayam": round(float(rmse), 4),
                "R2": round(float(r2), 3)
            },
            "prediksi": {
                "harian_telur_kg": round(float(pred_harian_kg), 2),
                "bulanan_telur_kg": round(float(pred_bulanan_kg), 2),
                "telur_per_ayam": round(float(pred_per_ayam), 4),
                "harian_telur_butir": round(float(pred_harian_butir), 0),
                "bulanan_telur_butir": round(float(pred_bulanan_butir), 0)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "🚀 API Prediksi Telur Presisi (Per Ayam Aktif & Per Butir)"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
