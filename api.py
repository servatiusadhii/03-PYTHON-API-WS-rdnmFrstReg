from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

def internal_train_manual(dataset, training_params):
    df = pd.DataFrame(dataset)
    # Feature engineering
    df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
    X = df[["jumlah_ayam", "pakan_per_ayam", "kematian", "afkir"]]
    y = df["telur_kg"]

    n_estimators = int(training_params.get("n_estimators", 200))
    random_state = int(training_params.get("random_state", 42))
    max_depth = training_params.get("max_depth")
    if max_depth is not None: max_depth = int(max_depth)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "MAE": MAE,
        "MSE": MSE,
        "R2": R2,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "avg_ayam_hist": df["jumlah_ayam"].mean(),
        "features": list(X.columns)
    }

@app.route("/train", methods=["POST"])
def train():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Request JSON kosong"}), 400

    dataset = data.get("dataset")
    training = data.get("training", {})

    if not dataset or len(dataset) < 10:
        return jsonify({"status": "error", "message": "Dataset minimal 10 baris"}), 400

    try:
        df = pd.DataFrame(dataset)
        required_cols = ["jumlah_ayam", "pakan_total_kg", "kematian", "afkir", "telur_kg"]
        for c in required_cols:
            if c not in df.columns:
                return jsonify({"status": "error", "message": f"Kolom '{c}' tidak ditemukan"}), 400

        # Feature engineering
        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
        X = df[["jumlah_ayam", "pakan_per_ayam", "kematian", "afkir"]]
        y = df["telur_kg"]

        # Training params
        n_estimators = int(training.get("n_estimators", 150))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth", 6)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # Model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        RMSE = np.sqrt(MSE)
        R2 = r2_score(y_test, y_pred)

        avg_ayam = X_test["jumlah_ayam"].mean()
        MAE_per_ayam = MAE / avg_ayam
        MSE_per_ayam = MSE / (avg_ayam ** 2)
        RMSE_per_ayam = RMSE / avg_ayam

        # Prediksi ringkasan
        harian_telur_kg = y.mean()
        bulanan_telur_kg = y.sum()
        telur_per_ayam = harian_telur_kg / df["jumlah_ayam"].mean()
        harian_telur_butir = harian_telur_kg / 0.06  # 1 telur ≈ 60 gr
        bulanan_telur_butir = bulanan_telur_kg / 0.06

        # Save model
        with open("model_telur.pkl", "wb") as f:
            pickle.dump(model, f)

        # Final JSON output
        return jsonify({
            "status": "success",
            "MAE_kg": round(MAE, 3),
            "MSE_kg": round(MSE, 3),
            "RMSE_kg": round(RMSE, 3),
            "MAE_per_ayam": round(MAE_per_ayam, 6),
            "MSE_per_ayam": round(MSE_per_ayam, 6),
            "RMSE_per_ayam": round(RMSE_per_ayam, 6),
            "R2": round(float(R2), 3),
            "Train_rows": len(X_train),
            "Test_rows": len(X_test),
            "Features_used": list(X.columns),
            "prediksi": {
                "harian_telur_kg": round(harian_telur_kg, 2),
                "bulanan_telur_kg": round(bulanan_telur_kg, 2),
                "telur_per_ayam": round(telur_per_ayam, 4),
                "harian_telur_butir": int(round(harian_telur_butir)),
                "bulanan_telur_butir": int(round(bulanan_telur_butir))
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "JSON Body kosong"}), 400
        
    dataset_history = data.get("dataset")
    
    try:
        # 1. Jalankan training pakai helper tadi
        res = internal_train_manual(dataset_history, data.get("training", {}))

        # 2. Ambil data input manual
        jml_ayam = float(data.get("jumlah_ayam"))
        pakan_kg = float(data.get("pakan_total_kg"))
        kematian = float(data.get("kematian", 0))
        afkir = float(data.get("afkir", 0))

        # 3. Prediksi
        pakan_per_ayam = pakan_kg / jml_ayam
        X_input = [[jml_ayam, pakan_per_ayam, kematian, afkir]]
        pred_kg = res["model"].predict(X_input)[0]

        # 4. Response IDENTIK dengan /train
        return jsonify({
            "status": "success",
            "MAE_kg": round(float(res["MAE"]), 3),
            "MSE_kg": round(float(res["MSE"]), 3),
            "RMSE_kg": round(float(np.sqrt(res["MSE"])), 3),
            "MAE_per_ayam": round(float(res["MAE"] / res["avg_ayam_hist"]), 6),
            "MSE_per_ayam": round(float(res["MSE"] / (res["avg_ayam_hist"]**2)), 6),
            "RMSE_per_ayam": round(float(np.sqrt(res["MSE"]) / res["avg_ayam_hist"]), 6),
            "R2": round(float(res["R2"]), 3),
            "Train_rows": res["train_rows"],
            "Test_rows": res["test_rows"],
            "Features_used": res["features"],
            "prediksi": {
                "harian_telur_kg": round(float(pred_kg), 2),
                "bulanan_telur_kg": round(float(pred_kg * 30), 2),
                "telur_per_ayam": round(float(pred_kg / jml_ayam), 4),
                "harian_telur_butir": int(round(pred_kg / 0.06)),
                "bulanan_telur_butir": int(round((pred_kg * 30) / 0.06))
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 API Training Model Produksi Telur (ANTI DATA BOCOR)"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
