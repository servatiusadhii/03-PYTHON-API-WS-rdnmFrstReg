from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
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

# Variabel Global biar bisa diakses route
model_global = None
metrics_global = {}

# --- FUNGSI HELPER ---
def internal_train_manual(dataset, training_params):
    df = pd.DataFrame(dataset)
    
    cols_to_fix = ["jumlah_ayam", "pakan_total_kg", "kematian", "afkir", "telur_kg"]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fitur Engineering: Harus sama antara training dan prediction
    df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    # Fitur yang digunakan
    features = ["jumlah_ayam", "pakan_per_ayam", "kematian", "afkir"]
    X = df[features]
    y = df["telur_kg"]

    n_estimators = int(training_params.get("n_estimators", 200))
    random_state = int(training_params.get("random_state", 42))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return {
        "model": model,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "X_test": X_test, # Disimpan buat hitung skor real-time
        "y_test": y_test
    }

# --- ROUTE PREDICT MANUAL ---
@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    global model_global, metrics_global
    
    # Ambil data dari JSON Body (karena POST)
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data body"}), 400

    try:
        # 1. Ambil & Validasi Input
        jml_ayam = float(data.get('jumlah_ayam', 0))
        pakan_total = float(data.get('pakan_total_kg', 0))
        kematian = float(data.get('kematian', 0))
        afkir = float(data.get('afkir', 0))

        if jml_ayam <= 0:
            return jsonify({"status": "error", "message": "Jumlah ayam harus > 0"}), 400

        # Hitung fitur tambahan (pakan_per_ayam) agar sinkron dengan model
        pakan_per_ayam = pakan_total / jml_ayam
        
        # Susun array fitur (JUMLAH HARUS 4: ayam, pakan_per_ayam, kematian, afkir)
        input_array = np.array([[jml_ayam, pakan_per_ayam, kematian, afkir]])

        # 2. Cek apakah model sudah di-train
        if model_global is None:
            return jsonify({"status": "error", "message": "Model belum dilatih. Klik Tampilkan Data dulu!"}), 400

        # 3. Hitung Prediksi
        prediction = model_global.predict(input_array)[0]
        
        # 4. HITUNG ERROR DINAMIS (REAL berdasarkan variansi Tree)
        # Ambil prediksi dari tiap tree untuk melihat ketidakpastian
        all_tree_preds = [tree.predict(input_array)[0] for tree in model_global.estimators_]
        
        # MSE dinamis = variansi antar tree
        dynamic_mse = np.var(all_tree_preds) 
        # MAE dinamis = rata-rata selisih absolut dari rata-rata prediksi
        dynamic_mae = np.mean(np.abs(all_tree_preds - prediction))
        
        # R2 dinamis (bisa diambil dari performa test set terakhir)
        r2_base = metrics_global.get('R2', 0)

        return jsonify({
            "status": "success",
            "prediksi": {
                "harian_telur_kg": round(prediction, 2),
                "bulanan_telur_kg": round(prediction * 30, 2),
                "harian_telur_butir": int(prediction * 16),
                "telur_per_ayam": round(prediction / jml_ayam, 4)
            },
            "akurasi": {
                "MAE_kg": round(dynamic_mae, 3),
                "MSE_kg": round(dynamic_mse, 3),
                "RMSE_kg": round(np.sqrt(dynamic_mse), 3),
                "R2": round(r2_base, 4),
                "MAE_per_ayam": round(dynamic_mae / jml_ayam, 6),
                "MSE_per_ayam": round(dynamic_mse / jml_ayam, 6),
                "RMSE_per_ayam": round(np.sqrt(dynamic_mse) / jml_ayam, 6)
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 API Training Model Produksi Telur (ANTI DATA BOCOR)"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
