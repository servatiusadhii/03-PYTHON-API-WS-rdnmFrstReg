from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# --- FUNGSI HELPER (DIPERBAIKI) ---
def internal_train_manual(dataset, training_params):
    df = pd.DataFrame(dataset)
    
    # Konversi ke numeric dan buang baris yang rusak
    cols_to_fix = ["jumlah_ayam", "pakan_total_kg", "kematian", "afkir", "telur_kg"]
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=cols_to_fix, inplace=True)

    # Feature Engineering: Penting biar hasil dinamis!
    df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
    
    # Hindari pembagian dengan nol
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    X = df[["jumlah_ayam", "pakan_per_ayam", "kematian", "afkir"]]
    y = df["telur_kg"]

    # Parameter model dibuat lebih fleksibel biar gak "stuck" di angka rata-rata
    n_estimators = int(training_params.get("n_estimators", 100))
    random_state = int(training_params.get("random_state", 42))
    
    # Split data (sesuaikan test_size jika data dikit)
    test_size = 0.2 if len(df) > 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model Random Forest (Settingan dikurangi biar sensitif terhadap input manual)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None, # Biar model bisa belajar detail data
        min_samples_leaf=1, # Biar lebih sensitif terhadap perubahan input
        min_samples_split=2,
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

@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    data = request.get_json()
    try:
        dataset = data.get("dataset")
        if not dataset:
            return jsonify({"status": "error", "message": "Dataset kosong"}), 400
            
        df = pd.DataFrame(dataset)
        
        # 1. Konversi Data & Bersihkan sesuai input baru
        # Input: umur_ayam, jumlah_ayam, pakan_total_kg, kematian, persentase_bertelur
        # Target di dataset biasanya telur_kg (kita hitung dari input historis jika perlu)
        cols_required = ["umur_ayam", "jumlah_ayam", "pakan_total_kg", "kematian", "persentase_bertelur"]
        
        for col in cols_required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Karena target prediksi kita adalah telur_kg, kita buat kolom target di df
        # Rumus: (jumlah_ayam * persentase_bertelur / 100) * 0.062 (asumsi berat telur 62g)
        if "telur_kg" not in df.columns:
            df["telur_kg"] = (df["jumlah_ayam"] * (df["persentase_bertelur"] / 100)) * 0.062
            
        df.dropna(inplace=True)

        # 2. Feature Engineering
        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
        
        # X sekarang menggunakan fitur yang dikirim dari Laravel
        features = ["umur_ayam", "jumlah_ayam", "pakan_per_ayam", "kematian", "persentase_bertelur"]
        X = df[features]
        y = df["telur_kg"]

        # 3. Training Model (Random Forest agar lebih dinamis dibanding Linear)
        # Jika data sangat sedikit, perkecil test_size
        test_size = 0.2 if len(df) > 10 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Settingan model agar sensitif terhadap input manual user
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        MAE = mean_absolute_error(y_test, y_pred)
        MSE = mean_squared_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred)

        # 4. Prediksi Data Manual dari Request
        jml_ayam_input = float(data.get("jumlah_ayam", 0))
        pakan_input = float(data.get("pakan_total_kg", 0))
        kematian_input = float(data.get("kematian", 0))
        umur_input = float(data.get("umur_ayam", 0))
        persen_input = float(data.get("persentase_bertelur", 0))

        if jml_ayam_input <= 0:
            return jsonify({"status": "error", "message": "Jumlah ayam harus > 0"}), 400

        pakan_per_ayam_input = pakan_input / jml_ayam_input
        
        # Susun array input sesuai urutan 'features'
        X_input = pd.DataFrame([[
            umur_input, 
            jml_ayam_input, 
            pakan_per_ayam_input, 
            kematian_input, 
            persen_input
        ]], columns=features)
        
        pred_kg = float(model.predict(X_input)[0])
        pred_kg = max(pred_kg, 0) # Safety check

        # 5. Response Final
        return jsonify({
            "status": "success",
                "MAE": round(float(MAE), 4),
                "R2": round(float(R2), 4),
                "train_rows": len(X_train),
            "prediksi": {
                "harian_telur_kg": round(pred_kg, 2),
                "bulanan_telur_kg": round(pred_kg * 30, 2),
                "telur_per_ayam": round(pred_kg / jml_ayam_input, 4),
                "harian_telur_butir": int(round(pred_kg / 0.062)), # Konversi balik ke butir
                "bulanan_telur_butir": int(round((pred_kg * 30) / 0.062))
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route("/", methods=["GET"])
def home():
    return "🚀 API Training Model Produksi Telur (ANTI DATA BOCOR)"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
