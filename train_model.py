import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Konstanta Global
BERAT_TELUR_KG = 0.048 

def process_train_logic(dataset, training_params):
    df = pd.DataFrame(dataset)
    
    # 1. Feature Engineering
    df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]
    X = df[["jumlah_ayam", "pakan_per_ayam", "kematian", "afkir"]]
    y = df["telur_kg"]

    # 2. Params
    n_estimators = int(training_params.get("n_estimators", 150))
    random_state = int(training_params.get("random_state", 42))
    max_depth = training_params.get("max_depth", 6)

    # 3. Split & Train
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

    # 4. Evaluasi Regresi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 5. Metrik Threshold (Versi Dosen)
    threshold = 0.1
    y_test_class = np.ones(len(y_test))
    y_pred_class = [1 if (abs(r - p) / r if r != 0 else 0) <= threshold else 0 for r, p in zip(y_test, y_pred)]

    # 6. Produksi Summary
    avg_ayam = X_test["jumlah_ayam"].mean()
    harian_kg = y.mean()
    total_kg = y.sum()

    # Save Model
    with open("model_telur.pkl", "wb") as f:
        pickle.dump(model, f)

    # Return dict sesuai format JSON api.py
    return {
        "MAE_kg": round(mae, 3),
        "MSE_kg": round(mse, 3),
        "RMSE_kg": round(rmse, 3),
        "MAE_per_ayam": round(mae / avg_ayam, 6),
        "MSE_per_ayam": round(mse / (avg_ayam**2), 6),
        "RMSE_per_ayam": round(rmse / avg_ayam, 6),
        "R2": round(float(r2), 3),
        "Train_rows": len(X_train),
        "Test_rows": len(X_test),
        "Features_used": list(X.columns),
        "metrik": {
            "akurasi": f"{round(accuracy_score(y_test_class, y_pred_class) * 100, 2)}%",
            "presisi": round(float(precision_score(y_test_class, y_pred_class, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test_class, y_pred_class, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_test_class, y_pred_class, zero_division=0)), 3),
            "keterangan": "Toleransi error 10%"
        },
        "prediksi": {
            "harian_telur_kg": round(harian_kg, 2),
            "bulanan_telur_kg": round(total_kg, 2),
            "telur_per_ayam": round(harian_kg / df["jumlah_ayam"].mean(), 4),
            "harian_telur_butir": int(round(harian_kg / BERAT_TELUR_KG)),
            "bulanan_telur_butir": int(round(total_kg / BERAT_TELUR_KG))
        }
    }

def process_predict_manual_logic(data):
    # Logika yang sebelumnya numpuk di api.py dipindah ke sini
    dataset = data.get("dataset")
    df = pd.DataFrame(dataset)
    
    # Validasi Kolom & Konversi
    cols = ["umur_ayam", "jumlah_ayam", "pakan_total_kg", "kematian", "persentase_bertelur"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)

    # Ground Truth Calculation
    df["jumlah_butir"] = (df["jumlah_ayam"] * (df["persentase_bertelur"] / 100)).round().astype(int)
    df["telur_kg"] = df["jumlah_butir"] * BERAT_TELUR_KG
    df["pakan_per_ayam"] = df.apply(lambda x: x["pakan_total_kg"] / x["jumlah_ayam"] if x["jumlah_ayam"] > 0 else 0, axis=1)

    features = ["umur_ayam", "jumlah_ayam", "pakan_per_ayam", "kematian"]
    X, y = df[features], df["telur_kg"]

    # Train Model on the fly (Sesuai kodingan asli kamu)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y) # Menggunakan semua data karena ini manual check
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred) if len(y) > 1 else 1.0

    # Metrik Dosen
    threshold = 0.1
    y_class = np.ones(len(y))
    y_pred_class = [1 if (abs(r - p) / r if r != 0 else 0) <= threshold else 0 for r, p in zip(y, y_pred)]

    # Input User Prediction
    jml_ayam_in = float(data.get("jumlah_ayam", 0))
    pakan_in = float(data.get("pakan_total_kg", 0))
    persen_in = float(data.get("persentase_bertelur", 0))
    
    real_butir = int(round(jml_ayam_in * (persen_in / 100)))
    real_kg = round(real_butir * BERAT_TELUR_KG, 1)
    
    X_input = pd.DataFrame([[float(data.get("umur_ayam", 0)), jml_ayam_in, pakan_in/jml_ayam_in, float(data.get("kematian", 0))]], columns=features)
    pred_kg = max(float(model.predict(X_input)[0]), 0)

    return {
        "metrik": {
            "MAE": round(float(mae), 4),
            "R2": round(float(r2), 4),
            "akurasi_dosen": f"{round(accuracy_score(y_class, y_pred_class) * 100, 2)}%",
            "presisi_dosen": round(float(precision_score(y_class, y_pred_class, zero_division=0)), 3),
            "recall_dosen": round(float(recall_score(y_class, y_pred_class, zero_division=0)), 3),
            "f1_score_dosen": round(float(f1_score(y_class, y_pred_class, zero_division=0)), 3)
        },
        "prediksi": {
            "harian_telur_butir": real_butir,
            "harian_telur_kg": real_kg,
            "produktivitas_persen": round((real_butir / jml_ayam_in) * 100, 2),
            "fcr": round(pakan_in / real_kg, 2) if real_kg > 0 else 0,
            "model_telur_kg": round(pred_kg, 2),
            "model_telur_butir": int(round(pred_kg / BERAT_TELUR_KG)),
            "selisih_butir_model_vs_real": int(round(pred_kg / BERAT_TELUR_KG)) - real_butir
        }
    }