import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------- CATATAN ------------------------
# Random Forest Regression vs Random Forest Classification

# File yang kamu punya (RandomForestRegressor) itu fungsinya untuk Regresi (memprediksi angka kontinu, seperti berat telur dalam kg).
# Sedangkan Akurasi, Presisi, Recall, dan F1-Score itu adalah metrik untuk Klasifikasi (RandomForestClassifier) seperti (memprediksi kategori, misal: "Spam" vs "Bukan Spam").
# Analogi: Kamu nggak bisa ngitung "Akurasi" (Benar/Salah) pada timbangan digital. Kalau berat telur aslinya 1.0 kg dan timbangan bilang 1.1 kg, itu namanya Error (0.1 kg), bukan "Salah" secara mutlak.

# Solusinya: Pakai "Tolerance Threshold"
# Supaya sesuai dengan permintaan, kita bisa "mengubah" hasil prediksi regresi tadi jadi klasifikasi sementara. Kita anggap prediksi "BENAR" (1) kalau selisihnya tipis banget dari aslinya, dan "SALAH" (0) kalau meleset jauh.
# Kita bisa tentukan batas toleransinya, misalnya 10%.
# Formula logikanya:
# Correct = ∣ytest​−ypred​∣ ≤ (0.1×ytest​)
# ------------------------------------------------------------

def train_model(dataset, training_params):
    """
    dataset: list of dict (dari Laravel)
    training_params: dict (optional)
    """

    # =====================
    # 1. Dataset → DataFrame
    # =====================
    df = pd.DataFrame(dataset)

    # pastikan kolom wajib ada
    required_cols = [
        "jumlah_ayam",
        "pakan_total_kg",
        "kematian",
        "afkir",
        "telur_kg"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset")

    # =====================
    # 2. Feature Engineering (ANTI BOCOR)
    # =====================
    # normalisasi pakan (lebih realistis)
    df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]

    # target
    y = df["telur_kg"]

    # fitur FINAL (AMAN)
    X = df[[
        "jumlah_ayam",
        "pakan_per_ayam",
        "kematian",
        "afkir",
    ]]

    # =====================
    # 3. Training params
    # =====================
    n_estimators = training_params.get("n_estimators", 150)
    random_state = training_params.get("random_state", 42)
    max_depth = training_params.get("max_depth", 6)

    # =====================
    # 4. Split data
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    # =====================
    # 5. Model (ANTI OVERFIT)
    # =====================
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # =====================
    # 6. Evaluasi
    # =====================
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # akurasi persentase (interpretatif, bukan sklearn default)
    accuracy_pct = 100 - (mae / y_test.mean() * 100)

    # =====================
    # 7. Save model
    # =====================
    with open("model_telur.pkl", "wb") as f:
        pickle.dump(model, f)

    # =====================
    # 8. Return hasil
    # =====================
    return {
        "status": "success",
        "MAE (kg)": round(mae, 2),
        "RMSE (kg)": round(rmse, 2),
        "R2": round(r2, 3),
        "Accuracy (%)": round(accuracy_pct, 2),
        "Train_rows": len(X_train),
        "Test_rows": len(X_test),
        "Features_used": list(X.columns)
    }
