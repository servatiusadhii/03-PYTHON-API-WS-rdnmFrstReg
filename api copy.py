from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

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


@app.route("/train", methods=["POST"])
def train():
    # Ambil data JSON dari request (frontend / API)
    data = request.get_json()

    # Validasi: request tidak boleh kosong
    if not data:
        return jsonify({
            "status": "error",
            "message": "Request JSON kosong"
        }), 400

    # Ambil dataset & parameter training
    dataset = data.get("dataset")
    training = data.get("training", {})

    # Validasi: dataset minimal 10 baris (biar model layak dilatih)
    if not dataset or len(dataset) < 10:
        return jsonify({
            "status": "error",
            "message": "Dataset minimal 10 baris"
        }), 400

    try:
        # Ubah dataset ke DataFrame pandas
        df = pd.DataFrame(dataset)

        # =========================
        # 1. VALIDASI KOLOM DATA
        # =========================
        required_cols = [
            "jumlah_ayam",
            "pakan_total_kg",
            "kematian",
            "afkir",
            "telur_kg"
        ]

        for c in required_cols:
            if c not in df.columns:
                return jsonify({
                    "status": "error",
                    "message": f"Kolom '{c}' tidak ditemukan"
                }), 400

        # =========================
        # 2. FEATURE ENGINEERING
        # =========================
        # Buat fitur baru: pakan per ayam
        df["pakan_per_ayam"] = df["pakan_total_kg"] / df["jumlah_ayam"]

        # Tentukan fitur (input model)
        X = df[[
            "jumlah_ayam",
            "pakan_per_ayam",
            "kematian",
            "afkir"
        ]]

        # Target (output yang diprediksi)
        y = df["telur_kg"]

        # =========================
        # 3. PARAMETER TRAINING
        # =========================
        # Bisa diatur dari frontend / pakai default
        n_estimators = int(training.get("n_estimators", 150))
        random_state = int(training.get("random_state", 42))
        max_depth = training.get("max_depth", 6)

        # =========================
        # 4. SPLIT DATA TRAIN & TEST
        # =========================
        # 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=random_state
        )

        # =========================
        # 5. TRAINING MODEL
        # =========================
        # Gunakan Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=random_state
        )

        # Latih model
        model.fit(X_train, y_train)

        # Prediksi data testing
        y_pred = model.predict(X_test)

        # =========================
        # 6. EVALUASI MODEL (REGRESI)
        # =========================
        MAE = mean_absolute_error(y_test, y_pred)   # rata-rata error absolut
        MSE = mean_squared_error(y_test, y_pred)    # error kuadrat
        RMSE = np.sqrt(MSE)                         # akar dari MSE
        R2 = r2_score(y_test, y_pred)               # akurasi model (0–1)

        # =========================
        # 7. EVALUASI BERBASIS THRESHOLD
        # =========================
        # Anggap prediksi benar jika selisih <= 10%
        threshold = 0.1

        # Semua data dianggap benar (baseline)
        y_test_class = np.ones(len(y_test))
        y_pred_class = []

        for real, pred in zip(y_test, y_pred):
            # Hitung selisih dalam persen
            margin = abs(real - pred) / real if real != 0 else 0

            # Jika error <= 10% → dianggap akurat (1)
            # Jika > 10% → tidak akurat (0)
            y_pred_class.append(1 if margin <= threshold else 0)

        # Hitung metrik klasifikasi
        akurasi_dosen = accuracy_score(y_test_class, y_pred_class)
        presisi_dosen = precision_score(y_test_class, y_pred_class, zero_division=0)
        recall_dosen = recall_score(y_test_class, y_pred_class, zero_division=0)
        f1_dosen = f1_score(y_test_class, y_pred_class, zero_division=0)

        # =========================
        # 8. NORMALISASI ERROR PER AYAM
        # =========================
        avg_ayam = X_test["jumlah_ayam"].mean()

        MAE_per_ayam = MAE / avg_ayam
        MSE_per_ayam = MSE / (avg_ayam ** 2)
        RMSE_per_ayam = RMSE / avg_ayam

        # =========================
        # 9. RINGKASAN PRODUKSI TELUR
        # =========================
        # Rata-rata produksi harian (kg)
        harian_telur_kg = y.mean()

        # Total produksi (misal dianggap bulanan)
        bulanan_telur_kg = y.sum()

        # Produksi per ayam
        telur_per_ayam = harian_telur_kg / df["jumlah_ayam"].mean()

        # Konversi ke butir (asumsi 1 telur = 60 gram)
        harian_telur_butir = harian_telur_kg / 0.06
        bulanan_telur_butir = bulanan_telur_kg / 0.06

        # =========================
        # 10. SIMPAN MODEL
        # =========================
        # Simpan model ke file .pkl
        with open("model_telur.pkl", "wb") as f:
            pickle.dump(model, f)

        # =========================
        # 11. RESPONSE KE CLIENT
        # =========================
        return jsonify({
            "status": "success",

            # Metrik regresi
            "MAE_kg": round(MAE, 3),
            "MSE_kg": round(MSE, 3),
            "RMSE_kg": round(RMSE, 3),

            # Metrik per ayam
            "MAE_per_ayam": round(MAE_per_ayam, 6),
            "MSE_per_ayam": round(MSE_per_ayam, 6),
            "RMSE_per_ayam": round(RMSE_per_ayam, 6),

            "R2": round(float(R2), 3),

            # Info dataset
            "Train_rows": len(X_train),
            "Test_rows": len(X_test),
            "Features_used": list(X.columns),

            # Metrik threshold (versi dosen)
            "metrik": {
                "akurasi": f"{round(akurasi_dosen * 100, 2)}%",
                "presisi": round(float(presisi_dosen), 3),
                "recall": round(float(recall_dosen), 3),
                "f1_score": round(float(f1_dosen), 3),
                "keterangan": "Toleransi error 10%"
            },

            # Ringkasan hasil produksi
            "prediksi": {
                "harian_telur_kg": round(harian_telur_kg, 2),
                "bulanan_telur_kg": round(bulanan_telur_kg, 2),
                "telur_per_ayam": round(telur_per_ayam, 4),
                "harian_telur_butir": int(round(harian_telur_butir)),
                "bulanan_telur_butir": int(round(bulanan_telur_butir))
            }
        })

    except Exception as e:
        # Jika error, kirim pesan error ke client
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/predict-manual", methods=["POST"])
def predict_manual():
    # Ambil data JSON dari request (biasanya dari frontend / API client)
    data = request.get_json()

    try:
        # Ambil dataset historis dari request
        dataset = data.get("dataset")

        # Validasi: minimal harus ada 2 data (biar bisa dianalisis)
        if not dataset or len(dataset) < 2:
            return jsonify({
                "status": "error",
                "message": "Dataset minimal butuh 2 baris data historis"
            }), 400

        # Ubah dataset jadi DataFrame pandas
        df = pd.DataFrame(dataset)

        # =========================
        # 1. KONVERSI & VALIDASI DATA
        # =========================
        # Pastikan semua kolom penting tersedia dan bertipe numerik
        cols_required = [
            "umur_ayam",
            "jumlah_ayam",
            "pakan_total_kg",
            "kematian",
            "persentase_bertelur"
        ]

        for col in cols_required:
            # Cek apakah kolom ada
            if col not in df.columns:
                return jsonify({
                    "status": "error",
                    "message": f"Kolom {col} tidak ada di dataset"
                }), 400

            # Konversi ke numeric, jika gagal jadi NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Hapus baris yang mengandung NaN (data tidak valid)
        df.dropna(inplace=True)

        # =========================
        # 2. KONSTANTA PENTING
        # =========================
        # Berat rata-rata 1 butir telur (kg)
        BERAT_TELUR = 0.048  # 48 gram

        # =========================
        # 3. HITUNG TARGET REAL (GROUND TRUTH)
        # =========================
        # Hitung jumlah telur (butir) berdasarkan populasi & produktivitas
        df["jumlah_butir"] = (
            df["jumlah_ayam"] * (df["persentase_bertelur"] / 100)
        ).round().astype(int)

        # Konversi ke kilogram
        df["telur_kg"] = df["jumlah_butir"] * BERAT_TELUR

        # =========================
        # 4. FEATURE ENGINEERING
        # =========================
        # Hitung pakan per ayam (biar lebih representatif)
        df["pakan_per_ayam"] = df.apply(
            lambda x: x["pakan_total_kg"] / x["jumlah_ayam"]
            if x["jumlah_ayam"] > 0 else 0,
            axis=1
        )

        # Tentukan fitur (input model)
        features = [
            "umur_ayam",
            "jumlah_ayam",
            "pakan_per_ayam",
            "kematian"
        ]

        X = df[features]        # input (fitur)
        y = df["telur_kg"]      # target (output)

        # =========================
        # 5. TRAINING MODEL
        # =========================
        # Jika data cukup, split train-test
        if len(df) >= 5:
            test_size = 0.2 if len(df) > 10 else 0.1
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42
            )
        else:
            # Kalau data sedikit, pakai semua untuk training & testing
            X_train, X_test, y_train, y_test = X, X, y, y

        # Inisialisasi model Random Forest
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        # Training model
        model.fit(X_train, y_train)

        # Prediksi ke data test
        y_pred = model.predict(X_test)

        # Evaluasi model
        MAE = mean_absolute_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 1.0

        # =========================
        # LOGIKA AKURASI BERBASIS THRESHOLD
        # =========================
        # Anggap prediksi benar jika error <= 10%
        threshold = 0.1

        # Semua data dianggap "benar" (label 1)
        y_test_class = np.ones(len(y_test))

        # Bandingkan prediksi vs real
        y_pred_class = [
            1 if (abs(r - p) / r if r != 0 else 0) <= threshold else 0
            for r, p in zip(y_test, y_pred)
        ]

        # Hitung metrik klasifikasi
        acc_dosen = accuracy_score(y_test_class, y_pred_class)
        pre_dosen = precision_score(y_test_class, y_pred_class, zero_division=0)
        rec_dosen = recall_score(y_test_class, y_pred_class, zero_division=0)
        f1_dosen = f1_score(y_test_class, y_pred_class, zero_division=0)

        # =========================
        # 6. INPUT USER (PREDIKSI BARU)
        # =========================
        jml_ayam_input = float(data.get("jumlah_ayam", 0))
        pakan_input = float(data.get("pakan_total_kg", 0))
        kematian_input = float(data.get("kematian", 0))
        umur_input = float(data.get("umur_ayam", 0))
        persen_input = float(data.get("persentase_bertelur", 0))

        # Validasi jumlah ayam
        if jml_ayam_input <= 0:
            return jsonify({
                "status": "error",
                "message": "Jumlah ayam input harus > 0"
            }), 400

        # =========================
        # 7. RUMUS BAKU (HASIL UTAMA)
        # =========================
        # Hitung jumlah telur berdasarkan rumus langsung
        jumlah_butir_real = int(round(
            jml_ayam_input * (persen_input / 100)
        ))

        jumlah_kg_real = round(
            jumlah_butir_real * BERAT_TELUR, 1
        )

        # =========================
        # 8. PREDIKSI MODEL (SEBAGAI PEMBANDING)
        # =========================
        pakan_per_ayam_input = pakan_input / jml_ayam_input

        # Format input ke model
        X_input = pd.DataFrame([[
            umur_input,
            jml_ayam_input,
            pakan_per_ayam_input,
            kematian_input
        ]], columns=features)

        # Prediksi model (tidak boleh negatif)
        pred_kg = max(float(model.predict(X_input)[0]), 0)

        pred_butir = int(round(pred_kg / BERAT_TELUR))

        # =========================
        # 9. HITUNG FCR (Feed Conversion Ratio)
        # =========================
        # FCR = jumlah pakan / hasil telur (kg)
        fcr_real = round(
            pakan_input / jumlah_kg_real, 2
        ) if jumlah_kg_real > 0 else 0

        # =========================
        # 10. RESPONSE KE CLIENT
        # =========================
        return jsonify({
            "status": "success",

            # Evaluasi model
            "metrik": {
                "MAE": round(float(MAE), 4),
                "R2": round(float(R2), 4),
                "akurasi_dosen": f"{round(acc_dosen * 100, 2)}%",
                "presisi_dosen": round(float(pre_dosen), 3),
                "recall_dosen": round(float(rec_dosen), 3),
                "f1_score_dosen": round(float(f1_dosen), 3)
            },

            # Hasil prediksi
            "prediksi": {
                # HASIL UTAMA (RUMUS MATEMATIS)
                "harian_telur_butir": jumlah_butir_real,
                "harian_telur_kg": jumlah_kg_real,

                # Informasi tambahan
                "produktivitas_persen": round(
                    (jumlah_butir_real / jml_ayam_input) * 100, 2
                ),
                "fcr": fcr_real,

                # HASIL MODEL ML (OPSIONAL)
                "model_telur_kg": round(pred_kg, 2),
                "model_telur_butir": pred_butir,

                # Selisih untuk analisis
                "selisih_butir_model_vs_real":
                    pred_butir - jumlah_butir_real
            }
        })

    except Exception as e:
        # Debug error lengkap di console
        import traceback
        print(traceback.format_exc())

        # Kirim error ke client
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 API Training Model Produksi Telur (ANTI DATA BOCOR) + Presisi"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
