import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(dataset, training_params):
    """
    dataset: list of dict (dari Laravel)
    training_params: dict (n_estimators, random_state, max_depth)
    """

    # =====================
    # 1. Dataset → DataFrame
    # =====================
    df = pd.DataFrame(dataset)

    # sesuaikan kolom target
    y = df["telur_kg"]      # atau telur_butir
    X = df.drop([
        "telur_kg",
        "created_at",
        "updated_at",
        "catatan"
    ], axis=1, errors="ignore")

    # =====================
    # 2. Training params
    # =====================
    n_estimators = training_params.get("n_estimators", 100)
    random_state = training_params.get("random_state", 42)
    max_depth = training_params.get("max_depth")

    # =====================
    # 3. Split data
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    # =====================
    # 4. Model
    # =====================
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    # =====================
    # 5. Evaluasi
    # =====================
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # =====================
    # 6. Save model
    # =====================
    with open("model_telur.pkl", "wb") as f:
        pickle.dump(model, f)

    # =====================
    # 7. Return hasil
    # =====================
    return {
        "status": "success",
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }
