import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Sépare les données en train/test."""
    X = df.drop(columns=['price'])
    y = df['price']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test) -> dict:
    """Évalue le modèle et retourne les métriques."""
    y_pred = model.predict(X_test)
    return {
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'r2': round(r2_score(y_test, y_pred), 4)
    }

def train(df: pd.DataFrame) -> LGBMRegressor:
    """Entraîne le modèle LightGBM et sauvegarde les artefacts."""
    # Split
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train : {X_train.shape} | Test : {X_test.shape}")

    # Entraîner
    model = LGBMRegressor(
        n_estimators=500,
        max_depth=-1,
        learning_rate=0.1,
        num_leaves=100,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Évaluer
    metrics = evaluate_model(model, X_test, y_test)
    print(f"MAE  : {metrics['mae']}£")
    print(f"RMSE : {metrics['rmse']}£")
    print(f"R²   : {metrics['r2']}")

    # Sauvegarder le modèle
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "lightgbm_model.pkl")
    print("Modèle sauvegardé ✓")

    # Sauvegarder les features
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)
    print("Features sauvegardées ✓")

    # Sauvegarder les métriques
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Métriques sauvegardées ✓")

    return model

if __name__ == "__main__":
    from src.preprocessing import preprocess
    from src.features import build_features

    print("=== Chargement des données ===")
    df = preprocess(str(DATA_DIR / "raw" / "listings.csv.gz"))

    print("\n=== Feature Engineering ===")
    df = build_features(df)

    print("\n=== Entraînement ===")
    train(df)