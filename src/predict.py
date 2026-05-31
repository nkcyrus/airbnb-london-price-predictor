import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

def load_model():
    """Charge le modèle sauvegardé."""
    return joblib.load(MODEL_DIR / "lightgbm_model.pkl")

def load_feature_names() -> list:
    """Charge la liste des features."""
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        return json.load(f)

def load_metrics() -> dict:
    """Charge les métriques du modèle."""
    with open(MODEL_DIR / "metrics.json", "r") as f:
        return json.load(f)

def predict(features: dict) -> float:
    """
    Prédit le prix d'un logement.
    
    Args:
        features: dictionnaire des caractéristiques du logement
        
    Returns:
        Prix prédit en GBP
    """
    model = load_model()
    feature_names = load_feature_names()

    # Créer DataFrame avec toutes les features à zéro
    df = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Remplir les features disponibles
    for col, val in features.items():
        if col in df.columns:
            df[col] = val

    # Encoder le quartier
    neighbourhood = features.get('neighbourhood', '')
    col_name = f"neighbourhood_{neighbourhood}"
    if col_name in df.columns:
        df[col_name] = 1.0

    price = model.predict(df)[0]
    return max(10.0, round(float(price), 2))

if __name__ == "__main__":
    # Test rapide
    test_listing = {
        'room_type': 3,
        'accommodates': 4,
        'bedrooms': 2,
        'beds': 2,
        'bathrooms_count': 1.0,
        'bathrooms_shared': 0,
        'latitude': 51.4975,
        'longitude': -0.1357,
        'neighbourhood': 'Westminster',
        'availability_365': 150,
        'minimum_nights': 2,
        'host_is_superhost': 1,
        'host_identity_verified': 1,
        'wifi': 1,
        'kitchen': 1,
    }

    price = predict(test_listing)
    print(f"Prix prédit : £{price}")