import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

def load_model():
    """Charge le modèle LightGBM sauvegardé."""
    model = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
    return model

def load_feature_names():
    """Charge la liste des features attendues par le modèle."""
    with open(MODEL_DIR / "feature_names.json", "r") as f:
        return json.load(f)

def prepare_features(listing: dict, feature_names: list) -> pd.DataFrame:
    """
    Transforme les données brutes de l'API 
    en DataFrame prêt pour le modèle.
    """
    # Créer un DataFrame vide avec toutes les features
    df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # Remplir les features numériques directement
    direct_features = [
        'room_type', 'accommodates', 'bedrooms', 'beds',
        'bathrooms_count', 'bathrooms_shared', 'latitude', 'longitude',
        'availability_365', 'minimum_nights', 'host_is_superhost',
        'host_identity_verified', 'wifi', 'kitchen', 'air_conditioning',
        'heating', 'washer', 'dishwasher', 'gym', 'hot_tub', 'pool'
    ]
    
    for feature in direct_features:
        if feature in listing and feature in df.columns:
            df[feature] = listing[feature]
    
    # Encoder le quartier (one-hot)
    neighbourhood = listing.get('neighbourhood', '')
    col_name = f"neighbourhood_{neighbourhood}"
    if col_name in df.columns:
        df[col_name] = 1.0
    
    return df

def predict_price(listing: dict) -> dict:
    """Prédit le prix d'un logement."""
    model = load_model()
    feature_names = load_feature_names()
    
    # Préparer les features
    df = prepare_features(listing, feature_names)
    
    # Prédire
    price = model.predict(df)[0]
    price = max(10, round(float(price), 2))
    
    # Définir la confiance selon la localisation
    neighbourhood = listing.get('neighbourhood', '')
    known_neighbourhoods = [
        col.replace('neighbourhood_', '') 
        for col in feature_names 
        if col.startswith('neighbourhood_')
    ]
    
    if neighbourhood in known_neighbourhoods:
        confidence = "high"
    else:
        confidence = "low"
    
    return {
        "predicted_price": price,
        "currency": "GBP",
        "confidence": confidence
    }