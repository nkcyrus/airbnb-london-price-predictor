import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset Airbnb depuis un fichier .csv.gz ou .csv."""
    path = Path(filepath)
    if path.suffix == '.gz':
        df = pd.read_csv(path, compression='gzip', engine='python', quotechar='"')
    else:
        df = pd.read_csv(path, low_memory=False)
    print(f"Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les colonnes utiles pour la modélisation."""
    cols = [
        'neighbourhood_cleansed', 'latitude', 'longitude',
        'property_type', 'room_type', 'accommodates',
        'bathrooms_text', 'bedrooms', 'beds', 'amenities',
        'host_is_superhost', 'host_identity_verified',
        'minimum_nights', 'availability_365',
        'number_of_reviews', 'review_scores_rating',
        'review_scores_cleanliness', 'review_scores_location',
        'price'
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie la colonne price : supprime $, virgules, outliers."""
    df = df.dropna(subset=['price'])
    df['price'] = df['price'].str.replace('$', '', regex=False)
    df['price'] = df['price'].str.replace(',', '', regex=False)
    df['price'] = df['price'].astype(float)
    df = df[df['price'] > 0]
    p99 = df['price'].quantile(0.99)
    df = df[(df['price'] >= 10) & (df['price'] <= p99)]
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Gère les valeurs manquantes selon la stratégie définie."""
    # Supprimer les lignes sans beds
    df = df.dropna(subset=['beds'])

    # Imputer les scores par la moyenne
    score_cols = ['review_scores_rating', 'review_scores_cleanliness', 
                  'review_scores_location']
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Imputer bedrooms par la médiane
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())

    # Supprimer les lignes avec NaN restants
    df = df.dropna(subset=['host_is_superhost', 'bathrooms_text', 
                            'host_identity_verified'])
    return df

def encode_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes t/f en 0/1."""
    bool_cols = ['host_is_superhost', 'host_identity_verified']
    for col in bool_cols:
        if col in df.columns:
            df[col] = (df[col] == 't').astype(int)
    return df

def preprocess(filepath: str) -> pd.DataFrame:
    """Pipeline complet de preprocessing."""
    df = load_data(filepath)
    df = select_columns(df)
    df = clean_price(df)
    df = handle_missing_values(df)
    df = encode_boolean_columns(df)
    print(f"Preprocessing terminé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df