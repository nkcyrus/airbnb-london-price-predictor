import pandas as pd
import numpy as np

def engineer_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le nombre et le type de salle de bain depuis bathrooms_text."""
    df['bathrooms_count'] = df['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['bathrooms_shared'] = df['bathrooms_text'].str.contains(
        'shared', case=False, na=False
    ).astype(int)
    df['bathrooms_count'] = df['bathrooms_count'].fillna(df['bathrooms_count'].median())
    df = df.drop(columns=['bathrooms_text'])
    return df

def engineer_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les équipements clés en colonnes binaires."""
    key_amenities = [
        'Wifi', 'Air conditioning', 'Heating',
        'Kitchen', 'Dishwasher', 'Coffee maker',
        'Smoke alarm', 'Lock on bedroom door',
        'Pool', 'Hot tub', 'Gym', 'Elevator',
        'Washer', 'Dryer', 'Iron', 'Dedicated workspace'
    ]
    for amenity in key_amenities:
        col_name = amenity.lower().replace(' ', '_')
        df[col_name] = df['amenities'].str.contains(
            amenity, case=False, regex=False, na=False
        ).astype(int)
    df = df.drop(columns=['amenities'])
    return df

def encode_room_type(df: pd.DataFrame) -> pd.DataFrame:
    """Encode room_type en ordinal."""
    room_type_order = {
        'Shared room': 0,
        'Private room': 1,
        'Hotel room': 2,
        'Entire home/apt': 3
    }
    df['room_type'] = df['room_type'].map(room_type_order)
    return df

def encode_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding sur neighbourhood_cleansed."""
    df = pd.get_dummies(df, columns=['neighbourhood_cleansed'], prefix='neighbourhood')
    neighbourhood_cols = [c for c in df.columns if c.startswith('neighbourhood_')]
    df[neighbourhood_cols] = df[neighbourhood_cols].astype(int)
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles pour la modélisation."""
    cols_to_drop = ['property_type', 'id'] 
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de feature engineering."""
    df = engineer_bathrooms(df)
    df = engineer_amenities(df)
    df = encode_room_type(df)
    df = encode_neighbourhood(df)
    df = drop_unused_columns(df)
    print(f"Feature engineering terminé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df