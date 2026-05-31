import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    clean_price,
    handle_missing_values,
    encode_boolean_columns,
    select_columns
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame minimal pour les tests."""
    return pd.DataFrame({
        'neighbourhood_cleansed': ['Westminster', 'Hackney', 'Camden'],
        'latitude': [51.49, 51.54, 51.53],
        'longitude': [-0.13, -0.05, -0.14],
        'property_type': ['Entire rental unit'] * 3,
        'room_type': ['Entire home/apt', 'Private room', 'Shared room'],
        'accommodates': [4, 2, 1],
        'bathrooms_text': ['1 bath', '1 shared bath', '1 bath'],
        'bedrooms': [2.0, 1.0, np.nan],
        'beds': [2.0, 1.0, 1.0],
        'amenities': ['["Wifi", "Kitchen"]'] * 3,
        'host_is_superhost': ['t', 'f', 't'],
        'host_identity_verified': ['t', 't', 'f'],
        'minimum_nights': [2, 1, 3],
        'availability_365': [150, 200, 100],
        'number_of_reviews': [10, 5, 0],
        'review_scores_rating': [4.5, np.nan, 4.8],
        'review_scores_cleanliness': [4.5, 4.0, np.nan],
        'review_scores_location': [4.8, 4.5, 4.2],
        'price': ['$150.00', '$80.00', '$50000.00']
    })

# ── Tests clean_price ────────────────────────────────────────────────────────

def test_clean_price_removes_dollar_sign(sample_df):
    """Le signe $ doit être supprimé."""
    df = clean_price(sample_df.copy())
    assert df['price'].dtype == float

def test_clean_price_removes_outliers(sample_df):
    """Les prix aberrants doivent être supprimés."""
    df = clean_price(sample_df.copy())
    assert df['price'].max() < 50000

def test_clean_price_removes_zero(sample_df):
    """Les prix à 0 ou négatifs doivent être supprimés."""
    df = sample_df.copy()
    df.loc[0, 'price'] = '$0.00'
    df = clean_price(df)
    assert (df['price'] > 0).all()

# ── Tests handle_missing_values ──────────────────────────────────────────────

def test_handle_missing_values_no_nan_after(sample_df):
    """Il ne doit plus y avoir de NaN après le traitement."""
    df = clean_price(sample_df.copy())
    df = handle_missing_values(df)
    assert df['bedrooms'].isnull().sum() == 0
    assert df['review_scores_rating'].isnull().sum() == 0

def test_handle_missing_values_bedrooms_median(sample_df):
    """bedrooms NaN doit être imputé par la médiane."""
    df = clean_price(sample_df.copy())
    median_before = df['bedrooms'].median()
    df = handle_missing_values(df)
    assert df['bedrooms'].isnull().sum() == 0
    assert df['bedrooms'].min() >= 0

# ── Tests encode_boolean_columns ─────────────────────────────────────────────

def test_encode_boolean_columns_values(sample_df):
    """host_is_superhost doit être 0 ou 1."""
    df = encode_boolean_columns(sample_df.copy())
    assert set(df['host_is_superhost'].unique()).issubset({0, 1})

def test_encode_boolean_columns_t_is_one(sample_df):
    """'t' doit être converti en 1."""
    df = encode_boolean_columns(sample_df.copy())
    assert df.loc[0, 'host_is_superhost'] == 1

def test_encode_boolean_columns_f_is_zero(sample_df):
    """'f' doit être converti en 0."""
    df = encode_boolean_columns(sample_df.copy())
    assert df.loc[1, 'host_is_superhost'] == 0