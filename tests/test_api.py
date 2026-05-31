import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_listing():
    """Logement exemple pour les tests."""
    return {
        "room_type": 3,
        "accommodates": 4,
        "bedrooms": 2.0,
        "beds": 2.0,
        "bathrooms_count": 1.0,
        "bathrooms_shared": 0,
        "latitude": 51.4975,
        "longitude": -0.1357,
        "neighbourhood": "Westminster",
        "availability_365": 150,
        "minimum_nights": 2,
        "host_is_superhost": 1,
        "host_identity_verified": 1,
        "wifi": 1,
        "kitchen": 1,
        "air_conditioning": 1,
        "heating": 1,
        "washer": 0,
        "dishwasher": 1,
        "gym": 0,
        "hot_tub": 0,
        "pool": 0
    }

# ── Tests endpoints ──────────────────────────────────────────────────────────

def test_root_returns_200():
    """Le endpoint racine doit retourner 200."""
    response = client.get("/")
    assert response.status_code == 200

def test_health_returns_200():
    """Le endpoint health doit retourner 200."""
    response = client.get("/health")
    assert response.status_code == 200

def test_neighbourhoods_returns_list():
    """Le endpoint neighbourhoods doit retourner une liste."""
    response = client.get("/neighbourhoods")
    assert response.status_code == 200
    data = response.json()
    assert "neighbourhoods" in data
    assert isinstance(data["neighbourhoods"], list)
    assert len(data["neighbourhoods"]) > 0

def test_predict_returns_200(sample_listing):
    """Le endpoint predict doit retourner 200."""
    response = client.post("/predict", json=sample_listing)
    assert response.status_code == 200

def test_predict_returns_price(sample_listing):
    """La réponse doit contenir un prix positif."""
    response = client.post("/predict", json=sample_listing)
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0

def test_predict_returns_gbp(sample_listing):
    """La devise doit être GBP."""
    response = client.post("/predict", json=sample_listing)
    data = response.json()
    assert data["currency"] == "GBP"

def test_predict_high_confidence_westminster(sample_listing):
    """Westminster doit retourner une confiance high."""
    response = client.post("/predict", json=sample_listing)
    data = response.json()
    assert data["confidence"] == "high"

def test_predict_low_confidence_unknown_neighbourhood(sample_listing):
    """Un quartier inconnu doit retourner une confiance low."""
    sample_listing["neighbourhood"] = "UnknownPlace"
    response = client.post("/predict", json=sample_listing)
    data = response.json()
    assert data["confidence"] == "low"

def test_predict_invalid_room_type(sample_listing):
    """Un room_type invalide doit retourner 422."""
    sample_listing["room_type"] = 99
    response = client.post("/predict", json=sample_listing)
    assert response.status_code == 422

def test_predict_missing_field(sample_listing):
    """Un champ obligatoire manquant doit retourner 422."""
    del sample_listing["accommodates"]
    response = client.post("/predict", json=sample_listing)
    assert response.status_code == 422