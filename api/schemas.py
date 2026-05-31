from pydantic import BaseModel, Field
from typing import Optional

class ListingFeatures(BaseModel):
    """Caractéristiques d'un logement Airbnb."""
    
    # Logement
    room_type: int = Field(..., ge=0, le=3, 
                           description="0=Shared, 1=Private, 2=Hotel, 3=Entire home")
    accommodates: int = Field(..., ge=1, le=20)
    bedrooms: float = Field(..., ge=0, le=20)
    beds: float = Field(..., ge=0, le=20)
    bathrooms_count: float = Field(..., ge=0, le=10)
    bathrooms_shared: int = Field(..., ge=0, le=1)
    
    # Localisation
    latitude: float = Field(..., ge=51.3, le=51.7)
    longitude: float = Field(..., ge=-0.5, le=0.3)
    neighbourhood: str = Field(..., description="Nom du quartier londonien")
    
    # Disponibilité
    availability_365: int = Field(..., ge=0, le=365)
    minimum_nights: int = Field(..., ge=1)
    
    # Hôte
    host_is_superhost: int = Field(0, ge=0, le=1)
    host_identity_verified: int = Field(0, ge=0, le=1)
    
    # Équipements
    wifi: int = Field(0, ge=0, le=1)
    kitchen: int = Field(0, ge=0, le=1)
    air_conditioning: int = Field(0, ge=0, le=1)
    heating: int = Field(0, ge=0, le=1)
    washer: int = Field(0, ge=0, le=1)
    dishwasher: int = Field(0, ge=0, le=1)
    gym: int = Field(0, ge=0, le=1)
    hot_tub: int = Field(0, ge=0, le=1)
    pool: int = Field(0, ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "room_type": 3,
                "accommodates": 4,
                "bedrooms": 2,
                "beds": 2,
                "bathrooms_count": 1,
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
        }

class PredictionResponse(BaseModel):
    """Réponse de l'API."""
    predicted_price: float
    currency: str = "GBP"
    confidence: str