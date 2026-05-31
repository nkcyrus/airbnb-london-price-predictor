from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import ListingFeatures, PredictionResponse
from api.model_loader import predict_price

# Initialiser l'app
app = FastAPI(
    title="Airbnb Price Predictor API",
    description="Prédit le prix d'un logement Airbnb à Londres",
    version="1.0.0"
)

# CORS — permet à Streamlit et d'autres clients d'appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "message": "Airbnb Price Predictor API"}

@app.get("/health")
def health():
    """Vérifie que le modèle est chargé."""
    try:
        from api.model_loader import load_model
        load_model()
        return {"status": "healthy", "model": "LightGBM loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict(listing: ListingFeatures):
    """
    Prédit le prix d'un logement Airbnb.
    
    - **room_type**: 0=Shared, 1=Private, 2=Hotel, 3=Entire home
    - **neighbourhood**: Quartier londonien (ex: Westminster, Hackney)
    - **confidence**: 'high' si quartier connu, 'low' sinon
    """
    try:
        result = predict_price(listing.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/neighbourhoods")
def get_neighbourhoods():
    """Retourne la liste des quartiers supportés."""
    from api.model_loader import load_feature_names
    feature_names = load_feature_names()
    neighbourhoods = [
        col.replace("neighbourhood_", "")
        for col in feature_names
        if col.startswith("neighbourhood_")
    ]
    return {"neighbourhoods": sorted(neighbourhoods)}