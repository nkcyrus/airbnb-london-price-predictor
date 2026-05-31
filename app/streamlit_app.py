import streamlit as st
import requests
import json

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Airbnb Price Predictor — London")
st.markdown("Estime le prix d'un logement Airbnb à Londres grâce au Machine Learning.")

# Charger les quartiers depuis l'API
@st.cache_data
def get_neighbourhoods():
    try:
        response = requests.get(f"{API_URL}/neighbourhoods")
        return response.json()["neighbourhoods"]
    except:
        return ["Westminster", "Hackney", "Camden", "Southwark"]

neighbourhoods = get_neighbourhoods()

# Formulaire
st.subheader("Caractéristiques du logement")

col1, col2 = st.columns(2)

with col1:
    room_type = st.selectbox(
        "Type de logement",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Chambre partagée",
            1: "Chambre privée",
            2: "Chambre d'hôtel",
            3: "Appartement entier"
        }[x]
    )
    bedrooms = st.number_input("Chambres", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    beds = st.number_input("Lits", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
    accommodates = st.number_input("Capacité (personnes)", min_value=1, max_value=20, value=2)

with col2:
    neighbourhood = st.selectbox("Quartier", options=neighbourhoods)
    bathrooms_count = st.number_input("Salles de bain", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    bathrooms_shared = st.selectbox("Salle de bain", options=[0, 1],
                                     format_func=lambda x: "Privée" if x == 0 else "Partagée")
    availability_365 = st.slider("Disponibilité (jours/an)", 0, 365, 180)

st.subheader("Équipements")

col3, col4, col5 = st.columns(3)
with col3:
    wifi = int(st.checkbox("Wifi", value=True))
    kitchen = int(st.checkbox("Cuisine", value=True))
    heating = int(st.checkbox("Chauffage", value=True))

with col4:
    air_conditioning = int(st.checkbox("Climatisation"))
    washer = int(st.checkbox("Lave-linge"))
    dishwasher = int(st.checkbox("Lave-vaisselle"))

with col5:
    gym = int(st.checkbox("Salle de sport"))
    hot_tub = int(st.checkbox("Jacuzzi"))
    pool = int(st.checkbox("Piscine"))

st.subheader("Hôte")
col6, col7 = st.columns(2)
with col6:
    host_is_superhost = int(st.checkbox("Superhost"))
with col7:
    host_identity_verified = int(st.checkbox("Identité vérifiée", value=True))

minimum_nights = st.number_input("Nuits minimum", min_value=1, max_value=365, value=2)

# Coordonnées par défaut selon le quartier
coords = {
    "Westminster": (51.4975, -0.1357),
    "Hackney": (51.5450, -0.0553),
    "Camden": (51.5390, -0.1425),
    "Kensington and Chelsea": (51.5020, -0.1947),
    "Tower Hamlets": (51.5099, -0.0059),
    "Southwark": (51.5030, -0.0800),
}
lat, lon = coords.get(neighbourhood, (51.5074, -0.1278))

# Prédiction
if st.button("🔮 Estimer le prix", type="primary"):
    payload = {
        "room_type": room_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms_count": bathrooms_count,
        "bathrooms_shared": bathrooms_shared,
        "latitude": lat,
        "longitude": lon,
        "neighbourhood": neighbourhood,
        "availability_365": availability_365,
        "minimum_nights": minimum_nights,
        "host_is_superhost": host_is_superhost,
        "host_identity_verified": host_identity_verified,
        "wifi": wifi,
        "kitchen": kitchen,
        "air_conditioning": air_conditioning,
        "heating": heating,
        "washer": washer,
        "dishwasher": dishwasher,
        "gym": gym,
        "hot_tub": hot_tub,
        "pool": pool
    }

    with st.spinner("Calcul en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            st.success("Estimation réalisée !")
            st.metric(
                label="Prix estimé par nuit",
                value=f"£{result['predicted_price']:.0f}",
            )

            confidence_color = "🟢" if result['confidence'] == 'high' else "🟡"
            st.caption(f"{confidence_color} Confiance : {result['confidence']}")

        except Exception as e:
            st.error(f"Erreur : {str(e)}")

st.markdown("---")
st.caption("Modèle : LightGBM • Données : Inside Airbnb London • R² : 0.69")