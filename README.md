# 🏠 Airbnb Price Predictor — London

Modèle de Machine Learning qui prédit le prix d'un logement Airbnb à Londres
à partir de ses caractéristiques. Déployé via une API REST et une interface web interactive.

## 📊 Résultats

| Modèle | MAE | R² |
|--------|-----|-----|
| Baseline (médiane) | 95.57£ | -0.07 |
| Régression linéaire | 62.61£ | 0.51 |
| Random Forest | 48.91£ | 0.66 |
| XGBoost | 49.18£ | 0.66 |
| **LightGBM (final)** | **46.65£** | **0.69** |

> Le modèle se trompe en moyenne de **46£** sur le prix d'un logement londonien.

## 🛠️ Stack technique

- **ML** : scikit-learn, LightGBM, XGBoost, SHAP
- **API** : FastAPI, Pydantic, Uvicorn
- **Interface** : Streamlit
- **Infra** : Docker
- **Tests** : pytest

## 🚀 Lancer le projet

### Avec Docker

```bash
docker build -f docker/Dockerfile -t airbnb-price-predictor .
docker run -p 8000:8000 airbnb-price-predictor
```

### En local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn api.main:app --reload

# Lancer l'interface Streamlit (nouveau terminal)
streamlit run app/streamlit_app.py
```

## 📁 Structure

```
airbnb-paris-price-predictor/
├── notebooks/          # EDA, feature engineering, modélisation
├── src/                # Code modulaire (preprocessing, features, train, predict)
├── api/                # FastAPI (endpoints, schemas, model_loader)
├── app/                # Interface Streamlit
├── tests/              # Tests unitaires (18 tests)
├── models/             # Modèle sauvegardé + feature names
├── data/               # Données brutes et traitées
└── docker/             # Dockerfile + docker-compose
```

## 📈 Interprétabilité

Les SHAP values révèlent que les variables les plus influentes sont :
- **`bedrooms`** et **`accommodates`** — la taille du logement
- **`room_type`** — appartement entier vs chambre privée
- **`latitude`/`longitude`** — la localisation précise
- **`neighbourhood_Westminster`** — les quartiers premium

## 📦 Données

Source : [Inside Airbnb](http://insideairbnb.com/get-the-data) — London listings
- 59 767 logements après nettoyage
- 65 features après feature engineering

## 🌐 Demo

- **API** : https://airbnb-london-price-predictor.onrender.com/docs
- **App** : https://airbnb-london-price-predictor-hijgqb7ey8hb2flpdoyove.streamlit.app