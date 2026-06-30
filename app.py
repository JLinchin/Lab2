import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
modele = joblib.load("model.joblib")

class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Features):
    X = [[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]]
    prediction = modele.predict(X)[0]
    return {"valeur prédite": round(float(prediction), 2)}