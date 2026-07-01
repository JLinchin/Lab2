import joblib
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

app = FastAPI(
    title="API Housing",
    description="Prédiction du prix médian d'un logement (California Housing)."
)
modele = joblib.load("model.joblib")

class Features(BaseModel):
    MedInc: float = Field(default=8.3)
    HouseAge: float = Field(default=41)
    AveRooms: float = Field(default=6.9)
    AveBedrms: float = Field(default=1)
    Population: float = Field(default=322)
    AveOccup: float = Field(default=2.5)
    Latitude: float = Field(default=37.88)
    Longitude: float = Field(default=-122.23)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Features = Body(default_factory=Features)):
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