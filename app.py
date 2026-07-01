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

def to_row(f: Features) -> list:
    return [f.MedInc, f.HouseAge, f.AveRooms, f.AveBedrms, f.Population, f.AveOccup, f.Latitude, f.Longitude]

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

@app.post("/predict_batch")
def predict_batch(features: list[Features]):
    X = [to_row(f) for f in features]
    predictions = modele.predict(X)
    return {"predicted_house_values": [round(float(p), 2) for p in predictions]}
