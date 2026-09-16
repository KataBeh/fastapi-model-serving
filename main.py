from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib


app = FastAPI()

model = joblib.load("model.joblib")                 # laddar min sparad modell


@app.get("/")                                       # skapar min enkla GET-endpoint
def home():
    return {"message": "Model API is running"}

class PredictionInput(BaseModel):
    monthly_spend: float
    months_as_customer: int
    support_cases: int
    usage_frequency: int

# lägger in POST endpointen, nu när GET är klar:
@app.post("/predict")                               # när någon skickar en POST-request till /predict så ska funktionen nedan köras
def predict(data: PredictionInput):                 # predict funktion där FastAPI förväntar sig att requests body ska följa min Pydantic-model PredictionImput
    features = [[
    data.monthly_spend,
    data.months_as_customer,
    data.support_cases,
    data.usage_frequency
]]                                                  # alla feature float gör Pydantic om till ett Pythonobjekt som man kan läsa med data.monthly_spend osv. 

    prediction = model.predict(features)[0]         # här tränar jag modellen och gör prediction för datan man skickar in
    probability = model.predict_proba(features)[0].max()        # proba ger sannolikhet för varje klass( för klass 0 och 1, tex. stay 76% och leave 24% ) samt .max tar den högsta av dom

    prediction_label = "leave" if prediction == 1 else "stay"

    return {
        "prediction": prediction_label,
        "probability": float(probability)
    }                                               # en vanlig python dictionary som FASTAPI gör automatiskt om till JSON