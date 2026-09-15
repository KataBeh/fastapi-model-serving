from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()

model = joblib.load("model.joblib")                 # laddar min sparad modell


@app.get("/")                                       # skapar min enkla GET-endpoint
def home():
    return {"message": "Model API is running"}

class PredictionInput(BaseModel):                   # för att få göra en prediction måste användaren skicka in fyra numeriska värden
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float

# lägger in POST endpointen, nu när GET är klar:
@app.post("/predict")                               # när någon skickar en POST-request till /predict så ska funktionen nedan köras
def predict(data: PredictionInput):                 # predict funktion där FastAPI förväntar sig att requests body ska följa min Pydantic-model PredictionImput
    features = [[
        data.feature_1,
        data.feature_2,
        data.feature_3,
        data.feature_4
    ]]                                              # alla feature float gör Pydantic om till ett Pythonobjekt som man kan läsa med data.feature_1 osv. 

    prediction = model.predict(features)[0]         # här tränar jag modellen och gör prediction för datan man skickar in
    probability = model.predict_proba(features)[0].max()        # proba ger sannolikhet för varje klass( för klass 0 och 1, tex. stay 76% och leave 24% ) samt .max tar den högsta av dom

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }                                               # en vanlig python dictionary som FASTAPI gör automatiskt om till JSON