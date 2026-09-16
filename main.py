from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib


app = FastAPI()

model = joblib.load("model.joblib")                 # laddar min sparad modell


@app.get("/")                                       # skapar min enkla GET-endpoint
def home():
    return {"message": "Model API is running"}


class PredictionInput(BaseModel):
    monthly_spend: float = Field(
        ge=0,                                       # GE = greater than or equal to, LE = less than or equal to; alltså här måste vara ett decimaltal mellan 0 och 5000
        le=5000,
        description="Customer monthly spend"
    )
    months_as_customer: int = Field(
        ge=0,
        le=120,
        description="Number of months as customer"
    )
    support_cases: int = Field(                    # här tex support cases måste vara ett heltal mellan 0 och 50
        ge=0,
        le=50,
        description="Number of support cases"
    )
    usage_frequency: int = Field(
        ge=0,
        le=100,
        description="Usage frequency"
    )


class PredictionOutput(BaseModel):
    prediction: str
    probability: float



# lägger in POST endpointen, nu när GET är klar:
@app.post("/predict", response_model=PredictionOutput)  # när någon skickar en POST-request till /predict så ska funktionen nedan köras
def predict(data: PredictionInput):                     # predict funktion där FastAPI förväntar sig att requests body ska följa min Pydantic-model PredictionImput
    features = [[
    data.monthly_spend,
    data.months_as_customer,
    data.support_cases,
    data.usage_frequency
]]                                                      # alla feature float gör Pydantic om till ett Pythonobjekt som man kan läsa med data.monthly_spend osv. 

    prediction = model.predict(features)[0]             # här tränar jag modellen och gör prediction för datan man skickar in
    probability = model.predict_proba(features)[0].max()        # proba ger sannolikhet för varje klass( för klass 0 och 1, tex. stay 76% och leave 24% ) samt .max tar den högsta av dom

    prediction_label = "leave" if prediction == 1 else "stay"

    return {
        "prediction": prediction_label,
        "probability": float(probability)
    }                                                    # en vanlig python dictionary som FASTAPI gör automatiskt om till JSON


# lägger till en GET/health endpoint:
@app.get("/health")                                      # API:t lever och modellen är redo att användas
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None               # model is not None betyder = finns ett modellobjekt laddat ? Jag bör få ok och true här också som respons
    }