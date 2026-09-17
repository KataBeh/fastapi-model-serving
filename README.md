# FastAPI Model Serving

Det här projektet visar hur en tränad maskininlärningsmodell kan göras tillgänglig genom ett REST API med hjälp av FastAPI.

Modellen är en enkel Logistic Regression-modell som tränas på syntetisk kunddata. API:t tar emot kundinformation, validerar datan med Pydantic och returnerar en prediction tillsammans med en sannolikhet.

## Vad projektet gör

Projektet består av två huvuddelar:

- `train_model.py` skapar syntetisk kunddata, tränar en Logistic Regression-modell, utvärderar modellen och sparar den tränade modellen som `model.joblib`.
- `main.py` laddar den sparade modellen och innehåller API-endpoints byggda med FastAPI.

API:t innehåller:

- `GET /` – kontrollerar att API:t körs.
- `GET /health` – kontrollerar att tjänsten är igång och att modellen är laddad.
- `POST /predict` – tar emot kunddata och returnerar en prediction, `stay` eller `leave`, tillsammans med en sannolikhet.

## Installation

Skapa och aktivera en virtuell miljö.

På Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

senare installera projektets beroenden :

```bash
python -m pip install -r requirements.txt
```

## Hur projektet körs

Träna och spara först modellen:

```bash
python train_model.py
```

detta skapar filen **model.joblib**

Starta sedan FastAPI-applikationen:

```bash
fastapi dev main.py
```

Öppna därefter API-dokumentationen i webbläsaren:
http://127.0.0.1:8000/docs

Endpointen `/predict` kan testas direkt i Swagger UI genom att använda **Try it out**


### Exempel på input
{
  "monthly_spend": 400,
  "months_as_customer": 48,
  "support_cases": 1,
  "usage_frequency": 25
}

### Exempel på response
{
  "prediction": "stay",
  "probability": 0.99
}

Viktigt att nämna att eftersom datan är syntetiskt genererad utifrån ett tydligt churn-samband blir vissa predictioner väldigt säkra. Syftet med projektet enligt mig är inte att bygga en realistisk churnmodell, utan att demonstrera model serving med FastAPI.


## Beroenden

De viktigaste beroendena i projektet är:

- FastAPI
- Pydantic
- scikit-learn
- NumPy
- joblib

Alla paket som behövs finns listade i `requirements.txt`

## Data

Projektet använder syntetisk kunddata som skapas direkt i Python.

Variablerna är:

- monthly_spend
- months_as_customer
- support_cases
- usage_frequency

Target representerar om en kund klassificeras som **stay** eller **leave**.

Datan är skapad endast för demonstration och representerar inte riktiga kunder eller verkliga affärsdata.

## Externa tjänster

Projektet använder inga externa API:er, databaser, molntjänster eller andra externa tjänster.

Fokus ligger på FastAPI, datavalidering och model serving.

------------------------------------------------------------------------------