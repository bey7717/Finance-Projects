from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel
import joblib
import numpy

model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')


class BaselineModel(BaseModel):
    Car_Owner: int
    Propert_Owner: int
    CHILDREN: int
    EDUCATION: int
    Annual_income: float
    age: int
    Employed_years: float

app = FastAPI()

@app.get("/")
def check():
    return {"status": "online", "model": "RandomForest"}


# features = ["Car_Owner", "Propert_Owner", "CHILDREN", "EDUCATION", "Annual_income", "age", "Employed_years"]


@app.post("/predict")
def predict(data: BaselineModel):
    features = [[
        data.Car_Owner,
        data.Propert_Owner,
        data.CHILDREN,
        data.EDUCATION,
        data.Annual_income,
        data.age,
        data.Employed_years
    ]]
    scaler_input = scaler.transform(features)
    proba = model.predict_proba(scaler_input)[0][1]

    decision = "Rejected" if proba >= 0.15 else "Approved"

    return {
        "decision": decision,
        "risk_probablity": round(float(proba), 4),
        "explanation": "High risk detected" if decision == "Rejected" else "Criteria met"
    }
