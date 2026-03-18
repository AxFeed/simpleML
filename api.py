from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model   = joblib.load('models/model_AB.pkl')
cl_ab   = joblib.load('models/clustering_ab.pkl')
cl_aby  = joblib.load('models/clustering_aby.pkl')


class InputPredict(BaseModel):
    A: float
    B: float

class InputClasseAB(BaseModel):
    A: float
    B: float

class InputClasseABY(BaseModel):
    A: float
    B: float
    Y: float


@app.post('/predict')
def predict(data: InputPredict):
    X = np.array([[data.A, data.B]])
    y_pred = model.predict(X)[0]
    return {'Y_pred': round(float(y_pred), 4)}


@app.post('/classe')
def classe_ab(data: InputClasseAB):
    X = np.array([[data.A, data.B]])
    X_scaled = cl_ab['scaler'].transform(X)
    cluster  = int(cl_ab['kmeans'].predict(X_scaled)[0])
    return {'classe': cluster}


@app.post('/classe_aby')
def classe_aby(data: InputClasseABY):
    X = np.array([[data.A, data.B, data.Y]])
    X_scaled = cl_aby['scaler'].transform(X)
    cluster  = int(cl_aby['kmeans'].predict(X_scaled)[0])
    return {'classe': cluster}