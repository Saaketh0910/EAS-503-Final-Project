from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import os
import joblib
from typing import Literal
port = int(os.environ.get("PORT", 8080))
app = FastAPI()
model = joblib.load('bike_buyers_model.joblib')
class BikeBuyerInput(BaseModel):
    ID: int
    Marital_Status: Literal['Single', 'Married']
    Gender: Literal['Male', 'Female']
    Income: float = Field(..., ge=0)
    Children: int = Field(..., ge=0)
    Education: Literal['High School', 'Partial College', 'Bachelors', 'Graduate Degree']
    Occupation: Literal['Manual', 'Skilled Manual', 'Clerical', 'Professional', 'Management']
    Home_Owner: Literal['Yes', 'No']
    Cars: int = Field(..., ge=0)
    Commute_Distance: Literal['0-1 Miles', '1-2 Miles', '2-5 Miles', '5-10 Miles', '10+ Miles']
    Region: Literal['Europe', 'Pacific', 'North America']
    Age: int = Field(..., ge=0)
@app.post("/predict")
async def predict(input: BikeBuyerInput):
    try:
        features = np.array([[
            input.Income, input.Children, input.Cars, input.Age,
            input.Marital_Status, input.Gender, input.Education,
            input.Occupation, input.Home_Owner, input.Commute_Distance, input.Region
        ]])
        prediction = model.predict(features)
        return {"prediction": bool(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Bike Buyers Prediction API"}