from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ============================================================
# 1. LOAD MODEL
# ============================================================

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# 2. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PhishGuard API",
    description="API for detecting phishing websites",
    version="1.0.0"
)


# ============================================================
# 3. INPUT SCHEMA
# ============================================================

class URLFeatures(BaseModel):

    having_IP_Address: int
    URL_Length: int
    Shortining_Service: int
    having_At_Symbol: int
    double_slash_redirecting: int
    Prefix_Suffix: int
    having_Sub_Domain: int
    SSLfinal_State: int
    Domain_registeration_length: int
    Favicon: int
    port: int
    HTTPS_token: int
    Request_URL: int
    URL_of_Anchor: int
    Links_in_tags: int
    SFH: int
    Submitting_to_email: int
    Abnormal_URL: int
    Redirect: int
    on_mouseover: int
    RightClick: int
    popUpWidnow: int
    Iframe: int
    age_of_domain: int
    DNSRecord: int
    web_traffic: int
    Page_Rank: int
    Google_Index: int
    Links_pointing_to_page: int
    Statistical_report: int


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "PhishGuard API is running",
        "status": "healthy"
    }


# ============================================================
# 5. PHISHING PREDICTION
# ============================================================

@app.post("/predict")
def predict(features: URLFeatures):

    # Convert request data to dictionary
    data = features.model_dump()

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Convert model output to readable result
    if prediction == 1:

        result = "Legitimate"

    else:

        result = "Phishing"

    return {
        "prediction": int(prediction),
        "result": result
    }