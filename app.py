from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from datetime import datetime

# 1) Load your trained pipeline once at startup
MODEL_PATH = "models/fraud_detector_tuned_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# 2) Define the input schema using Field for validation
class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    user_id: int
    amount: float
    timestamp: datetime | None = None
    
    country: str = Field(
        ...,
        pattern="^(US|IN|CN|GB|DE|FR)$",
        description="One of US, IN, CN, GB, DE, FR"
    )
    device: str = Field(
        ...,
        pattern="^(Android|iOS|Windows|macOS)$",
        description="One of Android, iOS, Windows, macOS"
    )
    
    prev_transactions_24h: int = Field(..., ge=0)
    avg_amount_30d: float = Field(..., ge=0.0)

app = FastAPI(title="Fraud Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce feature engineering from training pipeline."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night'] = df['hour_of_day'].between(0,6).astype(int)
    df['is_foreign'] = (df['country'] != "US").astype(int)
    df['is_high_risk_country'] = df['country'].isin(["CN","IN"]).astype(int)
    df['amount_diff_from_avg'] = df['amount'] - df['avg_amount_30d']
    df['risk_score'] = (
        df['is_foreign'] * 1.5 +
        df['is_high_risk_country'] * 2.0 +
        df['is_night'] * 1.2 +
        df['amount_diff_from_avg'].abs() / 50
    )
    return df

@app.post("/predict")
async def predict(txn: TransactionIn):
    # 3) Build DataFrame from the validated input
    data = txn.dict()
    # If timestamp is None, use current time
    if data['timestamp'] is None:
        data['timestamp'] = datetime.now()
    df = pd.DataFrame([data])
    
    # 4) Feature engineering
    df = engineer_features(df)
    
    # 5) Prepare features
    X = df.drop(columns=['transaction_id','timestamp'], errors='ignore')
    
    # 6) Model inference
    try:
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # 7) Return result
    return {
        "transaction_id": txn.transaction_id,
        "is_fraud": bool(pred),
        "fraud_probability": round(proba, 4)
    }