from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional

# 1) Load your trained pipeline once at startup
MODEL_PATH = "models/fraud_detector_tuned_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# Constants for Karnataka-based model
COUNTRIES = ["IN", "US", "CN", "GB", "DE", "FR", "SG", "AU", "CA", "JP"]
INDIAN_STATES = [
    "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Gujarat", 
    "Rajasthan", "West Bengal", "Uttar Pradesh", "Telangana", "Kerala",
    "Andhra Pradesh", "Madhya Pradesh", "Haryana", "Punjab", "Bihar"
]

# Risk classifications
INTERNATIONAL_HIGH_RISK = {"CN", "PK", "BD", "NG", "RU", "IR", "AF"}
INTERNATIONAL_MEDIUM_RISK = {"US", "GB", "CA", "AU", "SG", "DE", "FR"}
HIGH_RISK_STATES = {"West Bengal", "Bihar", "Uttar Pradesh", "Rajasthan"}
MEDIUM_RISK_STATES = {"Delhi", "Maharashtra", "Gujarat", "Punjab"}
NEIGHBORING_STATES = {"Tamil Nadu", "Andhra Pradesh", "Telangana", "Kerala", "Goa"}

DEVICES = ["Android", "iOS", "Windows", "macOS"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]

# 2) Define the input schema with Karnataka-specific validation
class TransactionIn(BaseModel):
    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID")
    user_id: int = Field(..., gt=0, description="User ID")
    amount: float = Field(..., gt=0.0, description="Transaction amount in INR")
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp (current time if not provided)")
    
    country: str = Field(
        ...,
        description="Country code (IN for India, or international country codes)"
    )
    
    state: Optional[str] = Field(
        None,
        description="State name (required if country is IN, null for international)"
    )
    
    device: str = Field(
        ...,
        description="Device type used for transaction"
    )
    
    payment_method: str = Field(
        ...,
        description="Payment method used"
    )
    
    prev_transactions_24h: int = Field(..., ge=0, description="Number of transactions in last 24 hours")
    avg_amount_30d: float = Field(..., ge=0.0, description="Average transaction amount in last 30 days")
    
    @validator('country')
    def validate_country(cls, v):
        if v not in COUNTRIES:
            raise ValueError(f'Country must be one of: {", ".join(COUNTRIES)}')
        return v
    
    @validator('state')
    def validate_state(cls, v, values):
        if 'country' in values and values['country'] == 'IN':
            if v is None:
                raise ValueError('State is required when country is IN')
            if v not in INDIAN_STATES:
                raise ValueError(f'State must be one of: {", ".join(INDIAN_STATES)}')
        elif v is not None and values.get('country') != 'IN':
            raise ValueError('State should be null for international transactions')
        return v
    
    @validator('device')
    def validate_device(cls, v):
        if v not in DEVICES:
            raise ValueError(f'Device must be one of: {", ".join(DEVICES)}')
        return v
    
    @validator('payment_method')
    def validate_payment_method(cls, v):
        if v not in PAYMENT_METHODS:
            raise ValueError(f'Payment method must be one of: {", ".join(PAYMENT_METHODS)}')
        return v

app = FastAPI(
    title="Karnataka-Based Fraud Detection API",
    description="Fraud detection API with Karnataka as base state",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced feature engineering with Karnataka-based risk scoring."""
    
    # Basic time features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
    df['is_night'] = df['hour_of_day'].between(0, 6).astype(int)
    df['is_suspicious_time'] = df['hour_of_day'].between(1, 5).astype(int)
    
    # Karnataka-based location features
    base_state = "Karnataka"
    base_country = "IN"
    
    df['is_local'] = ((df['country'] == base_country) & (df['state'] == base_state)).astype(int)
    df['is_out_of_state'] = ((df['country'] == base_country) & (df['state'] != base_state)).astype(int)
    df['is_international'] = (df['country'] != base_country).astype(int)
    
    # State risk levels
    def get_state_risk_level(row):
        if row['country'] != 'IN' or pd.isna(row['state']):
            return 0
        state = row['state']
        if state in HIGH_RISK_STATES:
            return 3
        elif state in MEDIUM_RISK_STATES:
            return 2
        elif state in NEIGHBORING_STATES:
            return 1
        elif state == base_state:
            return 0
        else:
            return 1  # Default for other states
    
    df['state_risk_level'] = df.apply(get_state_risk_level, axis=1)
    
    # International risk levels
    def get_international_risk_level(row):
        if row['country'] == 'IN':
            return 0
        country = row['country']
        if country in INTERNATIONAL_HIGH_RISK:
            return 4
        elif country in INTERNATIONAL_MEDIUM_RISK:
            return 2
        else:
            return 3
    
    df['international_risk_level'] = df.apply(get_international_risk_level, axis=1)
    
    # Amount-based features
    df['amount_diff_from_avg'] = df['amount'] - df['avg_amount_30d']
    df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
    df['is_high_amount'] = (df['amount'] > 50000).astype(int)
    
    # Payment method risk features
    df['is_risky_payment_intl'] = (
        (df['is_international'] == 1) & 
        df['payment_method'].isin(['UPI', 'Net Banking'])
    ).astype(int)
    
    df['is_credit_card_intl'] = (
        (df['is_international'] == 1) & 
        (df['payment_method'] == 'Credit Card')
    ).astype(int)
    
    # Device-payment compatibility
    df['is_upi_desktop'] = (
        (df['payment_method'] == 'UPI') & 
        df['device'].isin(['Windows', 'macOS'])
    ).astype(int)
    
    # Velocity features
    df['high_velocity'] = (df['prev_transactions_24h'] > 5).astype(int)
    df['very_high_velocity'] = (df['prev_transactions_24h'] > 8).astype(int)
    
    # Comprehensive fraud score calculation
    def calculate_fraud_score(row):
        score = 0
        
        # Location-based risk
        if row['is_international']:
            score += row['international_risk_level']
            score += 3  # Base international penalty
        elif row['is_out_of_state']:
            score += row['state_risk_level']
            score += 1  # Base out-of-state penalty
        
        # Amount-based risk
        if row['country'] == 'IN':
            if row['amount'] > 100000:  # ₹1 Lakh+
                score += 4
            elif row['amount'] > 50000:  # ₹50k+
                score += 2
            elif row['amount'] > 25000:  # ₹25k+
                score += 1
        else:
            if row['amount'] > 200000:  # ₹2 Lakh+ for international
                score += 3
            elif row['amount'] > 100000:
                score += 2
            elif row['amount'] > 50000:
                score += 1
        
        # Velocity-based risk
        if row['prev_transactions_24h'] > 8:
            score += 3
        elif row['prev_transactions_24h'] > 5:
            score += 2
        elif row['prev_transactions_24h'] > 3:
            score += 1
        
        # Amount anomaly
        if row['amount_to_avg_ratio'] > 5:
            score += 3
        elif row['amount_to_avg_ratio'] > 3:
            score += 2
        elif row['amount_to_avg_ratio'] > 2:
            score += 1
        
        # Time-based risk
        if row['is_suspicious_time']:
            score += 2
        elif row['is_night']:
            score += 1
        
        # Payment method risk
        if row['is_risky_payment_intl']:
            score += 2
        elif row['is_credit_card_intl']:
            score += 1
        
        # Device risk
        if row['is_upi_desktop']:
            score += 1
        
        # Weekend international
        if row['is_weekend'] and row['is_international']:
            score += 1
        
        return score
    
    df['fraud_score'] = df.apply(calculate_fraud_score, axis=1)
    df['risk_score'] = df['fraud_score'] * 10  # Normalized to 0-100 scale
    df['risk_score'] = df['risk_score'].clip(0, 100)
    
    # Legacy features for backward compatibility
    df['is_foreign'] = df['is_international']
    df['is_high_risk_country'] = (
        df['country'].isin(INTERNATIONAL_HIGH_RISK) | 
        ((df['country'] == 'IN') & df['state'].isin(HIGH_RISK_STATES))
    ).astype(int)
    
    return df

@app.post("/predict")
async def predict(txn: TransactionIn):
    """Predict fraud probability for a transaction."""
    try:
        # 3) Build DataFrame from the validated input
        data = txn.dict()
        
        # If timestamp is None, use current time
        if data['timestamp'] is None:
            data['timestamp'] = datetime.now()
        
        df = pd.DataFrame([data])
        
        # 4) Feature engineering
        df = engineer_features(df)
        
        # 5) Prepare features (remove non-feature columns)
        feature_columns = [col for col in df.columns if col not in [
            'transaction_id', 'timestamp', 'state'  # Keep essential columns, remove identifiers
        ]]
        X = df[feature_columns]
        
        # 6) Model inference
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0, 1])
        
        # 7) Enhanced response with Karnataka-specific insights
        location = f"{txn.state}, India" if txn.country == "IN" else txn.country
        risk_level = "Low"
        if df['risk_score'].iloc[0] > 70:
            risk_level = "High"
        elif df['risk_score'].iloc[0] > 30:
            risk_level = "Medium"
        
        return {
            "transaction_id": txn.transaction_id,
            "is_fraud": bool(pred),
            "fraud_probability": round(proba, 4),
            "risk_score": int(df['risk_score'].iloc[0]),
            "risk_level": risk_level,
            "location": location,
            "is_local": bool(df['is_local'].iloc[0]),
            "is_out_of_state": bool(df['is_out_of_state'].iloc[0]),
            "is_international": bool(df['is_international'].iloc[0]),
            "fraud_score_components": {
                "location_risk": int(df['state_risk_level'].iloc[0] + df['international_risk_level'].iloc[0]),
                "amount_risk": int(df['is_high_amount'].iloc[0]),
                "velocity_risk": int(df['high_velocity'].iloc[0]),
                "time_risk": int(df['is_night'].iloc[0]),
                "payment_risk": int(df['is_risky_payment_intl'].iloc[0])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    """API information."""
    return {
        "message": "Karnataka-Based Fraud Detection API",
        "version": "2.0.0",
        "base_location": "Karnataka, India",
        "supported_countries": COUNTRIES,
        "supported_states": INDIAN_STATES,
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

# Example usage endpoint for testing
@app.get("/example")
async def get_example_requests():
    """Get example request payloads for testing."""
    return {
        "local_transaction": {
            "transaction_id": "TXN123456789",
            "user_id": 1234567,
            "amount": 5000.0,
            "country": "IN",
            "state": "Karnataka",
            "device": "Android",
            "payment_method": "UPI",
            "prev_transactions_24h": 2,
            "avg_amount_30d": 3000.0
        },
        "out_of_state_transaction": {
            "transaction_id": "TXN987654321",
            "user_id": 1234567,
            "amount": 25000.0,
            "country": "IN",
            "state": "Maharashtra",
            "device": "iOS",
            "payment_method": "Credit Card",
            "prev_transactions_24h": 1,
            "avg_amount_30d": 8000.0
        },
        "international_transaction": {
            "transaction_id": "TXN555666777",
            "user_id": 1234567,
            "amount": 75000.0,
            "country": "US",
            "state": None,
            "device": "Windows",
            "payment_method": "Credit Card",
            "prev_transactions_24h": 0,
            "avg_amount_30d": 15000.0
        }
    }