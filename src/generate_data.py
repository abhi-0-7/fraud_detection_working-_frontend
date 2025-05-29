import random
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np  # Added for Poisson distribution

# Constants
COUNTRIES = ["IN", "US", "CN", "GB", "DE", "FR", "SG", "AU", "CA", "JP"]
INTERNATIONAL_HIGH_RISK = {"CN", "PK", "BD", "NG", "RU", "IR", "AF"}
INTERNATIONAL_MEDIUM_RISK = {"US", "GB", "CA", "AU", "SG", "DE", "FR"}

# Indian States (Karnataka as base)
INDIAN_STATES = [
    "Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Gujarat", 
    "Rajasthan", "West Bengal", "Uttar Pradesh", "Telangana", "Kerala",
    "Andhra Pradesh", "Madhya Pradesh", "Haryana", "Punjab", "Bihar"
]

# State risk levels (from Karnataka perspective)
HIGH_RISK_STATES = {"West Bengal", "Bihar", "Uttar Pradesh", "Rajasthan"}
MEDIUM_RISK_STATES = {"Delhi", "Maharashtra", "Gujarat", "Punjab"}
NEIGHBORING_STATES = {"Tamil Nadu", "Andhra Pradesh", "Telangana", "Kerala", "Goa"}

DEVICES = ["Android", "iOS", "Windows", "macOS"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"]

# Device preferences by region
DEVICE_WEIGHTS = {
    "Karnataka": [0.75, 0.20, 0.03, 0.02],  # High Android usage in Karnataka
    "India_other": [0.70, 0.25, 0.03, 0.02],  # Other Indian states
    "International": [0.45, 0.45, 0.06, 0.04]  # International
}

# Payment method preferences
PAYMENT_WEIGHTS = {
    "Karnataka": [0.25, 0.35, 0.25, 0.10, 0.05],  # High UPI and Debit card usage
    "India_other": [0.30, 0.30, 0.20, 0.15, 0.05],
    "International": [0.60, 0.25, 0.02, 0.08, 0.05]  # Credit cards dominant internationally
}

def generate_transaction(user_id: int):
    now = datetime.now()
    txn_time = now - timedelta(days=random.randint(0, 60), seconds=random.randint(0, 86400))
    
    # Determine if transaction is domestic (India) or international
    is_domestic = random.random() < 0.75  # 75% domestic transactions
    
    if is_domestic:
        country = "IN"
        # Karnataka-centric state distribution
        state = random.choices(
            INDIAN_STATES,
            weights=[0.45, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]
        )[0]
        location = f"{state}, India"
    else:
        country = random.choices(
            COUNTRIES[1:],  # Exclude India
            weights=[0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.13]
        )[0]
        state = None
        location = country
    
    base_state = "Karnataka"
    base_country = "IN"
    
    # Amount generation based on location
    if country == "IN":
        if state == "Karnataka":
            # Local Karnataka transactions (lower amounts)
            amount = round(random.lognormvariate(3.8, 1.1), 2)  # Mean ~₹2,500
        else:
            # Other Indian states (medium amounts)
            amount = round(random.lognormvariate(4.2, 1.3), 2)  # Mean ~₹4,500
    else:
        # International transactions (higher amounts)
        amount = round(random.lognormvariate(5.8, 1.4), 2)  # Mean ~₹35,000
    
    # Realistic average amounts based on user's typical behavior
    if country == "IN" and state == "Karnataka":
        avg_amt = max(0, random.gauss(2000, 800))
    elif country == "IN":
        avg_amt = max(0, random.gauss(4000, 1500))
    else:
        avg_amt = max(0, random.gauss(20000, 10000))
    
    # Transaction velocity based on location (using numpy's Poisson)
    if country == "IN" and state == "Karnataka":
        prev_24h = np.random.poisson(2.0)  # Higher frequency for local
    elif country == "IN":
        prev_24h = np.random.poisson(0.8)  # Medium for other states
    else:
        prev_24h = np.random.poisson(0.2)  # Low for international
    
    # Device and payment method selection
    if country == "IN" and state == "Karnataka":
        device_weights = DEVICE_WEIGHTS["Karnataka"]
        payment_weights = PAYMENT_WEIGHTS["Karnataka"]
    elif country == "IN":
        device_weights = DEVICE_WEIGHTS["India_other"]
        payment_weights = PAYMENT_WEIGHTS["India_other"]
    else:
        device_weights = DEVICE_WEIGHTS["International"]
        payment_weights = PAYMENT_WEIGHTS["International"]
    
    device = random.choices(DEVICES, weights=device_weights)[0]
    payment_method = random.choices(PAYMENT_METHODS, weights=payment_weights)[0]
    
    # Risk categorization
    is_international = int(country != base_country)
    is_out_of_state = int(country == "IN" and state != base_state)
    is_local = int(country == "IN" and state == base_state)
    
    # State-specific risk levels
    state_risk_level = 0
    if country == "IN" and state:
        if state in HIGH_RISK_STATES:
            state_risk_level = 3
        elif state in MEDIUM_RISK_STATES:
            state_risk_level = 2
        elif state in NEIGHBORING_STATES:
            state_risk_level = 1
        elif state == base_state:
            state_risk_level = 0
    
    # International risk levels
    international_risk_level = 0
    if is_international:
        if country in INTERNATIONAL_HIGH_RISK:
            international_risk_level = 4
        elif country in INTERNATIONAL_MEDIUM_RISK:
            international_risk_level = 2
        else:
            international_risk_level = 3
    
    # Enhanced fraud scoring system
    fraud_score = 0
    
    # Location-based risk
    if is_international:
        fraud_score += international_risk_level
        fraud_score += 3  # Base international penalty
    elif is_out_of_state:
        fraud_score += state_risk_level
        fraud_score += 1  # Base out-of-state penalty
    
    # Amount-based risk (adjusted for Indian context)
    if country == "IN":
        if amount > 100000:  # ₹1 Lakh+
            fraud_score += 4
        elif amount > 50000:  # ₹50k+
            fraud_score += 2
        elif amount > 25000:  # ₹25k+
            fraud_score += 1
    else:
        if amount > 200000:  # ₹2 Lakh+ for international
            fraud_score += 3
        elif amount > 100000:
            fraud_score += 2
        elif amount > 50000:
            fraud_score += 1
    
    # Velocity-based risk
    if prev_24h > 8:
        fraud_score += 3
    elif prev_24h > 5:
        fraud_score += 2
    elif prev_24h > 3:
        fraud_score += 1
    
    # Amount anomaly detection
    if avg_amt > 0:
        amount_ratio = amount / avg_amt
        if amount_ratio > 5:
            fraud_score += 3
        elif amount_ratio > 3:
            fraud_score += 2
        elif amount_ratio > 2:
            fraud_score += 1
    
    # Time-based risk
    hour = txn_time.hour
    if 1 <= hour <= 5:  # Very late night
        fraud_score += 2
    elif 22 <= hour <= 24 or 5 <= hour <= 7:  # Late night/early morning
        fraud_score += 1
    
    # Payment method risk
    if payment_method == "Credit Card" and is_international:
        fraud_score += 1
    elif payment_method in ["Net Banking", "UPI"] and is_international:
        fraud_score += 2  # Unusual for international
    
    # Device risk
    if device in ["Windows", "macOS"] and payment_method == "UPI":
        fraud_score += 1  # UPI typically mobile
    
    # Weekend international transactions
    is_weekend = txn_time.weekday() >= 5
    if is_weekend and is_international:
        fraud_score += 1
    
    # Determine fraud label with threshold
    is_fraud = int(fraud_score >= 7)  # Adjustable threshold
    
    # Add realistic noise
    if not is_fraud and random.random() < 0.015:  # 1.5% false positives
        is_fraud = 1
    elif is_fraud and random.random() < 0.12:  # 12% false negatives
        is_fraud = 0
    
    # Calculate normalized risk score
    risk_score = min(100, max(0, fraud_score * 10 + random.randint(-8, 8)))
    
    return {
        "transaction_id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "amount": amount,
        "timestamp": txn_time.isoformat(),
        "country": country,
        "state": state if country == "IN" else None,
        "location": location,
        "device": device,
        "payment_method": payment_method,
        "is_local": is_local,
        "is_out_of_state": is_out_of_state,
        "is_international": is_international,
        "state_risk_level": state_risk_level,
        "international_risk_level": international_risk_level,
        "prev_transactions_24h": prev_24h,
        "avg_amount_30d": round(avg_amt, 2),
        "transaction_hour": hour,
        "is_weekend": int(is_weekend),
        "amount_to_avg_ratio": round(amount / (avg_amt + 1), 2),
        "risk_score": risk_score,
        "fraud_score": fraud_score,
        "label": is_fraud
    }

def generate_dataset(n=1000):
    data = []
    # Generate realistic user base (Karnataka-centric)
    num_unique_users = n // 4  # Users have multiple transactions
    users = [random.randint(1000000, 9999999) for _ in range(num_unique_users)]
    
    for i in range(n):
        if i < num_unique_users:
            user_id = users[i]
        else:
            # Repeat users for multiple transactions
            user_id = random.choice(users)
        
        row = generate_transaction(user_id)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['is_high_amount'] = (df['amount'] > 50000).astype(int)
    df['is_business_hours'] = df['transaction_hour'].between(9, 17).astype(int)
    df['is_suspicious_time'] = df['transaction_hour'].between(1, 5).astype(int)
    
    return df

def print_dataset_stats(df):
    print(f"📊 Dataset Statistics (Karnataka Base):")
    print(f"   Total transactions: {len(df)}")
    print(f"   Unique users: {df['user_id'].nunique()}")
    print(f"   Fraud rate: {df['label'].mean():.2%}")
    print(f"   Local (Karnataka): {df['is_local'].mean():.2%}")
    print(f"   Out-of-state: {df['is_out_of_state'].mean():.2%}")
    print(f"   International: {df['is_international'].mean():.2%}")
    print(f"   Average amount: ₹{df['amount'].mean():.2f}")
    
    print(f"\n🏛️ State/Country Distribution:")
    location_counts = df['location'].value_counts().head(10)
    for location, count in location_counts.items():
        print(f"     {location}: {count} ({count/len(df):.1%})")
    
    print(f"\n💳 Payment Methods:")
    for method, count in df['payment_method'].value_counts().items():
        print(f"     {method}: {count} ({count/len(df):.1%})")
    
    print(f"\n⚠️ Risk Analysis:")
    print(f"   High risk (score > 70): {(df['risk_score'] > 70).mean():.2%}")
    print(f"   Medium risk (30-70): {((df['risk_score'] >= 30) & (df['risk_score'] <= 70)).mean():.2%}")
    print(f"   Low risk (< 30): {(df['risk_score'] < 30).mean():.2%}")

if __name__ == "__main__":
    df = generate_dataset(10000)
    
    # Create directory if it doesn't exist
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    df.to_csv("data/raw/karnataka_transactions_1k.csv", index=False)
    print("✅ Generated 1000 Karnataka-based transactions at data/raw/karnataka_transactions_1k.csv")
    
    print_dataset_stats(df)
    
    # Optional: Generate larger dataset
    generate_large = input("\n🤔 Generate 10k dataset? (y/n): ").lower()
    if generate_large == 'y':
        df_large = generate_dataset(10000)
        df_large.to_csv("data/raw/karnataka_transactions_10k.csv", index=False)
        print("\n✅ Generated 10,000 transactions at data/raw/karnataka_transactions_10k.csv")
        print_dataset_stats(df_large)