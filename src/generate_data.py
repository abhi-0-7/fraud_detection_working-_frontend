import random
import uuid
from datetime import datetime, timedelta
import pandas as pd

# Constants
COUNTRIES = ["US", "IN", "CN", "GB", "DE", "FR"]
HIGH_RISK = {"CN", "IN"}
DEVICES = ["Android", "iOS", "Windows", "macOS"]

def generate_transaction(user_id: int):
    now = datetime.now()
    txn_time = now - timedelta(days=random.randint(0, 60), seconds=random.randint(0, 86400))
    country = random.choices(COUNTRIES, weights=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])[0]
    base_country = "US"  # Assume all users are based in US for now

    amount = round(random.expovariate(1/100), 2)
    avg_amt = max(0, random.gauss(50, 20))
    prev_24h = random.poisson(2) if hasattr(random, "poisson") else random.randint(0, 6)  # fallback

    is_foreign = int(country != base_country)
    is_high_risk = int(country in HIGH_RISK)

    # Simple fraud rule: foreign + high risk + large amount
    is_fraud = int(is_foreign and is_high_risk and amount > 300)

    return {
        "transaction_id": uuid.uuid4().hex[:12],
        "user_id": user_id,
        "amount": amount,
        "timestamp": txn_time.isoformat(),
        "country": country,
        "device": random.choice(DEVICES),
        "is_foreign": is_foreign,
        "is_high_risk_country": is_high_risk,
        "prev_transactions_24h": prev_24h,
        "avg_amount_30d": round(avg_amt, 2),
        "label": is_fraud
    }

def generate_dataset(n=1000):
    data = []
    for _ in range(n):
        user_id = random.randint(10000, 99999)
        row = generate_transaction(user_id)
        data.append(row)
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset(1000)
    df.to_csv("data/raw/sample_1k.csv", index=False)
    print("✅ Generated 1000 fake transactions at data/raw/sample_1k.csv")
