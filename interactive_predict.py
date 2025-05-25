import pandas as pd
import joblib
from datetime import datetime

def prompt_choice(prompt, choices):
    """Prompt until user enters one of the allowed choices."""
    while True:
        val = input(f"{prompt} {choices}: ").strip()
        if val in choices:
            return val
        print("❌ Invalid choice. Try again.")

def prompt_float(prompt):
    """Prompt until user enters a float."""
    while True:
        try:
            return float(input(f"{prompt}: ").strip())
        except ValueError:
            print("❌ Please enter a number.")

def prompt_int(prompt):
    """Prompt until user enters an int."""
    while True:
        try:
            return int(input(f"{prompt}: ").strip())
        except ValueError:
            print("❌ Please enter an integer.")

def get_transaction():
    txn = {}
    txn['transaction_id'] = input("Transaction ID (string): ").strip()
    txn['user_id'] = prompt_int("User ID (integer)")
    txn['amount'] = prompt_float("Amount (e.g. 123.45)")
    ts = input("Timestamp (YYYY-MM-DD HH:MM:SS, leave blank for now): ").strip()
    if not ts:
        txn['timestamp'] = datetime.now().isoformat()
    else:
        txn['timestamp'] = datetime.fromisoformat(ts).isoformat()
    txn['country'] = prompt_choice("Country", ["US","IN","CN","GB","DE","FR"])
    txn['device']  = prompt_choice("Device", ["Android","iOS","Windows","macOS"])
    txn['is_foreign']           = 1 if txn['country'] != "US" else 0
    txn['is_high_risk_country'] = 1 if txn['country'] in ("CN","IN") else 0
    txn['prev_transactions_24h'] = prompt_int("Previous transactions in last 24h")
    txn['avg_amount_30d']       = prompt_float("Avg amount last 30d")
    return txn

def engineer_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night']   = df['hour_of_day'].between(0,6).astype(int)
    df['amount_diff_from_avg'] = df['amount'] - df['avg_amount_30d']
    df['risk_score'] = (
        df['is_foreign'] * 1.5 +
        df['is_high_risk_country'] * 2.0 +
        df['is_night'] * 1.2 +
        df['amount_diff_from_avg'].abs() / 50
    )
    return df

def main():
    model = joblib.load("models/fraud_detector_tuned_pipeline.pkl")
    print("✅ Model loaded.")

    while True:
        print("\nEnter a new transaction:")
        txn = get_transaction()
        df  = pd.DataFrame([txn])
        df  = engineer_features(df)

        X    = df.drop(columns=['transaction_id','timestamp'], errors='ignore')
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0,1]

        print("\n🔍 Prediction Results")
        print(f"→ Predicted label : {'FRAUD' if pred==1 else 'LEGIT'}")
        print(f"→ Fraud probability: {proba:.2%}")

        cont = input("\nScore another? (y/n): ").strip().lower()
        if cont != 'y':
            print("👋 Exiting. Goodbye!")
            break

if __name__ == "__main__":
    main()
