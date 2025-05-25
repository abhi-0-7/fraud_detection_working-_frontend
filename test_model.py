import pandas as pd
import joblib

# 1) Load the pipeline
model = joblib.load("models/fraud_detector_tuned_pipeline.pkl")

# 2) Load NEW data (could be held‑out, could be fresh logs)
df = pd.read_csv("data/raw/sample_1k.csv")  # replace with any path

# 3) Apply feature engineering (exactly as in training)
df['timestamp']              = pd.to_datetime(df['timestamp'])
df['hour_of_day']            = df['timestamp'].dt.hour
df['day_of_week']            = df['timestamp'].dt.dayofweek
df['is_night']               = df['hour_of_day'].between(0, 6).astype(int)
df['amount_diff_from_avg']   = df['amount'] - df['avg_amount_30d']
df['risk_score']             = (
    df['is_foreign'] * 1.5 +
    df['is_high_risk_country'] * 2.0 +
    df['is_night'] * 1.2 +
    (df['amount_diff_from_avg'].abs() / 50)
)

# 4) Drop unused cols and predict
X = df.drop(columns=['transaction_id','timestamp','label'])
df['pred_label']  = model.predict(X)
df['fraud_proba'] = model.predict_proba(X)[:,1]

# 5) Inspect or save results
print(df[['pred_label','fraud_proba']].head())
df.to_csv("data/raw/with_predictions.csv", index=False)
