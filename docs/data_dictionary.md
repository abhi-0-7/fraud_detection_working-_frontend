| Field                  | Type        | Description                                                  | Example    | Notes                                       |
|------------------------|-------------|--------------------------------------------------------------|------------|---------------------------------------------|
| transaction_id         | integer     | Unique ID for each transaction                              | 1000001    | Auto‑incrementing                           |
| user_id                | integer     | Customer’s unique ID                                         | 54321      | Random from 1–100 000                       |
| amount                 | float       | Transaction amount in USD                                    | 123.45     | Exponential distribution, scale=100        |
| timestamp              | datetime    | Date & time of transaction                                   | 2025-05-19T14:32:10 | Uniform over past 60 days          |
| country                | categorical | Transaction country                                          | "IN"       | Probabilities: US .30, IN .20, …            |
| device                 | categorical | Device type used                                             | "iOS"      | Uniform among listed categories             |
| is_foreign             | boolean     | 1 if country ≠ user's base country                           | 1          | Derived: country != "US"                    |
| is_high_risk_country   | boolean     | 1 if country in high‑risk list (CN, IN)                      | 1          | Derived from country                        |
| prev_transactions_24h  | integer     | Number of txns in last 24h by user                           | 5          | Poisson(λ=2)                                |
| avg_amount_30d         | float       | User’s avg txn amount over last 30 days                      | 50.75      | Normal(μ=50, σ=20), abs value               |
| label                  | boolean     | Fraud label: 1 = fraudulent, 0 = legitimate                  | 0          | Injected via rule‑based patterns           |
