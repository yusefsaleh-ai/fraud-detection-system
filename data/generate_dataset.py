
import random
from datetime import datetime, timedelta

import pandas as pd


def generate_transactions(n: int = 10000) -> pd.DataFrame:
    merchant_categories = [
        "groceries",
        "electronics",
        "fashion",
        "travel",
        "gaming",
        "restaurants",
        "fuel",
        "subscriptions",
    ]

    payment_methods = [
        "debit_card",
        "credit_card",
        "mobile_wallet",
        "bank_transfer",
    ]

    rows = []
    start_time = datetime(2025, 1, 1, 0, 0, 0)

    for i in range(n):
        amount_gbp = round(random.uniform(5, 2000), 2)
        hour_of_day = random.randint(0, 23)
        merchant_category = random.choice(merchant_categories)
        payment_method = random.choice(payment_methods)
        transaction_velocity = random.randint(1, 15)
        location_risk = random.randint(0, 10)

        # Basic fraud rule to generate labels
        fraud_score = 0
        if amount_gbp > 1200:
            fraud_score += 1
        if transaction_velocity > 10:
            fraud_score += 1
        if location_risk > 7:
            fraud_score += 1
        if merchant_category in ["electronics", "travel", "gaming"]:
            fraud_score += 1

        fraud_label = 1 if fraud_score >= 3 else 0

        timestamp = start_time + timedelta(minutes=i * random.randint(1, 5))

        rows.append(
            {
                "transaction_id": i + 1,
                "amount_gbp": amount_gbp,
                "hour_of_day": hour_of_day,
                "merchant_category": merchant_category,
                "payment_method": payment_method,
                "transaction_velocity": transaction_velocity,
                "location_risk": location_risk,
                "fraud_label": fraud_label,
                "timestamp": timestamp,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_transactions(10000)
    output_path = "data/uk_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")
    print(df.head())
    print(df["fraud_label"].value_counts())