import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("data/uk_transactions.csv")

# Features and target
X = df[[
    "amount_gbp",
    "hour_of_day",
    "merchant_category",
    "payment_method",
    "transaction_velocity",
    "location_risk",
]]
y = df["fraud_label"]

# Load trained model
model = joblib.load("model/random_forest_model.pkl")

# Predict
y_pred = model.predict(X)

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("\nMODEL PERFORMANCE")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred))