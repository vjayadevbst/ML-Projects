import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# Load data
df = pd.read_csv(r"C:\Users\vinna\OneDrive\AI ML Projects\V Projects\Revenue_Recognition_Project\data\revenue_data.csv")

# Encode categorical columns
categorical_cols = ["ProductCategory", "ShippingMode", "Region", "Weather", "Carrier"]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df.drop(columns=["Delayed", "OrderID", "CustomerID", "OrderDate"])
y = df["Delayed"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model
os.makedirs("../src", exist_ok=True)
with open("../src/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved to ../src/model.pkl")
