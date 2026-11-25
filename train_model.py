"""
Power Pulse - Feature Engineering + Model Training (Multi-City)
Loads power_pulse_raw.csv, engineers rolling/lag features PER CITY,
one-hot encodes the city, trains a Random Forest, and saves the model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

df = pd.read_csv("power_pulse_raw.csv", parse_dates=["timestamp"])
df = df.sort_values(["city", "timestamp"]).reset_index(drop=True)

# Feature engineering, computed separately for each city
g = df.groupby("city")
for window, label in [(3, "30min"), (6, "1hr")]:
    df[f"voltage_mean_{label}"] = g["voltage"].transform(lambda s: s.rolling(window).mean())
    df[f"voltage_std_{label}"] = g["voltage"].transform(lambda s: s.rolling(window).std())
    df[f"frequency_mean_{label}"] = g["frequency"].transform(lambda s: s.rolling(window).mean())

df["voltage_change_rate"] = g["voltage"].diff()
df["frequency_change_rate"] = g["frequency"].diff()
df["voltage_lag_1"] = g["voltage"].shift(1)
df["voltage_lag_3"] = g["voltage"].shift(3)

df = df.dropna().reset_index(drop=True)

# One-hot encode city so the model can learn city-specific patterns
city_dummies = pd.get_dummies(df["city"], prefix="city").astype(int)
df = pd.concat([df, city_dummies], axis=1)

BASE_FEATURES = [
    "voltage", "current", "frequency", "load_kw",
    "hour", "is_peak_hour", "is_weekend", "is_monsoon",
    "voltage_mean_30min", "voltage_std_30min", "frequency_mean_30min",
    "voltage_mean_1hr", "voltage_std_1hr", "frequency_mean_1hr",
    "voltage_change_rate", "frequency_change_rate",
    "voltage_lag_1", "voltage_lag_3",
]
FEATURE_COLS = BASE_FEATURES + list(city_dummies.columns)
TARGET_COL = "power_cut_next_1hr"

# Time-based split inside each city: first 80% train, last 20% test
train_parts, test_parts = [], []
for city, part in df.groupby("city"):
    split = int(len(part) * 0.8)
    train_parts.append(part.iloc[:split])
    test_parts.append(part.iloc[split:])
train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

model = RandomForestClassifier(
    n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Overall classification report:")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPer-city recall on the power-cut class:")
for city, part in test_df.groupby("city"):
    preds = model.predict(part[FEATURE_COLS])
    positives = (part[TARGET_COL] == 1).sum()
    caught = ((preds == 1) & (part[TARGET_COL] == 1)).sum()
    print(f"  {city}: {caught}/{positives} caught")

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nTop feature importances:")
print(importance.head(10))

joblib.dump(model, "power_pulse_model.pkl")
joblib.dump(FEATURE_COLS, "feature_cols.pkl")
df.to_csv("power_pulse_features.csv", index=False)
print("\nSaved: power_pulse_model.pkl, feature_cols.pkl, power_pulse_features.csv")
