"""
Power Pulse - Synthetic Sensor Data Generator (Multi-City)
Generates simulated voltage, current, frequency and load readings for
several Indian cities, with power-cut events. Each city has its own
outage-risk profile so the model learns city-specific patterns.
Output: power_pulse_raw.csv (with a 'city' column)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

START_DATE = datetime(2025, 1, 1)
NUM_DAYS = 90
INTERVAL_MINUTES = 10
PEAK_HOURS = range(18, 21)
MONSOON_MONTHS = [6, 7, 8, 9]

# Illustrative per-city parameters for a student project (not official statistics)
CITIES = {
    "Bangalore": {"outage_base": 0.0030, "voltage_mean": 230, "monsoon_multiplier": 2.5},
    "Mumbai":    {"outage_base": 0.0018, "voltage_mean": 230, "monsoon_multiplier": 3.0},
    "Delhi":     {"outage_base": 0.0035, "voltage_mean": 225, "monsoon_multiplier": 2.0},
    "Chennai":   {"outage_base": 0.0025, "voltage_mean": 230, "monsoon_multiplier": 2.8},
    "Hyderabad": {"outage_base": 0.0022, "voltage_mean": 232, "monsoon_multiplier": 2.2},
}

np.random.seed(42)


def generate_city_data(city_name, params):
    total_points = int((NUM_DAYS * 24 * 60) / INTERVAL_MINUTES)
    timestamps = [START_DATE + timedelta(minutes=INTERVAL_MINUTES * i) for i in range(total_points)]

    df = pd.DataFrame({"timestamp": timestamps})
    df["city"] = city_name
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monsoon"] = df["month"].isin(MONSOON_MONTHS).astype(int)

    n = len(df)
    df["voltage"] = np.random.normal(params["voltage_mean"], 4, n)
    df["current"] = np.random.normal(8, 2, n).clip(0.5)
    df["frequency"] = np.random.normal(50, 0.15, n)
    df["load_kw"] = np.random.normal(3.5, 1.2, n).clip(0.2)

    peak_mask = df["is_peak_hour"] == 1
    df.loc[peak_mask, "load_kw"] += np.random.uniform(1.5, 3.0, peak_mask.sum())
    df.loc[peak_mask, "current"] += np.random.uniform(1.0, 2.5, peak_mask.sum())

    df["power_cut"] = 0
    i = 0
    while i < n:
        prob = params["outage_base"]
        if df.loc[i, "is_peak_hour"]:
            prob *= 4
        if df.loc[i, "is_monsoon"]:
            prob *= params["monsoon_multiplier"]

        if np.random.random() < prob:
            decline_steps = np.random.randint(3, 7)
            start_decline = max(0, i - decline_steps)
            for j, idx in enumerate(range(start_decline, i)):
                decay = (j + 1) / decline_steps
                df.loc[idx, "voltage"] -= decay * np.random.uniform(8, 15)
                df.loc[idx, "frequency"] -= decay * np.random.uniform(0.1, 0.3)

            outage_len = np.random.randint(1, 4)
            for k in range(outage_len):
                if i + k < n:
                    df.loc[i + k, "voltage"] = np.random.uniform(0, 15)
                    df.loc[i + k, "current"] = np.random.uniform(0, 0.5)
                    df.loc[i + k, "load_kw"] = 0
                    df.loc[i + k, "power_cut"] = 1
            i += outage_len + 1
        else:
            i += 1

    LOOKAHEAD_STEPS = 6  # 6 x 10 min = 1 hour
    df["power_cut_next_1hr"] = (
        df["power_cut"][::-1].rolling(LOOKAHEAD_STEPS, min_periods=1).max()[::-1].astype(int)
    )

    for col in ["voltage", "current", "frequency", "load_kw"]:
        df[col] = df[col].round(2)
    return df


all_dfs = []
for city_name, params in CITIES.items():
    city_df = generate_city_data(city_name, params)
    all_dfs.append(city_df)
    print(f"{city_name}: {len(city_df)} rows, {city_df['power_cut'].sum()} outage readings, "
          f"positive ratio {city_df['power_cut_next_1hr'].mean():.4f}")

df = pd.concat(all_dfs, ignore_index=True)
df.to_csv("power_pulse_raw.csv", index=False)
print(f"\nTotal rows across {len(CITIES)} cities: {len(df)}")
print("Saved to power_pulse_raw.csv")
