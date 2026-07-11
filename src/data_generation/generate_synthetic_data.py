"""
Synthetic data generator for the Energy Theft Detection portfolio project.

Generates a realistic-but-fake dataset simulating:
  - A population of smart meters across several distribution zones
  - Daily energy consumption per meter, with weekly seasonality and noise
  - Tamper alarm events reported by the meters (tilt, magnetic tamper,
    cover-open, reverse energy) — modeled after real Elster/Itron style
    anti-tamper features documented in industry literature
  - A "ground truth" fraud label, analogous to a utility's post-inspection
    confirmation records. This label is used ONLY to evaluate detection
    performance in this demo; in a real deployment it would not exist
    until a field crew physically confirmed a case.

None of this data represents real customers, real meters, or real company
data. It is generated purely to reproduce the *statistical shape* of the
problem described in the project README.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_METERS = 500
N_DAYS = 180
FRAUD_RATE = 0.10  # ~10% of meters will be simulated fraud cases
ZONES = [f"ZONE-{i:02d}" for i in range(1, 11)]
CUSTOMER_TYPES = ["residential", "residential", "residential", "commercial", "industrial"]
ALARM_TYPES = ["tilt", "magnetic_tamper", "cover_open", "reverse_energy"]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_meters(rng: np.random.Generator) -> pd.DataFrame:
    meter_ids = [f"MTR-{i:05d}" for i in range(1, N_METERS + 1)]
    zones = rng.choice(ZONES, size=N_METERS)
    customer_types = rng.choice(CUSTOMER_TYPES, size=N_METERS)
    contracted_kva = np.where(
        customer_types == "residential",
        rng.choice([5, 10, 15], size=N_METERS),
        np.where(customer_types == "commercial",
                 rng.choice([25, 50, 75], size=N_METERS),
                 rng.choice([100, 150, 200], size=N_METERS)),
    )
    return pd.DataFrame({
        "meter_id": meter_ids,
        "zone_id": zones,
        "customer_type": customer_types,
        "contracted_capacity_kva": contracted_kva,
    })


def base_consumption_profile(customer_type: str, rng: np.random.Generator) -> float:
    """Average daily kWh baseline by customer type."""
    if customer_type == "residential":
        return rng.uniform(8, 25)
    if customer_type == "commercial":
        return rng.uniform(40, 120)
    return rng.uniform(150, 400)  # industrial


def generate_consumption_and_labels(meters: pd.DataFrame, rng: np.random.Generator):
    dates = pd.date_range("2025-01-01", periods=N_DAYS, freq="D")
    n_fraud = int(N_METERS * FRAUD_RATE)
    fraud_meter_ids = set(rng.choice(meters["meter_id"], size=n_fraud, replace=False))

    consumption_rows = []
    alarm_rows = []
    ground_truth_rows = []

    for _, meter in meters.iterrows():
        meter_id = meter["meter_id"]
        baseline = base_consumption_profile(meter["customer_type"], rng)
        is_fraud = meter_id in fraud_meter_ids
        # Fraud starts at a random point after day 60, so there's always
        # a clean baseline period to compare against.
        fraud_start_idx = int(rng.integers(60, N_DAYS - 30)) if is_fraud else None
        drop_ratio = rng.uniform(0.35, 0.65) if is_fraud else 0.0  # 35-65% consumption drop

        for day_idx, date in enumerate(dates):
            weekday_factor = 0.85 if date.dayofweek >= 5 else 1.0  # slightly lower on weekends
            noise = rng.normal(1.0, 0.08)
            seasonal_drift = 1.0 + 0.05 * np.sin(2 * np.pi * day_idx / 365)
            kwh = baseline * weekday_factor * seasonal_drift * max(noise, 0.2)

            if is_fraud and day_idx >= fraud_start_idx:
                kwh *= (1 - drop_ratio)

            consumption_rows.append((meter_id, date, round(max(kwh, 0), 2)))

            # Alarm generation
            if is_fraud and fraud_start_idx <= day_idx <= fraud_start_idx + 20:
                # Elevated tamper-alarm probability in the window right
                # around when the physical tampering likely occurred
                if rng.random() < 0.12:
                    alarm_type = rng.choice(ALARM_TYPES, p=[0.35, 0.30, 0.25, 0.10])
                    alarm_rows.append((meter_id, date, alarm_type))
            else:
                # Background false-positive rate: alarms happen for
                # legitimate reasons too (installation work, vibration, etc.)
                if rng.random() < 0.002:
                    alarm_type = rng.choice(ALARM_TYPES)
                    alarm_rows.append((meter_id, date, alarm_type))

        ground_truth_rows.append((
            meter_id,
            is_fraud,
            dates[fraud_start_idx] if is_fraud else pd.NaT,
        ))

    consumption_df = pd.DataFrame(consumption_rows, columns=["meter_id", "date", "kwh_consumed"])
    alarms_df = pd.DataFrame(alarm_rows, columns=["meter_id", "date", "alarm_type"])
    ground_truth_df = pd.DataFrame(ground_truth_rows, columns=["meter_id", "is_fraud", "fraud_start_date"])
    return consumption_df, alarms_df, ground_truth_df


def main():
    rng = np.random.default_rng(RNG_SEED)
    meters = generate_meters(rng)
    consumption, alarms, ground_truth = generate_consumption_and_labels(meters, rng)

    meters.to_csv(OUT_DIR / "meters.csv", index=False)
    consumption.to_csv(OUT_DIR / "consumption.csv", index=False)
    alarms.to_csv(OUT_DIR / "alarms.csv", index=False)
    ground_truth.to_csv(OUT_DIR.parent / "processed" / "ground_truth.csv", index=False)

    print(f"Meters: {len(meters)}")
    print(f"Consumption rows: {len(consumption)}")
    print(f"Alarm events: {len(alarms)}")
    print(f"Fraud cases (ground truth, evaluation-only): {ground_truth['is_fraud'].sum()}")


if __name__ == "__main__":
    main()
