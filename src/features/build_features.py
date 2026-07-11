"""
Feature engineering for energy theft detection.

Turns raw daily consumption + alarm event logs into one feature row per
meter, comparing a "baseline period" (first 60 days, assumed clean) against
a "monitoring period" (the rest of the series) — mirroring how an analyst
would compare a customer's historical average against recent behavior.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASELINE_DAYS = 60
ROOT = Path(__file__).resolve().parents[2]


def load_raw_data():
    consumption = pd.read_csv(ROOT / "data/raw/consumption.csv", parse_dates=["date"])
    alarms = pd.read_csv(ROOT / "data/raw/alarms.csv", parse_dates=["date"])
    meters = pd.read_csv(ROOT / "data/raw/meters.csv")
    return consumption, alarms, meters


def build_feature_table(consumption: pd.DataFrame, alarms: pd.DataFrame, meters: pd.DataFrame) -> pd.DataFrame:
    consumption = consumption.sort_values(["meter_id", "date"])
    cutoff_date = consumption["date"].min() + pd.Timedelta(days=BASELINE_DAYS)

    baseline = consumption[consumption["date"] < cutoff_date]
    monitoring = consumption[consumption["date"] >= cutoff_date]

    baseline_avg = baseline.groupby("meter_id")["kwh_consumed"].mean().rename("baseline_avg_kwh")
    monitoring_avg = monitoring.groupby("meter_id")["kwh_consumed"].mean().rename("monitoring_avg_kwh")
    monitoring_std = monitoring.groupby("meter_id")["kwh_consumed"].std().rename("monitoring_std_kwh")

    # Steepest sustained drop: minimum 14-day rolling average in the monitoring
    # period, expressed as a ratio of the baseline average.
    def min_rolling_ratio(group):
        s = group.set_index("date")["kwh_consumed"].rolling("14D").mean()
        return s.min()

    min_rolling = monitoring.groupby("meter_id").apply(min_rolling_ratio).rename("min_14d_rolling_kwh")

    alarm_counts = alarms.groupby(["meter_id", "alarm_type"]).size().unstack(fill_value=0)
    alarm_counts.columns = [f"alarm_count_{c}" for c in alarm_counts.columns]
    alarm_counts["total_alarms"] = alarm_counts.sum(axis=1)

    features = meters.set_index("meter_id").join(
        [baseline_avg, monitoring_avg, monitoring_std, min_rolling, alarm_counts], how="left"
    ).fillna(0)

    features["drop_ratio"] = 1 - (features["monitoring_avg_kwh"] / features["baseline_avg_kwh"].replace(0, np.nan))
    features["min_drop_ratio"] = 1 - (features["min_14d_rolling_kwh"] / features["baseline_avg_kwh"].replace(0, np.nan))
    features = features.fillna(0)

    return features.reset_index()


def main():
    consumption, alarms, meters = load_raw_data()
    features = build_feature_table(consumption, alarms, meters)
    out_path = ROOT / "data/processed/features.csv"
    features.to_csv(out_path, index=False)
    print(f"Feature table written to {out_path} ({len(features)} meters, {features.shape[1]} columns)")


if __name__ == "__main__":
    main()
