"""
Unsupervised anomaly detection (Isolation Forest) as an ML-based extension
to the original rule-based approach.

Rationale: the rule-based method requires BOTH a consumption drop AND a
recorded tamper alarm. But alarms aren't always triggered or logged
correctly in the field, so theft cases without a clean alarm signal slip
through. Isolation Forest looks at the full feature space (consumption
behavior + alarm counts together) and can flag meters that look
statistically unusual even without an explicit alarm — at the cost of
being noisier and needing a human to review the flagged list.

This script also builds a simple ensemble (rule OR model) to show how the
two approaches complement each other.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `from rule_based import ...` from any cwd
CONTAMINATION = 0.10  # assume ~10% of the population may be anomalous, matches known fraud rate for this demo
FEATURE_COLS = [
    "baseline_avg_kwh", "monitoring_avg_kwh", "monitoring_std_kwh",
    "drop_ratio", "min_drop_ratio",
    "alarm_count_tilt", "alarm_count_magnetic_tamper",
    "alarm_count_cover_open", "alarm_count_reverse_energy", "total_alarms",
]


def load_data():
    features = pd.read_csv(ROOT / "data/processed/features.csv")
    ground_truth = pd.read_csv(ROOT / "data/processed/ground_truth.csv")
    return features.merge(ground_truth[["meter_id", "is_fraud"]], on="meter_id")


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    )
    df = df.copy()
    raw_pred = model.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    df["anomaly_score"] = model.decision_function(X_scaled) * -1  # higher = more anomalous
    df["ml_flagged"] = raw_pred == -1
    return df


def evaluate(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "flagged_count": int(y_pred.sum()),
    }


def main():
    df = load_data()
    df = run_isolation_forest(df)

    # Bring in the rule-based flags for comparison / ensemble
    from rule_based import flag_suspects  # local import to avoid circulars at module load
    df = flag_suspects(df)
    df["ensemble_flagged"] = df["rule_flagged"] | df["ml_flagged"]

    print("Isolation Forest results:")
    ml_metrics = evaluate(df["is_fraud"], df["ml_flagged"])
    for k, v in ml_metrics.items():
        print(f"  {k}: {v}")

    print("\nEnsemble (rule OR model) results:")
    ens_metrics = evaluate(df["is_fraud"], df["ensemble_flagged"])
    for k, v in ens_metrics.items():
        print(f"  {k}: {v}")

    df[[
        "meter_id", "zone_id", "drop_ratio", "min_drop_ratio", "total_alarms",
        "anomaly_score", "rule_flagged", "ml_flagged", "ensemble_flagged", "is_fraud",
    ]].sort_values("anomaly_score", ascending=False).to_csv(
        ROOT / "outputs/anomaly_detection_results.csv", index=False
    )

    return ml_metrics, ens_metrics


if __name__ == "__main__":
    main()
