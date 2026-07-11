"""
Rule-based energy theft detection.

Reproduces the original methodology: flag a meter as suspicious if BOTH
conditions hold within the monitoring period —
  1. A sustained consumption drop beyond a threshold vs. its own baseline
  2. At least one tamper-type alarm (tilt / magnetic tamper / cover open)
     consistent with physical meter interference

Reverse-energy alarms are excluded from the tamper trigger on their own
since they can also fire for legitimate reasons (e.g. distributed solar
generation) — this mirrors the real-world nuance that not every alarm
means theft; it's the pattern of drop + tamper evidence together that
matters.
"""

from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DROP_THRESHOLD = 0.30          # flag if sustained drop >= 30%
TAMPER_ALARM_COLS = ["alarm_count_tilt", "alarm_count_magnetic_tamper", "alarm_count_cover_open"]


def load_features_and_labels():
    features = pd.read_csv(ROOT / "data/processed/features.csv")
    ground_truth = pd.read_csv(ROOT / "data/processed/ground_truth.csv")
    return features.merge(ground_truth[["meter_id", "is_fraud"]], on="meter_id")


def flag_suspects(df: pd.DataFrame) -> pd.DataFrame:
    has_tamper_alarm = df[TAMPER_ALARM_COLS].sum(axis=1) > 0
    has_sustained_drop = df["min_drop_ratio"] >= DROP_THRESHOLD
    df = df.copy()
    df["rule_flagged"] = has_tamper_alarm & has_sustained_drop
    return df


def evaluate(df: pd.DataFrame) -> dict:
    y_true = df["is_fraud"]
    y_pred = df["rule_flagged"]
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
    df = load_features_and_labels()
    df = flag_suspects(df)
    metrics = evaluate(df)

    print("Rule-based detection results (drop >= 30% AND tamper alarm present):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    df[["meter_id", "zone_id", "drop_ratio", "min_drop_ratio", "total_alarms", "rule_flagged", "is_fraud"]].to_csv(
        ROOT / "outputs/rule_based_results.csv", index=False
    )
    return metrics


if __name__ == "__main__":
    main()
