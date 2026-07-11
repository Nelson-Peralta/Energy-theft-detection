"""Generates the summary charts used in the README and notebook."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams["figure.dpi"] = 120


def chart_consumption_example():
    consumption = pd.read_csv(ROOT / "data/raw/consumption.csv", parse_dates=["date"])
    ground_truth = pd.read_csv(ROOT / "data/processed/ground_truth.csv")
    alarms = pd.read_csv(ROOT / "data/raw/alarms.csv", parse_dates=["date"])

    fraud_meter = ground_truth[ground_truth["is_fraud"]].iloc[0]["meter_id"]
    normal_meter = ground_truth[~ground_truth["is_fraud"]].iloc[0]["meter_id"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for meter_id, label, color in [(fraud_meter, "Fraud case (flagged)", "#d62728"),
                                     (normal_meter, "Normal customer", "#1f77b4")]:
        series = consumption[consumption["meter_id"] == meter_id].sort_values("date")
        rolling = series.set_index("date")["kwh_consumed"].rolling(7).mean()
        ax.plot(rolling.index, rolling.values, label=f"{label} ({meter_id})", color=color, linewidth=1.6)

    meter_alarms = alarms[alarms["meter_id"] == fraud_meter]
    for _, row in meter_alarms.iterrows():
        ax.axvline(row["date"], color="#d62728", alpha=0.25, linestyle="--", linewidth=1)

    ax.set_title("7-day rolling consumption: fraud case vs. normal customer\n(dashed lines = tamper alarm events)")
    ax.set_ylabel("kWh/day")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "consumption_example.png")
    plt.close(fig)


def chart_model_comparison():
    labels = ["Rule-based", "Isolation Forest", "Ensemble (OR)"]
    precision = [1.0, 0.8, 0.828]
    recall = [0.88, 0.8, 0.96]
    f1 = [0.936, 0.8, 0.889]

    x = range(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width for i in x], precision, width, label="Precision", color="#4c72b0")
    ax.bar(x, recall, width, label="Recall", color="#dd8452")
    ax.bar([i + width for i in x], f1, width, label="F1 score", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection performance by approach (synthetic evaluation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)


def chart_drop_vs_alarms():
    features = pd.read_csv(ROOT / "data/processed/features.csv")
    ground_truth = pd.read_csv(ROOT / "data/processed/ground_truth.csv")
    df = features.merge(ground_truth[["meter_id", "is_fraud"]], on="meter_id")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for is_fraud, label, color, marker in [(True, "Confirmed fraud", "#d62728", "x"),
                                              (False, "Normal", "#1f77b4", "o")]:
        subset = df[df["is_fraud"] == is_fraud]
        ax.scatter(subset["min_drop_ratio"], subset["total_alarms"],
                   label=label, color=color, marker=marker, alpha=0.7)

    ax.axvline(0.30, color="gray", linestyle="--", linewidth=1, label="Rule threshold (30% drop)")
    ax.set_xlabel("Max sustained consumption drop (ratio vs. baseline)")
    ax.set_ylabel("Total tamper alarms in monitoring period")
    ax.set_title("Consumption drop vs. tamper alarms by meter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "drop_vs_alarms.png")
    plt.close(fig)


if __name__ == "__main__":
    chart_consumption_example()
    chart_model_comparison()
    chart_drop_vs_alarms()
    print("Charts saved to outputs/")
