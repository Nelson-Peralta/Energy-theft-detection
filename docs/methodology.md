# Methodology

## Background: the real-world problem

Electric utilities lose revenue to **non-technical losses (NTL)** — energy
that's consumed but never billed, most commonly due to meter tampering or
illegal connections. This is a well-documented, worldwide problem: countries
report NTL rates ranging from single digits up to 20%+ of distributed
energy in the most affected regions.

Modern smart meters (this project focuses on Elster-style AMI meters)
include built-in anti-tamper hardware:

- **Tilt sensors** — detect if the meter is physically removed, rotated, or
  opened.
- **Cover-open switches** — count how many times the meter's terminal or
  main cover has been removed.
- **Magnetic tamper detection** — flags a strong external magnet placed
  near the meter to saturate its current transformer.
- **Reverse energy detection** — flags sustained reverse current flow, which
  can indicate a wiring bypass (though it can also fire for legitimate
  reasons, like distributed solar generation, so it's a weaker signal on
  its own).

None of these alarms is proof of theft by itself — a meter can trip an
alarm for legitimate reasons (installation work, vibration, a nearby
solar installation). The real signal comes from **correlating an alarm
with an unexplained, sustained drop in consumption**.

## Real-world origin of this project

While working on the loss-reduction program of a national electric
utility, I co-designed this detection approach with the team: combining
meter tamper alarm data with consumption threshold analysis to flag likely
theft cases for field inspection. Confirmed fraud cases were fed into the
utility's loss-reduction efforts. We later iterated the production alarm
logic around a season-aware, per-customer consumption baseline (3 years of
history, so seasonal drops such as lower winter A/C usage don't trigger
false flags), combined with multi-parameter triggers such as meter tilt
events — which cut theft-detection false positives by ~35%.

**This repository is a from-scratch reproduction of the core methodology,
built entirely on synthetic data**, so the approach and reasoning can be
shown publicly without exposing any real customer or company data. The
production metric above is not reproduced here. See the main `README.md`
for the full data disclosure.

## Approach implemented here

### 1. Synthetic data generation
`src/data_generation/generate_synthetic_data.py` creates 500 meters across
10 zones with 180 days of daily consumption, plus a stream of tamper alarm
events. ~10% of meters are seeded as "fraud" cases: at a random point after
day 60, their consumption drops 35–65% and they have an elevated (but not
guaranteed) chance of tripping a tamper alarm in the following weeks —
mirroring how real theft doesn't always get caught by the hardware, and
how normal meters occasionally trip alarms for unrelated reasons.

A "ground truth" label is stored separately, standing in for what a real
utility would only know after a field crew physically confirms a case.
It's used here purely to *evaluate* detection performance — it would not
exist ahead of time in a live deployment.

### 2. Feature engineering
`src/features/build_features.py` compares each meter's first 60 days
(baseline) against the rest of the series (monitoring period): average
consumption, the steepest sustained 14-day rolling drop, and counts of
each alarm type.

### 3. Rule-based detection (the original approach)
`src/models/rule_based.py` flags a meter if it shows **both**:
- A sustained consumption drop ≥ 30% vs. baseline, **and**
- At least one tamper alarm (tilt, magnetic tamper, or cover-open) in the
  monitoring period.

This mirrors the original logic: don't act on a drop alone (could be a
vacant property, a closed business, seasonal use) and don't act on an
alarm alone (could be unrelated) — only the combination is a strong signal.

### 4. ML extension: Isolation Forest
`src/models/anomaly_detection.py` trains an unsupervised Isolation Forest
on the full feature set (consumption behavior + alarm counts together).
This is meant to complement, not replace, the rule-based method: it can
catch statistically unusual meters even when the alarm didn't fire or
wasn't logged correctly — a real limitation of hardware-dependent
detection — at the cost of a higher false-positive rate that needs human
review.

### 5. Evaluation
Both approaches, plus a simple ensemble (flag if either fires), are scored
against the synthetic ground truth using precision, recall, and F1. See
`README.md` for the results table and the business interpretation of the
precision/recall trade-off.

## Limitations (of this synthetic reproduction)

- The synthetic data is deliberately generated to contain the pattern
  being tested for — real-world data is messier, with more confounding
  factors (weather, tariff changes, meter malfunctions unrelated to fraud,
  partial-month billing, etc.).
- The 30% drop threshold and alarm-window parameters are illustrative,
  tuned to this synthetic dataset. In a production setting, thresholds
  would be calibrated against a labeled historical dataset and validated
  zone by zone, since consumption patterns vary a lot by customer type and
  region.
- The Isolation Forest `contamination` parameter was set using knowledge of
  the synthetic fraud rate (10%). In a real deployment, this would need to
  be estimated from historical inspection data or tuned via cross-validation.

These limitations are the starting point for v2 (in progress): realistic
confounders, leakage-free evaluation by zone and time, and dbt-duckdb
models with tests.
