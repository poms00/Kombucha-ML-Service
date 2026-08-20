"""Generate a labelled synthetic dataset for the kombucha fermentation-stage model.

The synthetic ranges intentionally follow the reference tables and include
overlap + noise so the XGBoost classifier has to learn a combination of
temperature_liquid + co2 + ph instead of a single-variable threshold rule.

Reference tables
----------------
pH:
  Very Acidic:      < 3.0
  Acidic:           3.0-3.9
  Ideal:            4.0-4.5
  Slightly Acidic:  4.6-5.0
  Neutral:          5.1-7.0
  Alkaline:         > 7.0

Temperature (C):
  Too Low:  < 20
  Ideal:    20-30
  Too High: > 30

CO2 (ppm):
  Low:     < 500
  Normal:  500-1000
  Active:  1000-2000
  High:    > 2000

Fermentation stages (5):
  NOT_STARTED
  STARTING_FERMENTATION
  ACTIVE_FERMENTATION
  OPTIMAL_FERMENTATION
  OVER_FERMENTED
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
out_path = Path("data") / "dataset_kombucha_2000.csv"

# (stage, n_obs, ph_band, co2_band, temp_band)
# Ranges follow the reference tables but include overlap so the model must
# use a combination of features instead of a single pH threshold.
STAGE_SPECS = [
    # Not started - pH still near neutral (5.1-7.0), CO2 low/normal, temp moderate
    ("NOT_STARTED", 400, (5.3, 6.9), (420, 700), (20.0, 28.0)),
    # Starter culture pitched - pH turns slightly acidic, CO2 begins to climb
    ("STARTING_FERMENTATION", 400, (4.5, 5.2), (600, 950), (21.0, 29.0)),
    # Active souring - pH in the acidic/ideal band, CO2 actively produced
    ("ACTIVE_FERMENTATION", 500, (3.9, 4.6), (1100, 1900), (23.0, 30.0)),
    # Peak healthy activity - pH in the ideal band (4.0-4.5), CO2 high
    ("OPTIMAL_FERMENTATION", 450, (4.0, 4.5), (1600, 2300), (23.0, 30.0)),
    # Over/ruined - pH too acidic, CO2 very high, temp drifts out of range
    ("OVER_FERMENTED", 250, (2.4, 3.4), (2000, 4500), (29.0, 34.0)),
]

rows: list[dict[str, float | str]] = []
for stage, count, ph_band, co2_band, temp_band in STAGE_SPECS:
    ph_lo, ph_hi = ph_band
    co2_lo, co2_hi = co2_band
    temp_lo, temp_hi = temp_band

    # Sample base values then add small noise for realistic overlap.
    ph = rng.uniform(ph_lo, ph_hi, count) + rng.normal(0, 0.09, count)
    co2 = rng.uniform(co2_lo, co2_hi, count) + rng.normal(0, 55, count)
    temp = rng.uniform(temp_lo, temp_hi, count) + rng.normal(0, 0.35, count)

    # Clamp into plausible sensor/model windows.
    ph = np.clip(ph, 2.2, 7.4)
    co2 = np.clip(co2, 350, 5000)
    temp = np.clip(temp, 18, 35)

    for t, c, p in zip(temp, co2, ph):
        rows.append({
            "temperature_liquid": round(float(t), 2),
            "co2": int(round(float(c))),
            "ph": round(float(p), 2),
            "fermentation_stage": stage,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Dataset dibuat: {out_path}")
print(f"Jumlah baris: {len(df)}")
print("\nDistribusi stage:")
print(df["fermentation_stage"].value_counts())
print("\nPreview:")
print(df.head(10).to_string(index=False))