import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

# Distribusi tahap fermentasi
stages = [
    ("BELUM_FERMENTASI", 400),
    ("FERMENTASI_AWAL", 500),
    ("FERMENTASI_AKTIF", 600),
    ("FERMENTASI_LANJUT", 350),
    ("MATANG", 150),
]

rows = []

# Synthetic ranges intentionally include overlap/noise so the classifier
# does not learn a perfectly separable rule.
stage_params = {
    "BELUM_FERMENTASI": dict(ph=(6.0, 7.2), co2=(400, 850), temp=(24.0, 30.0)),
    "FERMENTASI_AWAL": dict(ph=(5.0, 6.2), co2=(600, 1300), temp=(25.0, 31.0)),
    "FERMENTASI_AKTIF": dict(ph=(4.0, 5.2), co2=(900, 2200), temp=(25.0, 31.5)),
    "FERMENTASI_LANJUT": dict(ph=(3.3, 4.3), co2=(1300, 2800), temp=(24.5, 31.0)),
    "MATANG": dict(ph=(3.0, 4.0), co2=(1000, 2400), temp=(24.0, 30.0)),
}

for stage, count in stages:
    p = stage_params[stage]

    ph = rng.uniform(*p["ph"], count) + rng.normal(0, 0.10, count)
    co2 = rng.uniform(*p["co2"], count) + rng.normal(0, 90, count)
    temp = rng.uniform(*p["temp"], count) + rng.normal(0, 0.35, count)

    # Clamp to plausible sensor/model ranges
    ph = np.clip(ph, 2.8, 7.5)
    co2 = np.clip(co2, 400, 5000)
    temp = np.clip(temp, 20, 35)

    for t, c, pH in zip(temp, co2, ph):
        rows.append({
            "temperature_liquid": round(float(t), 2),
            "co2": int(round(float(c))),
            "ph": round(float(pH), 2),
            "stage": stage
        })

df = pd.DataFrame(rows)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

path = "data/dataset_kombucha_2000.csv"
df.to_csv(path, index=False)

print(f"Dataset dibuat: {path}")
print(f"Jumlah baris: {len(df)}")
print("\nDistribusi stage:")
print(df["stage"].value_counts())
print("\nPreview:")
print(df.head(10).to_string(index=False))
