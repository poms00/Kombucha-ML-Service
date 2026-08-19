"""Local, labelled training-dataset storage for kombucha sensor observations."""

from __future__ import annotations

import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "kombucha_fermentation_training.csv"
DATASET_COLUMNS = (
    "fermentator_id",
    "timestamp",
    "temperature_liquid",
    "co2",
    "ph",
    "fermentation_stage",
)


def append_training_sample(sample: dict) -> None:
    """Append one validated, human-labelled sensor observation to the CSV dataset."""
    stage = str(sample.get("fermentation_stage", "")).strip()
    values = (sample.get("temperature_liquid"), sample.get("co2"), sample.get("ph"))
    if not stage:
        raise ValueError("fermentation_stage wajib diisi.")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("temperature_liquid, co2, dan ph harus berupa angka yang valid.")

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not DATASET_PATH.exists() or DATASET_PATH.stat().st_size == 0
    with DATASET_PATH.open("a", newline="", encoding="utf-8") as dataset_file:
        writer = csv.DictWriter(dataset_file, fieldnames=DATASET_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: sample.get(column, "") for column in DATASET_COLUMNS})
