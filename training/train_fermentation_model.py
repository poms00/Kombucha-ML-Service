"""Train and save the kombucha fermentation-stage XGBoost model.

The input CSV must contain one row per historical observation with these columns:
temperature_liquid,co2,ph,fermentation_stage
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb


FEATURE_NAMES = ("temperature_liquid", "co2", "ph")
LABEL_NAME = "fermentation_stage"
LABEL_ALIASES = (LABEL_NAME, "stage")
MINIMUM_SAMPLES = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "kombucha_xgb.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "kombucha_xgb_metadata.json"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    if not csv_path.is_file():
        raise ValueError(f"File dataset tidak ditemukan: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as dataset_file:
        reader = csv.DictReader(dataset_file)
        feature_columns = set(FEATURE_NAMES)
        label_column = next(
            (column for column in LABEL_ALIASES if reader.fieldnames and column in reader.fieldnames),
            None,
        )
        if not reader.fieldnames or not feature_columns.issubset(reader.fieldnames) or not label_column:
            raise ValueError(
                "Dataset harus memiliki kolom: "
                + ", ".join(FEATURE_NAMES)
                + f", serta salah satu label: {', '.join(LABEL_ALIASES)}"
            )

        features: list[list[float]] = []
        labels: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                feature_row = [float(row[name]) for name in FEATURE_NAMES]
            except (TypeError, ValueError) as error:
                raise ValueError(f"Nilai fitur tidak valid pada baris {line_number}.") from error

            label = (row.get(label_column) or "").strip()
            if not label:
                raise ValueError(f"Label {label_column} kosong pada baris {line_number}.")
            if not np.isfinite(feature_row).all():
                raise ValueError(f"Nilai fitur harus terbatas pada baris {line_number}.")

            features.append(feature_row)
            labels.append(label)

    if len(features) < MINIMUM_SAMPLES:
        raise ValueError(
            f"Dataset hanya berisi {len(features)} baris; diperlukan minimal {MINIMUM_SAMPLES} observasi berlabel."
        )
    if len(set(labels)) < 2:
        raise ValueError("Dataset memerlukan setidaknya dua kelas fermentation_stage.")

    return np.asarray(features, dtype=np.float32), labels


def stratified_split(labels: np.ndarray, test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        test_size = max(1, round(len(indices) * test_fraction)) if len(indices) > 1 else 0
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])

    if not test_indices or not train_indices:
        raise ValueError("Setiap kelas harus memiliki cukup observasi untuk evaluasi.")
    return np.asarray(train_indices), np.asarray(test_indices)


def train_booster(features: np.ndarray, labels: np.ndarray, num_classes: int) -> xgb.Booster:
    training_data = xgb.DMatrix(
        features,
        label=labels,
        feature_names=list(FEATURE_NAMES),
    )
    return xgb.train(
        params={
            "objective": "multi:softprob",
            "num_class": num_classes,
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "seed": 42,
            "nthread": 1,
        },
        dtrain=training_data,
        num_boost_round=300,
    )


def train(dataset_path: Path, model_path: Path, metadata_path: Path) -> dict:
    features, raw_labels = load_dataset(dataset_path)
    classes = sorted(set(raw_labels))
    label_map = {label: index for index, label in enumerate(classes)}
    labels = np.asarray([label_map[label] for label in raw_labels], dtype=np.int32)
    train_indices, test_indices = stratified_split(labels)

    evaluation_model = train_booster(features[train_indices], labels[train_indices], len(classes))
    test_data = xgb.DMatrix(features[test_indices], feature_names=list(FEATURE_NAMES))
    predictions = np.argmax(evaluation_model.predict(test_data), axis=1)
    holdout_accuracy = float(np.mean(predictions == labels[test_indices]))

    final_model = train_booster(features, labels, len(classes))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(final_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "model_version": datetime.now(UTC).strftime("kombucha-xgb-%Y%m%dT%H%M%SZ"),
        "feature_names": list(FEATURE_NAMES),
        "label_name": LABEL_NAME,
        "classes": classes,
        "training_rows": len(features),
        "holdout_rows": len(test_indices),
        "holdout_accuracy": round(holdout_accuracy, 4),
        "trained_at": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model tahap fermentasi kombucha.")
    parser.add_argument("--dataset", required=True, type=Path, help="CSV data historis berlabel.")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    try:
        metadata = train(args.dataset, args.model_output, args.metadata_output)
    except ValueError as error:
        print(f"Training gagal: {error}", file=sys.stderr)
        return 1

    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
