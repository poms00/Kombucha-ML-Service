"""Load the trained kombucha model and expose a stable prediction function."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "kombucha_xgb.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "kombucha_xgb_metadata.json"
FEATURE_NAMES = ("temperature_liquid", "co2", "ph")


def _load_model() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Model atau metadata kombucha belum tersedia.")

    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if tuple(metadata.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Metadata model tidak sesuai dengan fitur sensor saat ini.")

    return joblib.load(MODEL_PATH), metadata


model, model_metadata = _load_model()


def predict_fermentation(temperature_liquid: float, co2: float, ph: float) -> dict[str, Any]:
    """Predict a fermentation stage and its model confidence."""
    values = (temperature_liquid, co2, ph)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("temperature_liquid, co2, dan ph harus berupa angka yang valid.")

    features = np.asarray([values], dtype=np.float32)
    classes = model_metadata["classes"]

    if isinstance(model, xgb.Booster):
        probabilities = model.predict(xgb.DMatrix(features, feature_names=list(FEATURE_NAMES)))[0]
        prediction_index = int(np.argmax(probabilities))
        fermentation_stage = classes[prediction_index]
    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        prediction_index = int(np.argmax(probabilities))
        fermentation_stage = str(model.predict(features)[0])
    else:
        raise ValueError("Format model tidak mendukung prediksi probabilitas.")

    return {
        "fermentation_stage": fermentation_stage,
        "confidence": round(float(probabilities[prediction_index]), 4),
        "model_version": model_metadata["model_version"],
    }
