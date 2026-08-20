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


def _probability_vector(features: np.ndarray):
    """Return the raw class-probability vector for a single feature row."""
    if isinstance(model, xgb.Booster):
        # Booster trained with multi:softprob already returns 0..1 probabilities.
        probabilities = model.predict(xgb.DMatrix(features, feature_names=list(FEATURE_NAMES)))
        # In some layouts predict may return a 1-D vector for a single class;
        # reshape so it is always 2-D (n_samples, n_classes).
        probabilities = np.atleast_2d(probabilities)
    elif hasattr(model, "predict_proba"):
        probabilities = np.atleast_2d(np.asarray(model.predict_proba(features), dtype=np.float64))
    else:
        raise ValueError("Format model tidak mendukung prediksi probabilitas.")

    return probabilities[0]


def predict_fermentation(temperature_liquid: float, co2: float, ph: float) -> dict[str, Any]:
    """Predict a fermentation stage and its model confidence.

    The class probabilities always come from the model's soft probabilities
    (``predict_proba``), so ``confidence`` is the probability of the winning
    class and ``probabilities`` mirrors that same output. No threshold rule is
    applied: the classifier combines temperature_liquid, co2, and ph.
    """
    values = (temperature_liquid, co2, ph)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("temperature_liquid, co2, dan ph harus berupa angka yang valid.")

    features = np.asarray([values], dtype=np.float32)
    classes = list(model_metadata["classes"])

    probabilities = _probability_vector(features)
    prediction_index = int(np.argmax(probabilities))
    fermentation_stage = classes[prediction_index]
    confidence = float(probabilities[prediction_index])

    # Normalize / clip to guard against tiny float drift (e.g. 0.99999997).
    confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
    prob_dict = {
        class_name: round(float(np.clip(probabilities[index], 0.0, 1.0)), 6)
        for index, class_name in enumerate(classes)
    }

    return {
        "fermentation_stage": fermentation_stage,
        "confidence": confidence,
        "probabilities": prob_dict,
        "model_version": model_metadata["model_version"],
    }