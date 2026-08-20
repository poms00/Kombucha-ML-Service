"""HTTP API and scheduled analytics for Smart Kombucha."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError, field_validator

from services.dataset_service import append_training_sample
from services.firebase_service import (
    get_fermentator_ids,
    get_sensors,
    save_analytics,
    save_sensors,
    save_training_sample,
)
from services.xgboost_service import predict_fermentation


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("kombucha_api")

WIB = timezone(timedelta(hours=7), name="WIB")


def _analytics_interval() -> int:
    try:
        return max(10, int(os.getenv("ANALYTICS_INTERVAL_SECONDS", "60")))
    except ValueError:
        return 60


FERMENTATOR_IDS = tuple(
    fermentator_id.strip()
    for fermentator_id in os.getenv("FERMENTATOR_IDS", "").split(",")
    if fermentator_id.strip()
)
ANALYTICS_INTERVAL_SECONDS = _analytics_interval()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def predicted_at() -> str:
    """Return a dashboard-friendly timestamp in Indonesia Western Time."""
    return datetime.now(WIB).strftime("%d-%m-%Y %H:%M:%S WIB")


class FermentationPredictionRequest(BaseModel):
    temperature_liquid: float = Field(ge=-20, le=100, description="Suhu cairan dalam Celsius")
    co2: float = Field(ge=0, description="Pembacaan sensor CO2")
    ph: float = Field(ge=0, le=14, description="Nilai pH cairan")


class SensorDataRequest(BaseModel):
    temperature_liquid: float = Field(ge=-20, le=100, description="Suhu cairan dalam Celsius")
    co2: float = Field(ge=0, description="Pembacaan sensor CO2")
    ph: float = Field(ge=0, le=14, description="Nilai pH cairan")
    timestamp: str | None = Field(default=None, description="Waktu pembacaan sensor")


class TrainingSampleRequest(FermentationPredictionRequest):
    fermentation_stage: str = Field(min_length=1, max_length=64)

    @field_validator("fermentation_stage")
    @classmethod
    def normalize_stage(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("fermentation_stage tidak boleh kosong.")
        return normalized


def sensor_payload(sensor_data: dict[str, Any]) -> FermentationPredictionRequest:
    """Validate Firebase readings before they reach the model."""
    # Accept both "temperature_liquid" and "temperature" as the liquid temperature field.
    temperature = sensor_data.get("temperature_liquid", sensor_data.get("temperature"))
    try:
        payload = FermentationPredictionRequest.model_validate(
            {
                "temperature_liquid": temperature,
                "co2": sensor_data.get("co2"),
                "ph": sensor_data.get("ph"),
            }
        )
    except ValidationError as error:
        raise ValueError("Data sensor belum lengkap atau nilainya tidak valid.") from error

    values = (payload.temperature_liquid, payload.co2, payload.ph)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Data sensor harus berupa angka terbatas.")
    return payload


def build_analytics(fermentator_id: str, sensor_data: dict[str, Any]) -> dict[str, Any]:
    """Build a Firebase-ready analytics record without inventing missing values."""
    prediction_time = predicted_at()
    sensor_timestamp = sensor_data.get("timestamp")

    try:
        payload = sensor_payload(sensor_data)
    except ValueError:
        return {
            "predicted_at": prediction_time,
            "fermentator_id": fermentator_id,
            "status": "INSUFFICIENT_SENSOR_DATA",
            "fermentation_stage": "WAITING_SENSOR",
            "confidence": 0.0,
            "sensor_timestamp": sensor_timestamp,
        }

    prediction = predict_fermentation(
        temperature_liquid=payload.temperature_liquid,
        co2=payload.co2,
        ph=payload.ph,
    )
    return {
        "predicted_at": prediction_time,
        "fermentator_id": fermentator_id,
        "status": "PREDICTED",
        "fermentation_stage": prediction["fermentation_stage"],
        "confidence": prediction["confidence"],
        "sensor_timestamp": sensor_timestamp,
    }


async def process_analytics(fermentator_id: str) -> None:
    """Read one fermentator, infer its stage, and save the result to Firebase."""
    try:
        sensor_data = await asyncio.to_thread(get_sensors, fermentator_id)
        if not sensor_data:
            logger.warning("[%s] Sensor data tidak ditemukan", fermentator_id)
            return

        analytics = build_analytics(fermentator_id, sensor_data)
        await asyncio.to_thread(save_analytics, fermentator_id, analytics)
        logger.info(
            "[%s] status=%s fermentation_stage=%s confidence=%s",
            fermentator_id,
            analytics["status"],
            analytics["fermentation_stage"],
            analytics["confidence"],
        )
    except Exception:
        logger.exception("[%s] Analytics gagal diproses", fermentator_id)


async def analytics_worker() -> None:
    mode = "configured IDs" if FERMENTATOR_IDS else "Firebase discovery"
    logger.info(
        "Analytics worker berjalan dengan %s (interval %s detik)",
        mode,
        ANALYTICS_INTERVAL_SECONDS,
    )
    while True:
        fermentator_ids = FERMENTATOR_IDS
        if not fermentator_ids:
            try:
                fermentator_ids = await asyncio.to_thread(get_fermentator_ids)
            except Exception:
                logger.exception("Daftar fermentator gagal dibaca dari Firebase")
                fermentator_ids = ()

        if fermentator_ids:
            await asyncio.gather(*(process_analytics(fermentator_id) for fermentator_id in fermentator_ids))
        else:
            logger.warning("Belum ada fermentator yang terdaftar untuk diproses")
        await asyncio.sleep(ANALYTICS_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = asyncio.create_task(analytics_worker(), name="kombucha-analytics-worker")
    app.state.analytics_worker = worker
    logger.info("Smart Kombucha Analytics API started")
    try:
        yield
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
        logger.info("Smart Kombucha Analytics API stopped")


app = FastAPI(
    title="Smart Kombucha Analytics API",
    version="1.0.0",
    description="API pembacaan sensor, pengumpulan data training, dan analytics fermentasi.",
    lifespan=lifespan,
)


@app.get("/", tags=["health"])
def root() -> dict[str, Any]:
    worker = getattr(app.state, "analytics_worker", None)
    return {
        "status": "online",
        "service": app.title,
        "analytics_worker": "running" if worker and not worker.done() else "stopped",
        "analytics_interval_seconds": ANALYTICS_INTERVAL_SECONDS,
    }


@app.post(
    "/fermentators/{fermentator_id}/sensors",
    status_code=status.HTTP_201_CREATED,
    tags=["sensors"],
)
def update_sensors(fermentator_id: str, payload: SensorDataRequest) -> dict[str, Any]:
    sensor_data = {
        "temperature_liquid": payload.temperature_liquid,
        "co2": payload.co2,
        "ph": payload.ph,
        "timestamp": payload.timestamp or utc_now(),
    }
    try:
        save_sensors(fermentator_id, sensor_data)
    except Exception as error:
        logger.exception("[%s] Sensor gagal disimpan ke Firebase", fermentator_id)
        raise HTTPException(status_code=503, detail="Firebase tidak dapat diakses.") from error

    return {"status": "saved", "fermentator_id": fermentator_id, "sensor": sensor_data}


@app.get("/fermentators/{fermentator_id}/sensors", tags=["sensors"])
def sensors(fermentator_id: str) -> dict[str, Any]:
    try:
        data = get_sensors(fermentator_id)
    except Exception as error:
        logger.exception("[%s] Firebase gagal dibaca", fermentator_id)
        raise HTTPException(status_code=503, detail="Firebase tidak dapat diakses.") from error

    if not data:
        raise HTTPException(status_code=404, detail="Sensor data tidak ditemukan.")

    return {
        "fermentator_id": fermentator_id,
        "temperature_liquid": data.get("temperature_liquid"),
        "co2": data.get("co2"),
        "ph": data.get("ph"),
        "timestamp": data.get("timestamp"),
    }


@app.post(
    "/fermentators/{fermentator_id}/training-samples",
    status_code=status.HTTP_201_CREATED,
    tags=["training"],
)
def create_training_sample(fermentator_id: str, payload: TrainingSampleRequest) -> dict[str, Any]:
    sample = {
        "fermentator_id": fermentator_id,
        "timestamp": utc_now(),
        **payload.model_dump(),
    }
    try:
        append_training_sample(sample)
        firebase_key = save_training_sample(fermentator_id, sample)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("[%s] Sampel training gagal disimpan ke Firebase", fermentator_id)
        raise HTTPException(status_code=503, detail="Sampel lokal tersimpan, tetapi Firebase gagal diperbarui.") from error

    return {"status": "saved", "firebase_key": firebase_key, "sample": sample}


@app.post("/fermentation/predict", tags=["prediction"])
def fermentation_prediction(payload: FermentationPredictionRequest) -> dict[str, Any]:
    try:
        prediction = predict_fermentation(**payload.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediksi model gagal")
        raise HTTPException(status_code=503, detail="Model belum siap digunakan.") from error

    return {"predicted_at": predicted_at(), **prediction}


@app.post("/fermentators/{fermentator_id}/analytics/predict", tags=["analytics"])
def predict_and_save_analytics(fermentator_id: str) -> dict[str, Any]:
    try:
        sensor_data = get_sensors(fermentator_id)
    except Exception as error:
        logger.exception("[%s] Firebase gagal dibaca", fermentator_id)
        raise HTTPException(status_code=503, detail="Firebase tidak dapat diakses.") from error

    if not sensor_data:
        raise HTTPException(status_code=404, detail="Sensor data tidak ditemukan.")

    try:
        analytics = build_analytics(fermentator_id, sensor_data)
        save_analytics(fermentator_id, analytics)
        return analytics
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("[%s] Analytics gagal disimpan", fermentator_id)
        raise HTTPException(status_code=503, detail="Analytics tidak dapat diproses.") from error
