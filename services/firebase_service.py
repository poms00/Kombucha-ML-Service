import json
import os

import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def firebase_credential() -> credentials.Base:
    """Load Railway credentials from an environment variable or local development file."""
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        try:
            return credentials.Certificate(json.loads(service_account_json))
        except json.JSONDecodeError as error:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON must contain valid JSON.") from error

    credential_path = PROJECT_ROOT / "firebase" / "serviceAccountKey.json"
    if not credential_path.is_file():
        raise RuntimeError(
            "Firebase credentials are missing. Set FIREBASE_SERVICE_ACCOUNT_JSON in the deployment environment."
        )
    return credentials.Certificate(credential_path)


cred = firebase_credential()

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://kombucha-83c73-default-rtdb.asia-southeast1.firebasedatabase.app"
    })


def get_sensors(fermentator_id: str):
    ref = db.reference(
        f"fermentators/{fermentator_id}/sensors"
    )

    return ref.get()


def save_sensors(fermentator_id: str, sensor_data: dict) -> None:
    """Write the latest sensor readings for a fermentator."""
    ref = db.reference(f"fermentators/{fermentator_id}/sensors")
    ref.set(sensor_data)


def get_fermentator_ids() -> tuple[str, ...]:
    """Return every fermentator currently registered in Firebase."""
    fermentators = db.reference("fermentators").get()
    if not isinstance(fermentators, dict):
        return ()
    return tuple(sorted(key for key, value in fermentators.items() if isinstance(key, str) and isinstance(value, dict)))


def save_training_sample(fermentator_id: str, sample: dict) -> str:
    """Store a labelled observation so the training data can be audited in Firebase."""
    ref = db.reference(f"fermentators/{fermentator_id}/training_samples")
    return ref.push(sample).key


def save_analytics(fermentator_id: str, analytics: dict) -> None:
    """Save the newest model inference directly under the analytics branch."""
    ref = db.reference(f"fermentators/{fermentator_id}/analytics")
    ref.set(analytics)
