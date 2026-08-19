import firebase_admin
from firebase_admin import credentials, db
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
cred = credentials.Certificate(PROJECT_ROOT / "firebase" / "serviceAccountKey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://kombucha-83c73-default-rtdb.asia-southeast1.firebasedatabase.app"
    })


def get_sensors(fermentator_id: str):
    ref = db.reference(
        f"fermentators/{fermentator_id}/current/sensors"
    )

    return ref.get()


def save_training_sample(fermentator_id: str, sample: dict) -> str:
    """Store a labelled observation so the training data can be audited in Firebase."""
    ref = db.reference(f"fermentators/{fermentator_id}/training_samples")
    return ref.push(sample).key


def save_analytics(fermentator_id: str, analytics: dict) -> None:
    """Save the newest model inference directly under the analytics branch."""
    ref = db.reference(f"fermentators/{fermentator_id}/analytics")
    ref.set(analytics)
