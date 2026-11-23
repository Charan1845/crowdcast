"""Saves "wrong count" corrections to a private Hugging Face dataset.

Needs two Streamlit secrets: HF_TOKEN (a write token) and HF_FEEDBACK_REPO (e.g. "user/crowdcast-feedback").
Without them the app simply doesn't show the feedback form. A photo is saved only if the visitor opts in.
"""
import io
import json
import uuid
from datetime import datetime, timezone

from PIL import Image


def build_record(result, correct_count: int, image: Image.Image, photo_shared: bool, entry_id: str, when: datetime) -> dict:
    """The saved row: the model's answer and the correction. No name, IP address or other visitor details."""
    return {
        "id": entry_id,
        "time_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predicted_count": round(float(result.count), 1),
        "correct_count": int(correct_count),
        "method": result.method,
        "street_estimate": round(float(result.street_estimate), 1),
        "image_width": image.width,
        "image_height": image.height,
        "photo_file": f"photos/{entry_id}.jpg" if photo_shared else None,
    }


def save_feedback(token: str, repo_id: str, result, correct_count: int, image: Image.Image, share_photo: bool) -> str:
    """Uploads one correction and returns its id. Refuses to write to a dataset that isn't private."""
    from huggingface_hub import CommitOperationAdd, HfApi   # imported here to keep the app's memory low until needed

    api = HfApi(token=token)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise RuntimeError("The feedback dataset must be private.")

    when = datetime.now(timezone.utc)
    entry_id = f"{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    record = build_record(result, correct_count, image, share_photo, entry_id, when)
    operations = [CommitOperationAdd(f"corrections/{entry_id}.json", json.dumps(record, indent=1).encode("utf-8"))]
    if share_photo:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)   # re-encoding also drops EXIF (location, device)
        operations.append(CommitOperationAdd(record["photo_file"], buf.getvalue()))
    api.create_commit(repo_id, operations, commit_message=f"Correction {entry_id}", repo_type="dataset")
    return entry_id
