"""Tests for saving corrections. Hugging Face is faked: nothing is uploaded."""
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import feedback


class FakeApi:
    private = True
    commits = []

    def __init__(self, token=None):
        self.token = token

    def repo_info(self, repo_id, repo_type=None):
        return types.SimpleNamespace(private=FakeApi.private)

    def create_commit(self, repo_id, operations, commit_message=None, repo_type=None):
        FakeApi.commits.append({op.path_in_repo: op for op in operations})


def photo_with_exif():
    image = Image.new("RGB", (64, 48), "white")
    exif = Image.Exif()
    exif[0x010F] = "SomeCamera"
    buf = io.BytesIO()
    image.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


class SaveFeedback(unittest.TestCase):
    def setUp(self):
        FakeApi.private, FakeApi.commits = True, []
        self.result = types.SimpleNamespace(count=12.0, method="detector", street_estimate=15.26)
        patcher = mock.patch("huggingface_hub.HfApi", FakeApi)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, share):
        return feedback.save_feedback("token", "user/repo", self.result, 8, photo_with_exif(), share)

    def test_no_photo_unless_opted_in(self):
        entry = self.save(False)
        files = FakeApi.commits[0]
        self.assertEqual(list(files), [f"corrections/{entry}.json"])
        record = json.loads(files[f"corrections/{entry}.json"].path_or_fileobj)
        self.assertEqual((record["predicted_count"], record["correct_count"]), (12.0, 8))
        self.assertIsNone(record["photo_file"])

    def test_shared_photo_has_no_exif(self):
        entry = self.save(True)
        data = FakeApi.commits[0][f"photos/{entry}.jpg"].path_or_fileobj
        self.assertEqual(len(Image.open(io.BytesIO(data)).getexif()), 0)

    def test_refuses_public_dataset(self):
        FakeApi.private = False
        with self.assertRaises(RuntimeError):
            self.save(True)
        self.assertEqual(FakeApi.commits, [])


if __name__ == "__main__":
    unittest.main()
