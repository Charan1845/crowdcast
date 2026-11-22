"""Unit tests for the hybrid counting routing in app/crowd_model.py.

Uses fake CSRNets and a fake detector, so no weights are downloaded and no real model is loaded.
Run: python -m unittest discover -s tests -v
"""
import math
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import numpy as np
import tensorflow as tf
from PIL import Image

import crowd_model as cm


def fake_csrnet(count: float):
    """A stand-in for a compiled CSRNet: returns a uniform density map whose sum / DENSITY_SCALE == count."""
    def run(x):
        _, h, w, _ = x.shape
        cells = math.ceil(h / cm.DOWNSAMPLE) * math.ceil(w / cm.DOWNSAMPLE)
        shape = (1, math.ceil(h / cm.DOWNSAMPLE), math.ceil(w / cm.DOWNSAMPLE), 1)
        return tf.constant(np.full(shape, count * cm.DENSITY_SCALE / cells, dtype=np.float32))
    return run


def fake_detector(n_boxes: int):
    """A stand-in for PersonDetector: returns n person boxes (top, left, bottom, right) in pixels."""
    def detect(image):
        return np.array([[10, 10 + 8 * i, 50, 16 + 8 * i] for i in range(n_boxes)], dtype=np.float32).reshape(-1, 4)
    return detect


class CountAutoRoutingTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (320, 240), (120, 110, 100))

    def count(self, street, dense, boxes):
        models = {"street": fake_csrnet(street), "dense": fake_csrnet(dense)}
        return cm.count_auto(self.image, models, fake_detector(boxes))

    def test_small_group_uses_detector_count(self):
        r = self.count(street=12, dense=30, boxes=7)
        self.assertEqual(r.method, "detector")
        self.assertEqual(r.count, 7.0)
        self.assertAlmostEqual(r.street_estimate, 12, places=2)

    def test_full_detector_falls_back_to_street(self):
        r = self.count(street=30, dense=60, boxes=cm.DET_MAX)
        self.assertEqual(r.method, "street")
        self.assertAlmostEqual(r.count, 30, places=2)

    def test_dense_crowd_uses_dense_model(self):
        r = self.count(street=100, dense=600, boxes=0)
        self.assertEqual(r.method, "dense")
        self.assertAlmostEqual(r.count, 600, places=1)

    def test_medium_crowd_uses_street_model(self):
        r = self.count(street=100, dense=120, boxes=0)
        self.assertEqual(r.method, "street")
        self.assertAlmostEqual(r.count, 100, places=1)

    def test_boundary_small_group_max_uses_detector(self):
        r = self.count(street=cm.SMALL_GROUP_MAX, dense=80, boxes=3)
        self.assertEqual(r.method, "detector")
        self.assertEqual(r.count, 3.0)

    def test_negative_density_clamps_to_zero(self):
        r = self.count(street=-5, dense=-5, boxes=0)
        self.assertEqual(r.street_estimate, 0.0)
        self.assertEqual(r.method, "detector")
        self.assertEqual(r.count, 0.0)

    def test_overlay_matches_image_size(self):
        for street, dense, boxes in [(12, 30, 4), (100, 600, 0)]:
            self.assertEqual(self.count(street, dense, boxes).overlay.size, self.image.size)


class PrepareImageTest(unittest.TestCase):
    def test_large_image_is_shrunk_to_max_side(self):
        out, resized = cm.prepare_image(Image.new("RGB", (2048, 1000)))
        self.assertTrue(resized)
        self.assertEqual(max(out.size), cm.MAX_SIDE)
        self.assertEqual(out.size, (1024, 500))

    def test_small_image_is_untouched(self):
        out, resized = cm.prepare_image(Image.new("RGB", (800, 600)))
        self.assertFalse(resized)
        self.assertEqual(out.size, (800, 600))

    def test_non_rgb_is_converted(self):
        out, _ = cm.prepare_image(Image.new("RGBA", (100, 100)))
        self.assertEqual(out.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
