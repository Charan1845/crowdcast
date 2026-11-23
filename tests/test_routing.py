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


def fake_detector(n_boxes: int, tile_boxes: int = 0, full_size=(320, 240)):
    """A stand-in for PersonDetector: n boxes (top, left, bottom, right, score) on the whole photo, and optionally
    extra small people in the top-left tile only."""
    def detect(image):
        if image.size == full_size:
            rows = [[10, 10 + 8 * i, 50, 16 + 8 * i, 0.9] for i in range(n_boxes)]
        elif image.getpixel((0, 0)) == (1, 2, 3):
            rows = [[100, 4 + 6 * i, 110, 8 + 6 * i, 0.8] for i in range(tile_boxes)]
        else:
            rows = []
        return np.array(rows, dtype=np.float32).reshape(-1, 5)
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

    def test_tiles_rescue_a_full_detector(self):
        self.image.putpixel((0, 0), (1, 2, 3))   # marks the top-left tile for the fake detector
        models = {"street": fake_csrnet(30), "dense": fake_csrnet(60)}
        r = cm.count_auto(self.image, models, fake_detector(cm.DET_MAX, tile_boxes=5))
        self.assertEqual(r.method, "detector")
        self.assertEqual(r.count, cm.DET_MAX + 5)

    def test_group_overcounted_by_csrnet_uses_detector(self):
        r = self.count(street=90, dense=120, boxes=cm.GROUP_MIN)   # the boxes cover a small share of the photo
        self.assertEqual(r.method, "detector")
        self.assertEqual(r.count, cm.GROUP_MIN)

    def test_dense_crowd_beats_the_group_check(self):
        r = self.count(street=90, dense=600, boxes=cm.GROUP_MIN)   # a few faces found in a packed crowd
        self.assertEqual(r.method, "dense")
        self.assertAlmostEqual(r.count, 600, places=1)

    def test_boxes_filling_the_photo_stay_a_crowd(self):
        models = {"street": fake_csrnet(90), "dense": fake_csrnet(120)}
        def big(image):
            return np.array([[0, x, image.height, x + 30, 0.9] for x in range(0, image.width - 30, 32)], dtype=np.float32)
        self.assertEqual(cm.count_auto(self.image, models, big).method, "street")

    def test_merge_drops_duplicates_and_cut_people(self):
        boxes = np.array([[0, 0, 100, 50, 0.9], [2, 1, 100, 50, 0.8], [0, 0, 60, 50, 0.7], [0, 80, 100, 130, 0.6]],
                         dtype=np.float32)
        self.assertEqual(len(cm.merge_boxes(boxes, cm.TILE_IOU, cm.TILE_IOM)), 2)

    def test_tiles_cover_the_whole_photo(self):
        tiles = cm.tile_grid(1024, 768)
        self.assertEqual((min(t[0] for t in tiles), min(t[1] for t in tiles)), (0, 0))
        self.assertEqual((max(t[2] for t in tiles), max(t[3] for t in tiles)), (1024, 768))

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
