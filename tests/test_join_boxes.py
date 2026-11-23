"""Unit tests for joining people cut in half by a tile edge (app/crowd_model.py)."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import numpy as np

import crowd_model as cm

TOP_TILE = (0, 0, 100, 60)
BOTTOM_TILE = (0, 40, 100, 100)


def boxes(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


class JoinCutBoxesTest(unittest.TestCase):
    def test_top_and_bottom_halves_become_one_person(self):
        out = cm.join_cut_boxes([(TOP_TILE, boxes([5, 20, 60, 40, 0.8])),
                                 (BOTTOM_TILE, boxes([40, 21, 95, 41, 0.6]))], 100, 100)
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0], [5, 20, 95, 41, 0.8])

    def test_neighbours_that_do_not_line_up_stay_separate(self):
        out = cm.join_cut_boxes([(TOP_TILE, boxes([5, 10, 60, 30, 0.8])),
                                 (BOTTOM_TILE, boxes([40, 50, 95, 70, 0.6]))], 100, 100)
        self.assertEqual(len(out), 2)

    def test_box_not_touching_a_tile_edge_is_not_joined(self):
        out = cm.join_cut_boxes([(TOP_TILE, boxes([5, 20, 30, 40, 0.8])),
                                 (BOTTOM_TILE, boxes([40, 20, 95, 40, 0.6]))], 100, 100)
        self.assertEqual(len(out), 2)

    def test_each_half_is_used_only_once(self):
        out = cm.join_cut_boxes([(TOP_TILE, boxes([5, 20, 60, 40, 0.8])),
                                 (BOTTOM_TILE, boxes([40, 20, 95, 40, 0.6], [40, 22, 90, 42, 0.5]))], 100, 100)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
