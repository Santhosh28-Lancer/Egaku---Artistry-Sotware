from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from egaku.sketch_engine import (
    MAX_WORKING_SIDE,
    _compose_graphite,
    _multi_scale_edges,
    _read_rgb,
    _working_copy,
)


class GraphiteFlowTests(unittest.TestCase):
    def make_test_image(self, size=(320, 400)):
        width, height = size
        image = np.full((height, width, 3), 225, np.uint8)
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (95, 165, 220), -1)
        cv2.ellipse(image, (width // 2, height // 3), (55, 72), 0, 0, 360, (170, 190, 205), -1)
        cv2.circle(image, (width // 2 - 20, height // 3 - 12), 5, (25, 25, 25), -1)
        cv2.circle(image, (width // 2 + 20, height // 3 - 12), 5, (25, 25, 25), -1)
        cv2.line(image, (width // 2 - 18, height // 3 + 30), (width // 2 + 18, height // 3 + 30), (65, 65, 65), 3)
        cv2.rectangle(image, (width // 2 - 85, height // 2), (width // 2 + 85, height - 1), (85, 105, 130), -1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_working_copy_respects_resolution_cap(self):
        large = np.zeros((2200, 3200, 3), np.uint8)
        reduced = _working_copy(large)
        self.assertLessEqual(max(reduced.shape[:2]), MAX_WORKING_SIDE)
        small = np.zeros((200, 300, 3), np.uint8)
        self.assertEqual(_working_copy(small).shape, small.shape)

    def test_multiscale_edges_are_detected(self):
        source = self.make_test_image()
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
        edges, gradient = _multi_scale_edges(gray, 65)
        self.assertEqual(edges.shape, gray.shape)
        self.assertGreater(int(np.count_nonzero(edges)), 100)
        self.assertTrue(np.isfinite(gradient).all())

    def test_graphite_pipeline_is_monochrome_and_deterministic(self):
        source = self.make_test_image()
        first = _compose_graphite(source, 64, 58)
        second = _compose_graphite(source, 64, 58)
        self.assertEqual(first.shape, source.shape)
        self.assertTrue(np.array_equal(first[:, :, 0], first[:, :, 1]))
        self.assertTrue(np.array_equal(first[:, :, 1], first[:, :, 2]))
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(int(first.std()), 10)

    def test_renderer_does_not_modify_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.fromarray(self.make_test_image()).save(path)
            original_bytes = path.read_bytes()
            _read_rgb(path)
            self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
