from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from egaku.sketch_engine import (
    MAX_WORKING_SIDE,
    PencilRenderer,
    _working_size,
    detect_structural_edges,
    edge_aware_sharpen,
    graphite_render,
    load_rgb,
)


class GraphiteEngineTests(unittest.TestCase):
    def _sample(self, width: int = 320, height: int = 240) -> np.ndarray:
        image = np.full((height, width, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (width - 1, height - 1), (95, 165, 220), -1)
        cv2.ellipse(image, (width // 2, height // 3), (55, 72), 0, 0, 360, (170, 190, 205), -1)
        cv2.circle(image, (width // 2 - 20, height // 3 - 12), 5, (25, 25, 25), -1)
        cv2.circle(image, (width // 2 + 20, height // 3 - 12), 5, (25, 25, 25), -1)
        cv2.line(image, (width // 2 - 18, height // 3 + 30), (width // 2 + 18, height // 3 + 30), (65, 65, 65), 3)
        cv2.rectangle(image, (width // 2 - 85, height // 2), (width // 2 + 85, height - 1), (85, 105, 130), -1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def test_working_size_preserves_aspect_and_caps_long_side(self):
        width, height = _working_size(4000, 3000)
        self.assertEqual(max(width, height), MAX_WORKING_SIDE)
        self.assertAlmostEqual(width / height, 4 / 3, delta=0.002)

    def test_working_size_rejects_zero_dimensions(self):
        with self.assertRaises(ValueError):
            _working_size(0, 100)

    def test_edge_aware_sharpen_preserves_shape_and_dtype(self):
        gray = cv2.cvtColor(self._sample(), cv2.COLOR_RGB2GRAY)
        sharpened = edge_aware_sharpen(gray, detail=72)
        self.assertEqual(sharpened.shape, gray.shape)
        self.assertEqual(sharpened.dtype, np.uint8)
        self.assertGreater(int(sharpened.max()), int(sharpened.min()))

    def test_edge_detection_returns_binary_contours(self):
        gray = cv2.cvtColor(self._sample(), cv2.COLOR_RGB2GRAY)
        edges = detect_structural_edges(edge_aware_sharpen(gray), detail=70)
        self.assertEqual(edges.shape, gray.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(edges), 100)
        self.assertLess(np.count_nonzero(edges), edges.size * 0.25)
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))

    def test_graphite_render_preserves_shape_and_is_monochrome(self):
        rgb = self._sample()
        output = graphite_render(rgb, graphite_depth=65, fine_detail=58)
        self.assertEqual(output.shape, rgb.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.array_equal(output[..., 0], output[..., 1]))
        self.assertTrue(np.array_equal(output[..., 1], output[..., 2]))

    def test_graphite_render_is_deterministic_and_does_not_modify_input(self):
        rgb = self._sample()
        original = rgb.copy()
        first = graphite_render(rgb, 64, 58)
        second = graphite_render(rgb, 64, 58)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(rgb, original))

    def test_loader_caps_working_resolution_and_keeps_original_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "large.png"
            Image.fromarray(self._sample(2200, 1400)).save(path)
            rgb, original_size = load_rgb(path)
            self.assertEqual(original_size, (2200, 1400))
            self.assertLessEqual(max(rgb.shape[:2]), MAX_WORKING_SIDE)

    def test_renderer_writes_png_at_original_dimensions_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.png"
            output = Path(temp) / "result.png"
            Image.fromarray(self._sample()).save(source)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            statuses = []
            result = PencilRenderer(progress=statuses.append).render(source, output, 64, 58)
            after = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(output.is_file())
            with Image.open(output) as saved:
                self.assertEqual(saved.size, (320, 240))
                self.assertEqual(saved.mode, "RGB")
            self.assertEqual(result.size, (320, 240))
            self.assertTrue(any("Sharpening" in message for message in statuses))
            self.assertTrue(any("edge" in message.lower() for message in statuses))


if __name__ == "__main__":
    unittest.main()
