from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from egaku.sketch_engine import _model_dimensions, _restore_full_resolution, _sky_background_mask


class EngineHelperTests(unittest.TestCase):
    def test_model_dimensions_are_multiple_of_eight(self):
        width, height = _model_dimensions(960, 1280)
        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)
        self.assertLessEqual(max(width, height), 768)

    def test_restore_dimensions(self):
        image = Image.new("RGB", (512, 768), "white")
        restored = _restore_full_resolution(image, (960, 1280))
        self.assertEqual(restored.size, (960, 1280))

    def test_sky_mask_picks_near_white_low_saturation(self):
        arr = np.array([[[250, 250, 250], [100, 120, 130], [245, 220, 220]]], dtype=np.uint8)
        mask = _sky_background_mask(arr)
        self.assertTrue(bool(mask[0, 0]))
        self.assertFalse(bool(mask[0, 1]))
        self.assertFalse(bool(mask[0, 2]))


if __name__ == "__main__":
    unittest.main()
