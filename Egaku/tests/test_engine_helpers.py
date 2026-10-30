from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from egaku.sketch_engine import (
    DEFAULT_MAX_SIDE,
    OOM_RETRY_SIZES,
    _load_image,
    _model_dimensions,
    _validate_model_path,
)


class EgakuEngineTests(unittest.TestCase):
    def test_model_dimensions_are_divisible_by_eight(self):
        width, height = _model_dimensions(960, 1280)
        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)
        self.assertLessEqual(max(width, height), DEFAULT_MAX_SIDE)

    def test_aspect_ratio_is_preserved(self):
        width, height = _model_dimensions(960, 1280)
        self.assertAlmostEqual(width / height, 960 / 1280, delta=0.02)

    def test_small_image_is_not_upscaled(self):
        self.assertEqual(_model_dimensions(320, 240), (320, 240))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            _model_dimensions(0, 500)

    def test_oom_retry_sizes_descend(self):
        self.assertEqual(tuple(sorted(OOM_RETRY_SIZES, reverse=True)), OOM_RETRY_SIZES)
        self.assertEqual(OOM_RETRY_SIZES[-1], 384)

    def test_load_image_limits_size_and_preserves_original_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "portrait.png"
            Image.new("RGB", (960, 1280), "white").save(source)
            image, original = _load_image(source)
            self.assertEqual(original, (960, 1280))
            self.assertEqual(max(image.size), DEFAULT_MAX_SIDE)
            self.assertEqual(image.size[0] % 8, 0)
            self.assertEqual(image.size[1] % 8, 0)

    def test_model_folder_requires_model_index(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "model_index.json"):
                _validate_model_path(Path(temp))

    def test_model_folder_rejects_sdxl_architecture(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "model_index.json").write_text(
                json.dumps({"_class_name": "StableDiffusionXLPipeline"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "standard Stable Diffusion 1.5"):
                _validate_model_path(folder)


if __name__ == "__main__":
    unittest.main()
