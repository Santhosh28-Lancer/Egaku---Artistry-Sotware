from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from egaku.sketch_engine import (
    DEFAULT_MAX_SIDE,
    OOM_RETRY_SIZES,
    _configure_low_vram,
    _load_image,
    _model_dimensions,
    _unique_retry_sides,
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

    def test_retry_skips_duplicate_profiles_for_small_images(self):
        self.assertEqual(_unique_retry_sides(320, 240), (512,))
        self.assertEqual(_unique_retry_sides(960, 1280), (512, 448, 384))

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



    def test_diffusers_detects_peft_backend(self):
        import peft
        from diffusers.utils.import_utils import is_peft_available

        self.assertTrue(is_peft_available())
        self.assertTrue(peft.__version__)

    def test_low_vram_uses_vae_component_methods_and_pipeline_offload(self):
        class FakeVAE:
            def __init__(self):
                self.calls = []

            def enable_slicing(self):
                self.calls.append("vae_slicing")

            def enable_tiling(self):
                self.calls.append("vae_tiling")

        class FakePipeline:
            def __init__(self):
                self.vae = FakeVAE()
                self.calls = []

            def enable_attention_slicing(self, mode):
                self.calls.append(("attention", mode))

            def enable_sequential_cpu_offload(self):
                self.calls.append("sequential_cpu_offload")

        pipe = FakePipeline()
        _configure_low_vram(pipe)

        self.assertEqual(pipe.vae.calls, ["vae_slicing", "vae_tiling"])
        self.assertIn(("attention", "max"), pipe.calls)
        self.assertIn("sequential_cpu_offload", pipe.calls)
        self.assertFalse(hasattr(pipe, "enable_vae_slicing"))

    def test_low_vram_requires_sequential_offload(self):
        class FakePipeline:
            def __init__(self):
                self.vae = object()

        with self.assertRaisesRegex(RuntimeError, "sequential CPU offload"):
            _configure_low_vram(FakePipeline())


    def test_setup_uses_standalone_gpu_diagnostic(self):
        setup = (Path(__file__).resolve().parents[1] / "setup_gpu.bat").read_text(
            encoding="utf-8"
        )
        self.assertIn("python.exe egaku\\verify_runtime.py", setup)
        self.assertNotIn("VRAM: %.2f", setup)

    def test_runtime_diagnostic_script_is_present(self):
        script = Path(__file__).resolve().parents[1] / "egaku" / "verify_runtime.py"
        self.assertTrue(script.is_file())


if __name__ == "__main__":
    unittest.main()
