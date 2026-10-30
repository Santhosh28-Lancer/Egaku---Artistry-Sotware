"""Egaku local AI pencil renderer.

Runs Stable Diffusion 1.5 img2img with a pencil-sketch LoRA, entirely on the
user's machine. It never sends an input image to a remote service.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionImg2ImgPipeline

ADAPTER_URL = "https://huggingface.co/imform-dev/webui/resolve/main/models/Lora/Pencil_Sketch_Style.safetensors"
ADAPTER_SHA256 = "07629a55cfdefdc72577c8fc146b03d522e741ef93401210224cb1435ec1d791"
ADAPTER_NAME = "Pencil_Sketch_Style.safetensors"


def adapter_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Egaku" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root / ADAPTER_NAME


def verify_sha256(path: Path) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == ADAPTER_SHA256


def ensure_adapter(progress: Callable[[str], None] | None = None) -> Path:
    path = adapter_path()
    if path.exists() and verify_sha256(path):
        return path
    if path.exists():
        path.unlink()
    if progress:
        progress("Downloading the pencil adapter (first run only)…")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="imform-dev/webui",
            filename="models/Lora/Pencil_Sketch_Style.safetensors",
            local_dir=str(path.parent / "hf-cache"),
        )
        source = Path(downloaded)
        path.write_bytes(source.read_bytes())
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pencil adapter. Connect to the internet once "
            "or place the verified adapter in %LOCALAPPDATA%\Egaku\models."
        ) from exc
    if not verify_sha256(path):
        path.unlink(missing_ok=True)
        raise RuntimeError("Pencil adapter SHA-256 verification failed.")
    return path


def _model_dimensions(width: int, height: int, limit: int = 768) -> tuple[int, int]:
    scale = min(1.0, limit / max(width, height))
    w = max(64, int(round(width * scale / 8) * 8))
    h = max(64, int(round(height * scale / 8) * 8))
    return w, h


def _restore_full_resolution(generated: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    return generated.resize(original_size, Image.Resampling.LANCZOS)


def _sky_background_mask(rgb: np.ndarray) -> np.ndarray:
    """Conservative near-white / low-saturation bright-region mask."""
    arr = rgb.astype(np.float32)
    hi = arr.max(axis=2)
    lo = arr.min(axis=2)
    brightness = arr.mean(axis=2)
    saturation = hi - lo
    return (brightness > 232) & (saturation < 30)


def _load_image(path: str | Path, max_side: int = 768) -> tuple[Image.Image, tuple[int, int]]:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    original_size = image.size
    w, h = _model_dimensions(*image.size, limit=max_side)
    if (w, h) != image.size:
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    return image, original_size


class PencilRenderer:
    def __init__(self, model_path: str | Path, progress: Callable[[str], None] | None = None):
        self.model_path = Path(model_path)
        self.progress = progress or (lambda _message: None)
        self.pipe: StableDiffusionImg2ImgPipeline | None = None

    def load(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was not detected. Egaku Local AI Pencil requires an NVIDIA GPU "
                "and a CUDA-enabled PyTorch installation."
            )
        self.progress("Loading the Stable Diffusion 1.5 model…")
        if self.model_path.is_dir():
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                str(self.model_path), torch_dtype=torch.float16, safety_checker=None,
                requires_safety_checker=False, local_files_only=True,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                str(self.model_path), torch_dtype=torch.float16, safety_checker=None,
                requires_safety_checker=False, local_files_only=True,
            )
        adapter = ensure_adapter(self.progress)
        pipe.load_lora_weights(str(adapter), adapter_name="pencil")
        pipe.set_adapters(["pencil"], adapter_weights=[0.85])
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.enable_sequential_cpu_offload()
        self.pipe = pipe

    def render(
        self, image_path: str | Path, output_path: str | Path,
        strength: float = 0.58, detail: int = 70, seed: int | None = None,
    ) -> Image.Image:
        if self.pipe is None:
            self.load()
        assert self.pipe is not None
        source, original_size = _load_image(image_path)
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        prompt = (
            "highly detailed hand drawn graphite pencil portrait, traditional realistic "
            "pencil drawing on clean white paper, fine directional pencil strokes, "
            "individual curved hair strands, carefully rendered facial anatomy, "
            "controlled cross hatching in shadows, subtle graphite gradients, "
            "crisp dark linework, delicate highlights, professional pencil illustration"
        )
        negative = (
            "blurry, watercolor, oil painting, color, gray dirty paper, smudged face, "
            "cartoon, anime, plastic skin, oversmoothed, low detail, distorted glasses, "
            "extra features, text, watermark"
        )
        steps = int(np.interp(detail, [0, 100], [24, 38]))
        guidance = float(np.interp(detail, [0, 100], [6.5, 8.0]))
        adapter_weight = float(np.interp(detail, [0, 100], [0.65, 1.0]))
        self.pipe.set_adapters(["pencil"], adapter_weights=[adapter_weight])
        self.progress("Rendering the pencil illustration…")
        result = self.pipe(
            prompt=prompt, negative_prompt=negative, image=source,
            strength=float(np.clip(strength, 0.35, 0.78)), num_inference_steps=steps,
            guidance_scale=guidance, generator=generator,
        ).images[0]
        result = _restore_full_resolution(result, original_size)
        result.save(output_path)
        return result
