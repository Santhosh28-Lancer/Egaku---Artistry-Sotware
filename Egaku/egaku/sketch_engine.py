"""Egaku local pencil renderer.

Stable Diffusion 1.5 img2img with a verified pencil-style LoRA.
Input images remain on the local computer.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionImg2ImgPipeline

ADAPTER_REPO = "imform-dev/webui"
ADAPTER_FILE = "models/Lora/Pencil_Sketch_Style.safetensors"
ADAPTER_SHA256 = "07629a55cfdefdc72577c8fc146b03d522e741ef93401210224cb1435ec1d791"
ADAPTER_NAME = "Pencil_Sketch_Style.safetensors"
DEFAULT_MAX_SIDE = 512
OOM_RETRY_SIZES = (512, 448, 384)


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
    if path.is_file() and verify_sha256(path):
        return path
    if path.exists():
        path.unlink()
    if progress:
        progress("Downloading the pencil adapter (first run only)…")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=ADAPTER_REPO,
            filename=ADAPTER_FILE,
            local_dir=str(path.parent / "hf-cache"),
        )
        source = Path(downloaded)
        path.write_bytes(source.read_bytes())
    except Exception as exc:
        raise RuntimeError(
            "Could not download the pencil adapter. Connect to the internet once, "
            "then retry. The verified file can also be placed at: "
            f"{path}"
        ) from exc

    if not verify_sha256(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "The downloaded pencil adapter failed SHA-256 verification. "
            "The invalid file was deleted; retry the download."
        )
    return path


def _model_dimensions(width: int, height: int, max_side: int = DEFAULT_MAX_SIDE) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("The selected image has invalid dimensions.")
    scale = min(1.0, max_side / max(width, height))
    out_w = max(64, int(round((width * scale) / 8) * 8))
    out_h = max(64, int(round((height * scale) / 8) * 8))
    return out_w, out_h


def _load_image(path: str | Path, max_side: int = DEFAULT_MAX_SIDE) -> tuple[Image.Image, tuple[int, int]]:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    try:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read this image: {image_path}\n{exc}") from exc

    original_size = image.size
    width, height = _model_dimensions(*original_size, max_side=max_side)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image, original_size


def _unique_retry_sides(width: int, height: int) -> tuple[int, ...]:
    """Skip retry profiles that would produce exactly the same input dimensions."""
    sides: list[int] = []
    seen_dimensions: set[tuple[int, int]] = set()
    for max_side in OOM_RETRY_SIZES:
        dimensions = _model_dimensions(width, height, max_side=max_side)
        if dimensions not in seen_dimensions:
            seen_dimensions.add(dimensions)
            sides.append(max_side)
    return tuple(sides)


def _check_runtime() -> None:
    """Fail early with actionable messages before loading several GB of weights."""
    try:
        import accelerate
        import peft
        from diffusers.utils.import_utils import is_peft_available
    except Exception as exc:
        raise RuntimeError(
            "Egaku's AI dependencies are incomplete. Close Egaku and run setup_gpu.bat."
        ) from exc

    if not is_peft_available():
        raise RuntimeError(
            "Diffusers cannot detect PEFT. Run setup_gpu.bat, then fully restart Egaku."
        )
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            "CUDA is unavailable to PyTorch. Run setup_gpu.bat and confirm that "
            "the NVIDIA driver is working."
        )

    # Run a tiny real GPU operation to catch driver/runtime issues before model loading.
    try:
        sample = torch.ones((16, 16), device="cuda", dtype=torch.float16)
        _ = sample @ sample
        torch.cuda.synchronize()
        del sample
        torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(
            "PyTorch sees the NVIDIA GPU but its CUDA compute test failed. "
            "Update/reinstall the NVIDIA driver, restart Windows, and run setup_gpu.bat again."
        ) from exc


def _validate_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Selected model does not exist: {model_path}")
    if model_path.is_dir():
        config_path = model_path / "model_index.json"
        if not config_path.is_file():
            raise ValueError(
                "The selected folder is not a Diffusers model folder because "
                "model_index.json is missing. Select a complete Stable Diffusion 1.5 "
                "Diffusers folder or choose the .safetensors/.ckpt checkpoint file instead."
            )
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"The model_index.json file is invalid: {exc}") from exc
        class_name = str(config.get("_class_name", ""))
        lowered = class_name.lower()
        if "xl" in lowered or "flux" in lowered or ("stable" in lowered and "diffusion" not in lowered):
            raise ValueError(
                f"The selected model folder identifies itself as {class_name!r}. "
                "Egaku's pencil adapter requires a standard Stable Diffusion 1.5 model."
            )
    elif model_path.suffix.lower() not in {".safetensors", ".ckpt"}:
        raise ValueError(
            "Choose a complete .safetensors or .ckpt Stable Diffusion 1.5 checkpoint."
        )
    elif model_path.stat().st_size < 100 * 1024 * 1024:
        raise ValueError(
            "The selected checkpoint is unusually small. It may be a LoRA or an incomplete "
            "download, not a full Stable Diffusion 1.5 model."
        )


class PencilRenderer:
    def __init__(
        self,
        model_path: str | Path,
        progress: Callable[[str], None] | None = None,
    ):
        self.model_path = Path(model_path).expanduser()
        self.progress = progress or (lambda _message: None)
        self.pipe: StableDiffusionImg2ImgPipeline | None = None

    def load(self) -> None:
        if self.pipe is not None:
            return

        _check_runtime()
        _validate_model_path(self.model_path)
        self.progress("Loading the Stable Diffusion 1.5 model…")

        try:
            if self.model_path.is_dir():
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
            else:
                pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    str(self.model_path),
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "The selected base model could not be loaded as Stable Diffusion 1.5. "
                "Use a complete SD 1.5 checkpoint or Diffusers folder. "
                f"Original loader message: {exc}"
            ) from exc

        try:
            adapter = ensure_adapter(self.progress)
            pipe.load_lora_weights(str(adapter), adapter_name="egaku_pencil")
            pipe.set_adapters(["egaku_pencil"], adapter_weights=[0.85])
        except Exception as exc:
            del pipe
            torch.cuda.empty_cache()
            text = str(exc).lower()
            if "peft" in text:
                raise RuntimeError(
                    "The pencil adapter needs the PEFT backend. Close all Egaku windows, "
                    "run setup_gpu.bat, and launch Egaku again."
                ) from exc
            raise RuntimeError(
                "The pencil adapter could not be applied to this base model. "
                "Select a standard Stable Diffusion 1.5 checkpoint, not SDXL, Flux, "
                "a LoRA file, or a different architecture. "
                f"Original adapter message: {exc}"
            ) from exc

        # Diffusers >=0.40 moved VAE slicing/tiling from the pipeline to the VAE.
        # Use the supported component methods, and never let an optional memory
        # optimisation crash an otherwise valid pipeline.
        vae = getattr(pipe, "vae", None)
        if vae is None:
            del pipe
            torch.cuda.empty_cache()
            raise RuntimeError("The selected Stable Diffusion model has no VAE component.")

        # These are optional memory features. If a particular VAE implementation
        # does not support one, continue with sequential CPU offload and OOM retries.
        try:
            if callable(getattr(vae, "enable_slicing", None)):
                vae.enable_slicing()
        except Exception as exc:
            self.progress(f"VAE slicing unavailable; continuing with CPU offload ({exc})")
        try:
            if callable(getattr(vae, "enable_tiling", None)):
                vae.enable_tiling()
        except Exception as exc:
            self.progress(f"VAE tiling unavailable; continuing with CPU offload ({exc})")

        # Attention slicing is optional and varies by attention backend.
        attention_slicing = getattr(pipe, "enable_attention_slicing", None)
        if callable(attention_slicing):
            try:
                attention_slicing("max")
            except Exception as exc:
                self.progress(f"Attention slicing unavailable; continuing ({exc})")

        # Do not call pipe.to('cuda') before sequential offload. That defeats much
        # of the memory saving and can exhaust a 4 GB laptop GPU.
        sequential_offload = getattr(pipe, "enable_sequential_cpu_offload", None)
        if not callable(sequential_offload):
            del pipe
            torch.cuda.empty_cache()
            raise RuntimeError(
                "This Diffusers build does not provide sequential CPU offload. "
                "Run setup_gpu.bat to install the pinned Egaku dependencies."
            )
        try:
            sequential_offload()
        except Exception as exc:
            del pipe
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Could not activate sequential CPU offload. Confirm Accelerate is installed "
                "by running setup_gpu.bat. Do not move the pipeline to CUDA manually. "
                f"Original offload message: {exc}"
            ) from exc

        self.pipe = pipe
        self.progress("Model and pencil adapter loaded. Ready to render.")

    def _generate(
        self,
        source: Image.Image,
        strength: float,
        detail: int,
        seed: int | None,
    ) -> Image.Image:
        assert self.pipe is not None
        prompt = (
            "highly detailed realistic graphite pencil portrait, professional hand-drawn "
            "pencil illustration on clean white drawing paper, precise facial anatomy, "
            "individual curved pencil strokes in the hair, finely observed hair strands, "
            "crisp eyeglass frames, subtle graphite gradients on skin, controlled cross "
            "hatching in shadow regions, delicate directional strokes following fabric folds, "
            "deep graphite accents, clean highlights, refined traditional pencil artwork"
        )
        negative = (
            "blurry, watercolor, oil painting, color, dirty gray background, random noise, "
            "mottled skin, smudged facial features, cartoon, anime, plastic skin, "
            "oversmoothed face, distorted glasses, extra features, text, watermark"
        )
        steps = int(np.interp(detail, [0, 100], [20, 30]))
        guidance = float(np.interp(detail, [0, 100], [6.0, 7.5]))
        adapter_weight = float(np.interp(detail, [0, 100], [0.65, 1.0]))
        self.pipe.set_adapters(["egaku_pencil"], adapter_weights=[adapter_weight])

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=source,
            strength=float(np.clip(strength, 0.35, 0.78)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            num_images_per_prompt=1,
        )
        return result.images[0].convert("RGB")

    def render(
        self,
        image_path: str | Path,
        output_path: str | Path,
        strength: float = 0.58,
        detail: int = 70,
        seed: int | None = None,
    ) -> Image.Image:
        if self.pipe is None:
            self.load()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as opened:
            original_size = ImageOps.exif_transpose(opened).size
        retry_sides = _unique_retry_sides(*original_size)
        last_oom_message: str | None = None

        for attempt, max_side in enumerate(retry_sides, start=1):
            source, original_size = _load_image(image_path, max_side=max_side)
            self.progress(
                f"Rendering pencil illustration at {source.width} × {source.height} "
                f"(memory profile {attempt}/{len(retry_sides)})…"
            )
            try:
                with torch.inference_mode():
                    generated = self._generate(source, strength, detail, seed)
                if generated.size != original_size:
                    generated = generated.resize(original_size, Image.Resampling.LANCZOS)

                temporary = output.with_name(output.stem + ".egaku-tmp" + output.suffix)
                generated.save(temporary)
                temporary.replace(output)
                return generated

            except torch.cuda.OutOfMemoryError as exc:
                last_oom_message = str(exc)
                gc.collect()
                torch.cuda.empty_cache()
                if attempt == len(retry_sides):
                    break
                self.progress("GPU memory was tight. Retrying automatically at a smaller size…")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                last_oom = exc
                torch.cuda.empty_cache()
                if attempt == len(OOM_RETRY_SIZES):
                    break
                self.progress("GPU memory was tight. Retrying automatically at a smaller size…")

        raise RuntimeError(
            "The 4 GB GPU ran out of memory even after Egaku retried at lower resolutions. "
            "Close other GPU-heavy applications and retry. The original photo was not changed. "
            f"Last CUDA error: {last_oom_message}"
        )
