# Egaku — Artistry Software

Egaku is a Windows desktop application that turns photographs into monochrome, graphite-style pencil illustrations using a locally run image-to-image diffusion pipeline.

## What it does

- Imports common image formats and previews the original photograph.
- Generates a pencil-style result with Stable Diffusion 1.5 img2img and a pencil-style LoRA.
- Provides adjustable artistic strength and fine-detail controls.
- Exports generated artwork to PNG, JPEG, WebP, or TIFF.
- Runs inference locally; the application does not upload source photographs.
- Uses a low-VRAM rendering strategy with sequential CPU offload and automatic resolution retries after CUDA out-of-memory errors.
- Verifies the downloaded pencil adapter with SHA-256 before loading it.

## Technology

Python 3.12 · PySide6 · PyTorch · Hugging Face Diffusers · Pillow · NumPy

## Repository layout

- [Egaku application and setup guide](./Egaku/)
- [Rendering engine](./Egaku/egaku/sketch_engine.py)
- [Unit tests](./Egaku/tests/)

## Setup

See [Egaku/README.md](./Egaku/README.md) for the supported Windows/Python/CUDA configuration, model requirements, installation and repair steps.

The Stable Diffusion base model is not bundled. A compatible local Stable Diffusion 1.5 model is required; the pencil adapter is downloaded and checksum-verified on first render.

## Validation status

The repository includes syntax checks and CPU-side helper tests in its setup workflow. Actual model loading, LoRA compatibility, CUDA rendering and output quality still require validation on the target Windows machine with the selected model and NVIDIA driver. This repository should be treated as an actively developed desktop application, not a guaranteed production release.
