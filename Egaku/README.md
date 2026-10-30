# Egaku — Image to Art Studio

A Windows desktop application for creating monochrome pencil-style artwork from photographs.

## Features

- Single pencil illustration workflow.
- Local Stable Diffusion 1.5 image-to-image rendering with a pencil-style LoRA.
- PySide6 desktop interface.
- Local model inference. Input photographs are not uploaded by Egaku.

## Requirements

- Windows 10 or 11.
- Python 3.12.
- NVIDIA GPU with a compatible CUDA driver.
- A local Stable Diffusion 1.5 model directory or checkpoint.

The base model is not bundled. The pencil adapter is downloaded on first render and verified using SHA-256.

## Install

1. Run `setup_gpu.bat`.
2. Run `run.bat`.
3. Select the local Stable Diffusion 1.5 model in Egaku.
4. Import a photo and render the pencil illustration.

See the repository root README for the project overview.
