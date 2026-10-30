# Egaku · Local AI Pencil

A Windows desktop application for turning a photograph into a monochrome pencil-style illustration. It keeps Egaku's single-pencil UI and runs the image-generation pipeline on the user's own computer.

## Project contents

The application source, setup scripts, helper tests, and full installation instructions are stored in the `Egaku-LocalAI-Pencil/` directory.

## Quick start

1. Install Python 3.12 and a current NVIDIA driver.
2. Extract or clone this repository.
3. Open `Egaku-LocalAI-Pencil`.
4. Run `setup_gpu.bat`, then `run.bat`.
5. Select a local Stable Diffusion 1.5 model and render a pencil sketch.

The base model is not included. The small pencil-style LoRA adapter is downloaded and SHA-256 verified on the first render. See `Egaku-LocalAI-Pencil/README.md` for hardware requirements, licensing notes, offline setup, and validation limitations.

The project uses local inference. User photos are not uploaded by the application.
