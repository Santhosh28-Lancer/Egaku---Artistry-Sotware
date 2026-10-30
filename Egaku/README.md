# Egaku — Image to Art Studio

A Windows desktop application for creating monochrome pencil-style artwork from photographs.

## Features

- One focused pencil illustration workflow.
- Local Stable Diffusion 1.5 img2img rendering with a pencil-style LoRA.
- PySide6 desktop interface.
- Local model inference. Egaku does not upload input photographs.
- 4 GB GPU memory profile: 512 px first attempt, then automatic retries at 448 px and 384 px after a CUDA out-of-memory error.
- Preserves the original photograph and writes output to a separate PNG file.
- Conservative img2img redraw strength (default 0.28, maximum 0.42) to reduce face, pose, clothing and accessory changes.
- Explicit same-subject prompt and restrained pencil adapter weighting.

## Supported environment

- Windows 10/11, 64-bit.
- Python 3.12, 64-bit.
- NVIDIA GPU with a current driver.
- CUDA-enabled PyTorch 2.13.0 + CUDA 13.2 wheel with matching torchvision 0.28.0.
- A complete Stable Diffusion 1.5 model folder or .safetensors / .ckpt checkpoint.
- Internet access for the one-time pencil adapter download.

The base model weights are not included. The 37.9 MB pencil adapter is downloaded on first render and verified against its SHA-256 checksum. The adapter is from imform-dev/webui.

## Clean installation or repair

1. Close every running Egaku window.
2. Open this Egaku directory in File Explorer.
3. Run setup_gpu.bat and wait for the final EGAKU ENVIRONMENT READY message.
4. Run run.bat.
5. Select a standard Stable Diffusion 1.5 model.
6. Import a photograph and render.

The setup script installs the matching CUDA-enabled PyTorch 2.13.0 and torchvision 0.28.0 wheels from the official PyTorch CUDA 13.2 wheel index, installs the pinned application dependencies, runs pip check, compiles all Python files, runs the unit tests, and executes an actual small CUDA matrix operation through the standalone egaku/verify_runtime.py diagnostic. Keeping the diagnostic in a Python file avoids Windows CMD expanding percent signs inside inline Python code. It stops if any required check fails.

If an existing .venv was created with a Python version other than 64-bit Python 3.12, rename or delete that .venv folder and run setup_gpu.bat again. It does not delete an existing environment automatically.

## Repair commands

From PowerShell in this directory:

~~~powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m py_compile main.py egaku\sketch_engine.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

If PyTorch reports that CUDA is unavailable, rerun setup_gpu.bat and check the NVIDIA driver. Do not install a CPU-only PyTorch build.

## Model compatibility

The included pencil LoRA is intended for a standard Stable Diffusion 1.5 base model. SDXL, Flux, SD 2.x, another architecture, a LoRA file selected as the base model, or an incomplete checkpoint will not work. Model folders must contain a valid model_index.json.

## Low-VRAM behavior

Egaku does not move the full pipeline onto the GPU. It enables VAE slicing/tiling on the VAE component, maximum attention slicing where supported, and Diffusers sequential CPU offload. On a 4 GB GPU, generation can be slow. If CUDA runs out of memory, Egaku retries at lower input resolutions before reporting failure. Identity preservation is not guaranteed by Stable Diffusion img2img; lower redraw strength improves source retention but can make the pencil conversion subtler. Output is upscaled back to the original input dimensions; this does not restore details that were absent from the lower-resolution generation.

## Diagnostics

If a render still fails, the complete Python traceback is written to:

%LOCALAPPDATA%\Egaku\logs\last-render-error.log

The error dialog includes this path. Keep that file when reporting a problem; it identifies the failing component more accurately than a screenshot alone.

The project files are syntax-checked and have CPU-side helper tests. Actual CUDA model loading, LoRA compatibility, and image generation still need a test on the target Windows laptop with its selected SD 1.5 checkpoint and NVIDIA driver. No setup script can guarantee zero failures across arbitrary model files and driver states.
