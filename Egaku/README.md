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
- PEFT, required by Diffusers to load the LoRA adapter.

The base model is not bundled. The pencil adapter is downloaded on first render and verified using SHA-256.

## Install or repair an existing installation

1. Close every running Egaku window.
2. Run `setup_gpu.bat`. It installs the dependencies in `requirements.txt` and verifies that Diffusers can detect PEFT.
3. Run `run.bat`.
4. Select the local Stable Diffusion 1.5 model in Egaku.
5. Import a photo and render the pencil illustration.

To repair only the missing PEFT package from PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -U peft
```

After installing PEFT, fully close Egaku and start it again. Diffusers checks the PEFT backend when the application starts, so simply retrying the render in the already-open window is not enough.

## Model compatibility

Use a standard Stable Diffusion 1.5 model. SDXL, Flux, and other model architectures are not compatible with this SD 1.5 pencil adapter.
