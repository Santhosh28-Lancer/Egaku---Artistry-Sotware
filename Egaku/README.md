# Egaku · Local AI Pencil

A Windows desktop application for turning a photograph into a monochrome pencil-style illustration. It keeps Egaku's single-pencil UI and runs the image-generation pipeline on the user's own computer.

## What changed

- Replaces the OpenCV pencil filter with Stable Diffusion 1.5 img2img plus a pencil-sketch LoRA.
- Keeps one medium only: black-and-white graphite illustration.
- Supports selecting a local base-model folder or checkpoint.
- Uses an explicit render action so slider changes do not queue multiple GPU jobs.
- Uses sequential CPU offload, VAE slicing, and VAE tiling to reduce VRAM use.
- Preserves the original input file and exports at the input dimensions.
- Input photos are processed locally and are not uploaded by the app.

## Hardware and model requirements

- Windows 10/11, Python 3.12, recent NVIDIA driver, CUDA-capable NVIDIA GPU.
- A local Stable Diffusion 1.5 model directory or full .safetensors/.ckpt checkpoint.
- Approximately 4 GB VRAM can be attempted with sequential offload, but performance may be slow and compatibility is not guaranteed.
- The base model is not bundled. The pencil adapter is downloaded once on first render and SHA-256 verified.

Adapter source: https://huggingface.co/imform-dev/webui/blob/main/models/Lora/Pencil_Sketch_Style.safetensors
Adapter download: https://huggingface.co/imform-dev/webui/resolve/main/models/Lora/Pencil_Sketch_Style.safetensors

Expected SHA-256:
`07629a55cfdefdc72577c8fc146b03d522e741ef93401210224cb1435ec1d791`

The adapter's upstream licensing may differ from this project. Check the original model terms before commercial use or redistribution. The base model retains its own license.

## Install

1. Install Python 3.12 and a current NVIDIA driver.
2. Open this directory in Windows.
3. Run `setup_gpu.bat`.
4. Run `run.bat`.
5. Select your local SD 1.5 model if it is not detected.
6. Import an image and click **Render pencil sketch**.

On first render, the app downloads the adapter (~38 MB), verifies its SHA-256, and stores it under `%LOCALAPPDATA%\Egaku\models`.

## Validation and limits

Python syntax and helper tests can be checked in a CPU-only packaging environment. Actual AI inference and Qt GUI behavior require the target Windows machine, local model weights, and CUDA driver. This build has not been visually validated on an RTX 3050 here. Pencil LoRA output varies by checkpoint, source photograph, and settings; exact reproduction of a hand-drawn reference is not guaranteed.
