# Egaku — Image to Art Studio

A local, deterministic photograph-to-graphite application. The single Pencil workflow uses a structured OpenCV pipeline. It does not generate a new subject or redraw a person's identity.

## Rendering pipeline

1. **Edge-aware cleanup:** bilateral denoising and restrained unsharp masking sharpen the photograph without deliberately amplifying flat-region noise.
2. **Structural edges:** adaptive Canny and Sobel gradients find contours, then morphological closing and connected-component filtering remove small specks.
3. **Graphite tones:** a softened luminance field creates gradual shadow and highlight transitions on a paper-like white base.
4. **Graphite flow:** a structure tensor estimates local image direction; directional hatching follows contours and a second cross-hatch is limited to deeper shadows.
5. **Pencil line finish:** OpenCV's `pencilSketch` is blended with the tonal and flow layers to add fine pencil marks without relying on it as the entire renderer.

All operations are local. No Stable Diffusion model, LoRA, PEFT, CUDA, GPU, account, or network connection is required. Original framing and dimensions are retained. The original file is never overwritten.

## Requirements

- Windows 10/11, 64-bit
- Python 3.12, 64-bit
- OpenCV, NumPy, Pillow and PySide6 (installed by setup)
- CPU processing; an NVIDIA GPU is not required

## Install or repair

1. Extract the repository ZIP.
2. Open the `Egaku` folder.
3. Run `setup.bat`.
4. Run `run.bat`.

`setup_gpu.bat` remains as a compatibility wrapper for older instructions and calls `setup.bat`. It no longer installs PyTorch or CUDA.

## Controls

- **Graphite depth:** controls tonal darkness and depth in the shaded areas.
- **Fine detail:** controls sharpening, edge sensitivity and graphite line density.

The first render is non-generative and normally completes much faster than a diffusion model. Very large images are processed with a maximum working side of 1800 pixels, then restored to the original dimensions. The restored image has the original pixel size, but processing at a reduced working size cannot recreate detail absent from that working copy.

## Tests

Run from this directory:

```powershell
.\.venv\Scripts\python.exe -m compileall -q main.py egaku
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite checks edge detection, image dimensions, monochrome output, deterministic rendering, input preservation and PNG export. It does not promise that every photograph will look identical to a hand-drawn artist's work; stroke character depends on the source image's focus, lighting, contrast and detail.
