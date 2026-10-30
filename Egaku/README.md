# Egaku — Image to Art Studio

A deterministic, non-generative photograph-to-graphite application. The single Pencil workflow uses a structured OpenCV pipeline. It never invents another person or changes the input identity through a generative model.

## Graphite Flow pipeline

1. **Edge-aware preparation:** bilateral smoothing reduces compression noise, then an unsharp mask strengthens genuine fine detail without simply amplifying every pixel.
2. **Multi-scale structure detection:** adaptive Canny at two scales is combined with Sobel gradient strength. Very small isolated edge components are filtered to reduce fabric/sky speckle.
3. **Graphite tonal mapping:** a controlled luminance curve retains smooth face shadows, deep hair and clothing tones, and paper-white highlights.
4. **Structure-tensor graphite flow:** local image direction is estimated with a smoothed structure tensor. Short curved strokes follow coherent local contours; cross-hatching is restricted to deeper shadows.
5. **Pencil finish:** OpenCV's `pencilSketch()` contributes a small amount of fine graphite texture. It is not used as the entire effect.

The original subject, expression, pose, clothing, accessories and framing come directly from the photograph. No AI model, LoRA, PEFT, CUDA, NVIDIA driver, internet access or model weights are required. Original input files are never overwritten.

## Requirements

- Windows 10/11, 64-bit
- Python 3.12, 64-bit
- OpenCV-contrib, NumPy, Pillow and PySide6
- CPU processing; GPU is not required

## Install

1. Extract the repository ZIP or clone this repository.
2. Open the `Egaku` folder.
3. Run `setup.bat`.
4. Run `run.bat`.

`setup_gpu.bat` remains as a compatibility wrapper for older instructions and simply calls `setup.bat`. It does not install PyTorch or CUDA.

## Controls

- **Graphite depth:** adjusts tonal darkness and shadow depth.
- **Fine detail:** adjusts edge sensitivity, sharpening, and graphite stroke density.

The default fine-detail value is intentionally moderate to avoid turning smooth skin or backgrounds into noisy scratches. The renderer is deterministic: the same photo and settings produce the same output.

Images are processed on a working copy with a maximum side of 1600 pixels for performance. The result is restored to the original pixel dimensions before saving. Upscaling preserves output dimensions but cannot recreate source detail that was not present in the working copy.

## Tests

Run from this directory:

```powershell
.\.venv\Scripts\python.exe -m compileall -q main.py egaku
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover multi-scale edges, image sizing, grayscale output, deterministic rendering, original-file preservation and PNG export dimensions. Classical computer vision cannot guarantee an identical artist-drawn result for every photograph; the final stroke character depends on the source image's focus, lighting, contrast and background complexity.
