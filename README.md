# Egaku — Image to Art Studio

The Windows desktop application is in `Egaku/`. See `Egaku/README.md` for installation, controls, rendering pipeline and tests.

Egaku now uses a deterministic OpenCV graphite-flow renderer: edge-aware sharpening, multi-scale contours, tonal graphite shading, structure-tensor-guided directional hatching and OpenCV pencil line finishing. It does not require AI models, CUDA or an NVIDIA GPU.
