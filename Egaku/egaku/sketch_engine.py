"""Egaku's deterministic OpenCV graphite-flow renderer.

Pipeline: edge-aware denoising -> restrained unsharp mask -> multi-scale
Canny/Sobel contours -> tonal graphite shading -> structure-tensor-guided
hatching -> OpenCV pencil line finishing. No generative model, network, or GPU.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_WORKING_SIDE = 1800


def _working_size(width: int, height: int, max_side: int = MAX_WORKING_SIDE) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(1.0, max_side / max(width, height))
    return max(32, int(round(width * scale))), max(32, int(round(height * scale)))


def load_rgb(path: str | Path, max_side: int = MAX_WORKING_SIDE) -> tuple[np.ndarray, tuple[int, int]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    try:
        with Image.open(source) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not read image '{source}': {exc}") from exc
    original_size = (int(rgb.shape[1]), int(rgb.shape[0]))
    width, height = _working_size(*original_size, max_side=max_side)
    if (width, height) != original_size:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb, original_size


def edge_aware_sharpen(gray: np.ndarray, detail: int = 68) -> np.ndarray:
    """Denoise while retaining edges, then apply a restrained unsharp mask."""
    detail = int(np.clip(detail, 0, 100))
    sigma_color = float(np.interp(detail, [0, 100], [42, 24]))
    sigma_space = float(np.interp(detail, [0, 100], [5.5, 3.2]))
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    sigma = float(np.interp(detail, [0, 100], [1.45, 0.85]))
    blur = cv2.GaussianBlur(smooth, (0, 0), sigmaX=sigma, sigmaY=sigma)
    amount = float(np.interp(detail, [0, 100], [0.42, 0.82]))
    sharpened = cv2.addWeighted(smooth, 1.0 + amount, blur, -amount, 0)
    difference = cv2.absdiff(smooth, blur)
    threshold = float(np.interp(detail, [0, 100], [4.5, 2.4]))
    mask = cv2.GaussianBlur((difference > threshold).astype(np.float32), (0, 0), 0.7)
    result = smooth.astype(np.float32) * (1.0 - mask) + sharpened.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def _remove_tiny_edges(edges: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), 8)
    if count <= 1:
        return edges
    keep = np.zeros(count, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[label] = 1
    return (keep[labels] * 255).astype(np.uint8)


def detect_structural_edges(sharpened: np.ndarray, detail: int = 68) -> np.ndarray:
    """Combine adaptive Canny and Sobel responses; close gaps and reject specks."""
    detail = int(np.clip(detail, 0, 100))
    smooth = cv2.GaussianBlur(sharpened, (3, 3), 0.65)
    median = float(np.median(smooth))
    sensitivity = float(np.interp(detail, [0, 100], [0.88, 0.62]))
    low = int(np.clip((1.0 - sensitivity) * median, 18, 92))
    high = int(np.clip((1.0 + sensitivity) * median, low + 35, 210))
    canny = cv2.Canny(smooth, low, high, L2gradient=True)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(magnitude, np.interp(detail, [0, 100], [91, 83])))
    sobel = (magnitude >= max(12.0, threshold)).astype(np.uint8) * 255
    combined = cv2.bitwise_or(
        canny, cv2.bitwise_and(sobel, cv2.dilate(canny, np.ones((3, 3), np.uint8)))
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    min_area = max(
        3, int(round(sharpened.size * np.interp(detail, [0, 100], [0.000003, 0.0000015])))
    )
    return _remove_tiny_edges(combined, min_area)


def _structure_tensor_angle(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed gradient angle and energy-gated coherence map."""
    base = cv2.GaussianBlur(gray, (0, 0), 1.25).astype(np.float32) / 255.0
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.4)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.4)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.4)
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    strength = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    energy = jxx + jyy + 1e-6
    coherence = np.clip(strength / energy, 0.0, 1.0)
    energy_gate = np.clip((energy - 0.00028) / 0.0022, 0.0, 1.0)
    coherence *= energy_gate
    c2 = cv2.GaussianBlur(np.cos(2.0 * angle).astype(np.float32), (0, 0), 1.6)
    s2 = cv2.GaussianBlur(np.sin(2.0 * angle).astype(np.float32), (0, 0), 1.6)
    angle = 0.5 * np.arctan2(s2, c2)
    return angle, coherence


def _flow_hatch(
    darkness: np.ndarray, angle: np.ndarray, coherence: np.ndarray, detail: int
) -> np.ndarray:
    """Synthesize fine graphite marks aligned to local image structure."""
    height, width = darkness.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    detail = int(np.clip(detail, 0, 100))
    spacing = float(np.interp(detail, [0, 100], [9.2, 5.3]))
    line_width = float(np.interp(detail, [0, 100], [0.55, 0.82]))

    phase = np.mod(xx * np.cos(angle) + yy * np.sin(angle), spacing)
    distance = np.minimum(phase, spacing - phase)
    primary = np.clip((line_width - distance) / 0.55 + 0.5, 0.0, 1.0)

    cross_angle = angle + 0.92
    cross_phase = np.mod(
        xx * np.cos(cross_angle) + yy * np.sin(cross_angle), spacing * 1.48
    )
    cross_distance = np.minimum(cross_phase, spacing * 1.48 - cross_phase)
    secondary = np.clip((line_width * 0.78 - cross_distance) / 0.55 + 0.5, 0.0, 1.0)

    shadow_gate = np.clip((darkness - 0.22) / 0.48, 0.0, 1.0)
    deep_gate = np.clip((darkness - 0.43) / 0.40, 0.0, 1.0)
    flow_gate = np.clip((coherence - 0.20) / 0.62, 0.0, 1.0)
    strokes = primary * shadow_gate * flow_gate * 0.33
    strokes += secondary * deep_gate * flow_gate * 0.30
    return np.clip(strokes, 0.0, 0.52).astype(np.float32)


def graphite_render(
    rgb: np.ndarray, graphite_depth: int = 64, fine_detail: int = 68
) -> np.ndarray:
    """Convert an RGB image to monochrome graphite while preserving composition."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image with three channels.")
    depth = int(np.clip(graphite_depth, 0, 100))
    detail = int(np.clip(fine_detail, 0, 100))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.28, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    sharpened = edge_aware_sharpen(contrasted, detail)
    edges = detect_structural_edges(sharpened, detail)

    tone_source = cv2.bilateralFilter(sharpened, 7, 32, 5.0).astype(np.float32) / 255.0
    darkness = np.clip(1.0 - tone_source, 0.0, 1.0)
    tone_gamma = float(np.interp(depth, [0, 100], [1.48, 1.08]))
    tonal_weight = float(np.interp(depth, [0, 100], [0.56, 0.91]))
    tonal_darkness = np.power(darkness, tone_gamma) * tonal_weight
    graphite = 255.0 - tonal_darkness * 218.0

    highlight_start = float(np.interp(depth, [0, 100], [0.79, 0.88]))
    highlight = np.clip((tone_source - highlight_start) / (1.0 - highlight_start), 0.0, 1.0)
    graphite = graphite * (1.0 - highlight * 0.88) + 255.0 * (highlight * 0.88)

    angle, coherence = _structure_tensor_angle(sharpened)
    hatching = _flow_hatch(darkness, angle, coherence, detail)
    hatch_weight = float(np.interp(depth, [0, 100], [0.58, 1.0]))
    graphite *= (1.0 - hatching * hatch_weight)

    edge_soft = cv2.GaussianBlur(edges, (3, 3), 0.55).astype(np.float32) / 255.0
    edge_weight = float(np.interp(detail, [0, 100], [0.18, 0.36]))
    graphite -= edge_soft * edge_weight * (54.0 + depth * 0.34)

    sigma_r = float(np.interp(detail, [0, 100], [0.090, 0.060]))
    shade_factor = float(np.interp(depth, [0, 100], [0.035, 0.055]))
    pencil_lines, _ = cv2.pencilSketch(
        bgr, sigma_s=60, sigma_r=sigma_r, shade_factor=shade_factor
    )
    pencil_weight = float(np.interp(detail, [0, 100], [0.26, 0.42]))
    graphite = graphite * (1.0 - pencil_weight) + pencil_lines.astype(np.float32) * pencil_weight

    rng = np.random.default_rng(2026)
    grain = rng.normal(0.0, 1.05, graphite.shape).astype(np.float32)
    midtone_gate = (
        np.clip((darkness - 0.06) / 0.60, 0.0, 1.0)
        * np.clip((0.99 - darkness) / 0.25, 0.0, 1.0)
    )
    graphite += grain * midtone_gate

    graphite = np.clip(graphite, 0, 255).astype(np.uint8)
    graphite = cv2.GaussianBlur(graphite, (3, 3), 0.35)
    return cv2.cvtColor(graphite, cv2.COLOR_GRAY2RGB)


class PencilRenderer:
    """Background-worker renderer; accepts no model path and needs no GPU."""

    def __init__(self, progress: Callable[[str], None] | None = None):
        self.progress = progress or (lambda _message: None)

    def render(
        self,
        image_path: str | Path,
        output_path: str | Path,
        graphite_depth: int = 64,
        fine_detail: int = 68,
    ) -> Image.Image:
        self.progress("Loading photograph…")
        rgb, original_size = load_rgb(image_path)
        self.progress("Sharpening image while preserving natural edges…")
        self.progress("Detecting structural contours with multi-scale edge analysis…")
        self.progress("Building tonal graphite shading and directional flow strokes…")
        result = graphite_render(rgb, graphite_depth, fine_detail)
        if result.shape[1::-1] != original_size:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_CUBIC)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.stem + ".egaku-tmp" + output.suffix)
        Image.fromarray(result).save(temporary, format="PNG")
        temporary.replace(output)
        self.progress("Graphite pencil drawing ready.")
        gc.collect()
        return Image.fromarray(result)
