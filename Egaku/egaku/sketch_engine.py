"""Deterministic OpenCV graphite-flow renderer for Egaku.

Pipeline: edge-aware denoise -> unsharp mask -> multi-scale edges -> tonal
mapping -> structure-tensor-guided graphite strokes -> controlled edge finish.
No generative model is used, so the source subject cannot be replaced.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_WORKING_SIDE = 1600


def _read_rgb(path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input image does not exist: {source}")
    try:
        with Image.open(source) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Exception as exc:
        raise ValueError(f"Could not read image {source}: {exc}") from exc
    h, w = rgb.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("The image is too small to render.")
    return rgb, (w, h)


def _working_size(width: int, height: int, max_side: int = MAX_WORKING_SIDE) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    scale = min(1.0, max_side / max(width, height))
    return max(2, round(width * scale)), max(2, round(height * scale))


def load_rgb(path: str | Path, max_side: int = MAX_WORKING_SIDE) -> tuple[np.ndarray, tuple[int, int]]:
    rgb, original_size = _read_rgb(path)
    width, height = _working_size(*original_size, max_side=max_side)
    if (width, height) != original_size:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    return rgb, original_size


def _working_copy(rgb: np.ndarray, max_side: int = MAX_WORKING_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    nw, nh = _working_size(w, h, max_side=max_side)
    if (nw, nh) == (w, h):
        return rgb.copy()
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _unsharp_luminance(gray: np.ndarray, detail: int) -> np.ndarray:
    """Sharpen fine structures without ringing flat sky/skin regions."""
    amount = 0.55 + 0.75 * (detail / 100.0)
    sigma = 0.75 + 0.65 * (detail / 100.0)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    smooth = cv2.bilateralFilter(sharp, d=5, sigmaColor=22, sigmaSpace=3)
    return cv2.addWeighted(sharp, 0.72, smooth, 0.28, 0)


def edge_aware_sharpen(gray: np.ndarray, detail: int = 68) -> np.ndarray:
    return _unsharp_luminance(gray, int(np.clip(detail, 0, 100)))


def _multi_scale_edges(gray: np.ndarray, detail: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a thin structural edge map and normalized gradient strength."""
    median = float(np.median(gray))
    low = int(max(24, min(90, 0.62 * median)))
    high = int(max(low + 45, min(210, 1.42 * median)))
    fine = cv2.Canny(gray, low, high, L2gradient=True)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.25)
    low2 = max(18, int(low * 0.72))
    high2 = max(low2 + 35, int(high * 0.82))
    broad = cv2.Canny(blur, low2, high2, L2gradient=True)
    combined = cv2.max(broad, cv2.multiply(fine, 0.72))

    # Remove tiny isolated responses (sensor grain, JPEG blocks and fabric
    # weave) while retaining connected facial features and silhouette contours.
    binary = (combined > 24).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_component = max(6, int((gray.shape[0] * gray.shape[1]) * 0.000004))
    keep = np.zeros(n_labels, dtype=np.uint8)
    if n_labels > 1:
        keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= min_component).astype(np.uint8)
    edges = (keep[labels] * 255).astype(np.uint8)

    k = 1 if detail < 35 else 2
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    scale = float(np.percentile(mag, 98.5)) if np.any(mag) else 1.0
    grad = np.clip(mag / max(scale, 1.0), 0.0, 1.0)

    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            edges = cv2.ximgproc.thinning(edges)
    except cv2.error:
        pass
    return edges, grad


def detect_structural_edges(sharpened: np.ndarray, detail: int = 68) -> np.ndarray:
    return _multi_scale_edges(sharpened, int(np.clip(detail, 0, 100)))[0]


def _paper_background(rgb: np.ndarray, gray: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Find blue sky and conservative border-connected smooth paper regions."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    blue_sky = ((hue >= 78) & (hue <= 132) & (sat >= 48) & (val >= 90)).astype(np.uint8)
    blue_sky = cv2.morphologyEx(
        blue_sky, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    )
    blue_sky = cv2.GaussianBlur(blue_sky.astype(np.float32), (0, 0), 3.0)

    local_texture = cv2.GaussianBlur(grad, (0, 0), 5.0)
    flat = ((local_texture < 0.052) & (gray > 105)).astype(np.uint8)
    flat = cv2.morphologyEx(
        flat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    _, labels = cv2.connectedComponents(flat, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]
    border_bg = np.isin(labels, border_labels).astype(np.float32)
    border_bg = cv2.GaussianBlur(border_bg, (0, 0), 4.0)
    return np.clip(np.maximum(blue_sky, border_bg * 0.62), 0.0, 1.0)


def _graphite_tone(gray: np.ndarray, depth: int, background: np.ndarray) -> np.ndarray:
    """Smooth graphite value mapping with paper-white highlights."""
    clip = 1.25 + 0.65 * (depth / 100.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    contrast = cv2.bilateralFilter(contrast, 7, 28, 5)

    x_points = np.array([0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 255], np.float32)
    y_points = np.array([12, 30, 54, 82, 112, 143, 172, 198, 220, 238, 249, 255], np.float32)
    tone = np.interp(contrast.astype(np.float32), x_points, y_points)
    depth_gain = 0.88 + 0.24 * (depth / 100.0)
    tone = 255.0 - (255.0 - tone) * depth_gain
    tone = tone * (1.0 - 0.96 * background) + 251.0 * (0.96 * background)
    tone = np.where(contrast > 235, np.maximum(tone, 246.0), tone)
    return np.clip(tone, 0, 255).astype(np.uint8)


def _structure_tensor_flow(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate a coherent tangent field from a smoothed structure tensor."""
    base = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32) / 255.0
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 4.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 4.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 4.0)
    angle_normal = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    tangent = angle_normal + np.pi / 2.0
    strength = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (strength + 1e-6)
    return (
        np.cos(tangent).astype(np.float32),
        np.sin(tangent).astype(np.float32),
        np.clip(coherence, 0, 1),
    )


def _trace_flow(
    x0: float, y0: float, vx0: float, vy0: float,
    flow_x: np.ndarray, flow_y: np.ndarray,
    valid_mask: np.ndarray, steps: int, step_size: float = 1.65,
) -> list[tuple[int, int]]:
    h, w = valid_mask.shape
    x, y = x0, y0
    vx, vy = vx0, vy0
    points: list[tuple[int, int]] = [(int(round(x)), int(round(y)))]
    for _ in range(steps):
        ix, iy = int(round(x)), int(round(y))
        if ix < 1 or iy < 1 or ix >= w - 1 or iy >= h - 1 or valid_mask[iy, ix] == 0:
            break
        tx, ty = float(flow_x[iy, ix]), float(flow_y[iy, ix])
        if tx * vx + ty * vy < 0:
            tx, ty = -tx, -ty
        nx, ny = 0.68 * vx + 0.32 * tx, 0.68 * vy + 0.32 * ty
        norm = max((nx * nx + ny * ny) ** 0.5, 1e-6)
        vx, vy = nx / norm, ny / norm
        x += vx * step_size
        y += vy * step_size
        points.append((int(round(x)), int(round(y))))
    return points


def _draw_graphite_flow(
    tone: np.ndarray, gray: np.ndarray, depth: int, detail: int, background: np.ndarray
) -> np.ndarray:
    """Draw short curved pencil marks aligned to coherent local image structure."""
    h, w = tone.shape
    if min(h, w) < 16:
        return tone.copy()

    flow_x, flow_y, coherence = _structure_tensor_flow(gray)
    darkness = (255.0 - tone.astype(np.float32)) / 255.0
    shadow_mask = (
        (tone < (246 - int(depth * 0.12))) & (darkness > 0.06) & (background < 0.40)
    ).astype(np.uint8)
    shadow_mask = cv2.morphologyEx(
        shadow_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    rng = np.random.default_rng(7319)
    spacing = int(round(np.interp(detail, [0, 100], [18, 9])))
    max_steps = int(round(np.interp(detail, [0, 100], [5, 13])))
    stroke_alpha = float(np.interp(depth, [0, 100], [0.12, 0.25]))
    secondary_alpha = stroke_alpha * 0.46
    stroke_mask = np.zeros((h, w), dtype=np.uint8)
    cross_mask = np.zeros((h, w), dtype=np.uint8)

    y_offset = int(rng.integers(0, spacing))
    x_offset = int(rng.integers(0, spacing))
    for y in range(y_offset, h, spacing):
        for x in range(x_offset, w, spacing):
            if shadow_mask[y, x] == 0:
                continue
            if rng.random() > (0.35 + 0.62 * float(darkness[y, x])):
                continue
            vx, vy = float(flow_x[y, x]), float(flow_y[y, x])
            if coherence[y, x] < 0.12:
                if tone[y, x] >= 122 or rng.random() > 0.08:
                    continue
                vx, vy = 0.86, 0.51
            length = int(max(3, max_steps * (0.65 + 0.7 * float(darkness[y, x]))))
            forward = _trace_flow(x, y, vx, vy, flow_x, flow_y, shadow_mask, length)
            backward = _trace_flow(x, y, -vx, -vy, flow_x, flow_y, shadow_mask, length // 2)
            points = list(reversed(backward[1:])) + forward
            if len(points) >= 3:
                ink = int(np.clip(255 * stroke_alpha * (0.55 + darkness[y, x]), 8, 110))
                cv2.polylines(stroke_mask, [np.asarray(points, np.int32)], False, ink, 1, cv2.LINE_AA)

            if tone[y, x] < 122 and rng.random() < 0.26:
                angle = np.arctan2(vy, vx) + 0.95
                cx, cy = float(np.cos(angle)), float(np.sin(angle))
                q1 = (int(x - cx * length * 0.7), int(y - cy * length * 0.7))
                q2 = (int(x + cx * length * 0.7), int(y + cy * length * 0.7))
                ink2 = int(np.clip(255 * secondary_alpha * (0.55 + darkness[y, x]), 5, 62))
                cv2.line(cross_mask, q1, q2, ink2, 1, cv2.LINE_AA)

    overlay = np.maximum(stroke_mask, cross_mask).astype(np.float32) / 255.0
    result = tone.astype(np.float32) * (1.0 - overlay * 0.48)
    return np.clip(result, 0, 255).astype(np.uint8)


def _compose_graphite(rgb: np.ndarray, graphite_depth: int, fine_detail: int) -> np.ndarray:
    """Full deterministic OpenCV pipeline. Input and output are RGB uint8 arrays."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an 8-bit RGB image with three channels.")
    depth = int(np.clip(graphite_depth, 0, 100))
    detail = int(np.clip(fine_detail, 0, 100))

    gray_original = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = _unsharp_luminance(gray_original, detail)
    edges, grad = _multi_scale_edges(gray, detail)
    background = _paper_background(rgb, gray, grad)
    tone = _graphite_tone(gray, depth, background)
    flow = _draw_graphite_flow(tone, gray, depth, detail, background)

    edge_strength = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 0.65)
    edge_alpha = edge_strength * (0.18 + 0.24 * grad) * (1.0 - 0.96 * background)
    edge_alpha = np.clip(edge_alpha, 0, 0.28)
    final = flow.astype(np.float32) * (1.0 - edge_alpha) + 24.0 * edge_alpha

    # Use OpenCV pencilSketch as a restrained detail finish, not the full engine.
    sigma_s = 45 + int(detail * 0.28)
    sigma_r = 0.075 - detail * 0.00025
    try:
        pencil, _ = cv2.pencilSketch(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            sigma_s=float(sigma_s),
            sigma_r=float(np.clip(sigma_r, 0.045, 0.075)),
            shade_factor=float(0.04 + depth * 0.00018),
        )
        texture_weight = float(np.interp(detail, [0, 100], [0.025, 0.075]))
        final = final * (1.0 - texture_weight) + pencil.astype(np.float32) * texture_weight
    except cv2.error:
        pass

    paper = np.full_like(final, 250.0)
    paper_rng = np.random.default_rng(17)
    grain = paper_rng.normal(0.0, 0.65, size=final.shape).astype(np.float32)
    paper = np.clip(paper + grain, 246, 255)
    highlight = np.clip((final - 225.0) / 30.0, 0.0, 1.0) * 0.30
    final = final * (1.0 - highlight) + paper * highlight

    return np.repeat(np.clip(final, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def graphite_render(rgb: np.ndarray, graphite_depth: int = 64, fine_detail: int = 58) -> np.ndarray:
    return _compose_graphite(rgb, graphite_depth, fine_detail)


class PencilRenderer:
    """Background-worker renderer; accepts no model path and needs no GPU."""

    def __init__(self, progress: Callable[[str], None] | None = None):
        self.progress = progress or (lambda _message: None)

    def render(
        self,
        image_path: str | Path,
        output_path: str | Path,
        graphite_depth: int = 64,
        fine_detail: int = 58,
    ) -> Image.Image:
        rgb, original_size = _read_rgb(image_path)
        working = _working_copy(rgb)
        h, w = working.shape[:2]
        self.progress(f"Sharpening and detecting edges at {w} × {h}…")
        result = _compose_graphite(working, graphite_depth, fine_detail)
        if result.shape[1::-1] != original_size:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_CUBIC)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.progress("Tracing graphite flow and composing pencil tones…")
        temporary = output.with_name(output.stem + ".egaku-tmp" + output.suffix)
        Image.fromarray(result, mode="RGB").save(temporary, format="PNG")
        temporary.replace(output)
        self.progress("Graphite pencil drawing ready.")
        return Image.fromarray(result, mode="RGB")
