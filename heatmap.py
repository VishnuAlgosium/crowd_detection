"""
heatmap.py — Gaussian heatmap accumulation and rendering
"""

import cv2
import numpy as np

# Map config colormap name → OpenCV constant
COLORMAP_MAP = {
    "JET":     cv2.COLORMAP_JET,
    "HOT":     cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "TURBO":   cv2.COLORMAP_TURBO,
}


def make_gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    if size is None:
        size = int(6 * sigma + 1) | 1   # odd
    ax     = np.arange(-(size // 2), size // 2 + 1)
    gauss  = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(gauss, gauss)
    return (kernel / kernel.max()).astype(np.float32)


def add_heat(heatmap: np.ndarray, cx: int, cy: int,
             sigma: float, value: float = 1.0):
    """Stamp a Gaussian blob at (cx, cy) on the heatmap in-place."""
    h, w   = heatmap.shape
    kernel = make_gaussian_kernel(sigma)
    ks     = kernel.shape[0]
    half   = ks // 2

    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    kx0 = max(0, -x0);  kx1 = ks - max(0, x1 - w)
    ky0 = max(0, -y0);  ky1 = ks - max(0, y1 - h)
    x0  = max(0, x0);   x1  = min(w, x1)
    y0  = max(0, y0);   y1  = min(h, y1)

    if x1 > x0 and y1 > y0:
        heatmap[y0:y1, x0:x1] += kernel[ky0:ky1, kx0:kx1] * value


def render_heatmap(frame: np.ndarray, heatmap: np.ndarray,
                   alpha: float, colormap: str) -> np.ndarray:
    """Blend colour heatmap onto frame, transparent where heat=0."""
    norm = np.clip(heatmap, 0, None)
    if norm.max() > 0:
        norm = norm / norm.max()

    heat_u8  = (norm * 255).astype(np.uint8)
    cmap     = COLORMAP_MAP.get(colormap.upper(), cv2.COLORMAP_JET)
    heat_col = cv2.applyColorMap(heat_u8, cmap)
    mask     = (heat_u8 > 10).astype(np.float32)[:, :, None]

    blended = (frame * (1 - alpha * mask) +
               heat_col * alpha * mask).astype(np.uint8)
    return blended
