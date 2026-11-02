"""Hand-drawn style pencil-sketch generation using OpenCV.

This module converts an input image into a soft, graphite-like pencil sketch.
It combines edge-aware line art (XDoG + dodge), tonal shading, graphite grain,
cross-hatching in shadowed regions and a subtle paper texture.
"""

import os

import cv2
import numpy as np


def _paper_texture(height, width):
    """Generate a subtle, repeatable paper-like texture (0.88..1.12)."""
    rng = np.random.default_rng(12345)
    tex = np.zeros((height, width), dtype=np.float32)
    for octave in range(4):
        scale = 2 ** octave
        nh = height // scale + 2
        nw = width // scale + 2
        noise = rng.normal(0.0, 1.0, (nh, nw)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        tex += noise / float(2 ** octave)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    tex = (tex - tex.mean()) / (tex.std() + 1e-6)
    tex = 1.0 + 0.07 * tex
    return np.clip(tex, 0.88, 1.12)


def _hatch_layer(height, width, spacing, angle):
    """Create a soft hatch-line layer for a given angle."""
    theta = np.deg2rad(angle)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coord = grid_x * np.cos(theta) + grid_y * np.sin(theta)
    lines = (np.mod(coord, spacing) < 1.0).astype(np.float32)
    lines = cv2.GaussianBlur(lines, (3, 3), 0.8)
    return lines


def generate_sketch(input_path, output_path):
    """Generate a handmade-looking pencil sketch.

    Returns the output path on success. Raises ``ValueError`` for unreadable
    images and ``RuntimeError`` for save failures.
    """
    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not read the image for sketch generation.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Preserve edges while removing small noise.
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    gray_f = gray.astype(np.float32) / 255.0
    height, width = gray_f.shape

    # --- Line art: XDoG (soft, natural edges) ---
    sigma = 1.0
    g1 = cv2.GaussianBlur(gray_f, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray_f, (0, 0), sigma * 2.0)
    diff = g1 - 0.98 * g2
    xdog = np.where(diff >= 0.01, 1.0, 1.0 + np.tanh(40.0 * (diff - 0.01)))
    xdog = np.clip(xdog, 0.0, 1.0)

    # --- Line art: dodge blend (stronger core lines) ---
    inverted = 1.0 - gray_f
    blurred = cv2.GaussianBlur(inverted, (0, 0), 2.0)
    denom = np.clip(1.0 - blurred, 1e-3, 1.0)
    dodge = np.clip(gray_f / denom, 0.0, 1.0)
    dodge = np.power(dodge, 0.9)

    # Blend the two line sources so outlines stay soft, not harsh.
    lines = 0.55 * xdog + 0.45 * dodge
    lines = np.clip(lines, 0.0, 1.0)
    lines = cv2.GaussianBlur(lines, (3, 3), 0.6)

    # --- Tonal shading: darker original areas receive more graphite ---
    base_tone = gray_f * 0.85 + 0.15
    sketch = lines * base_tone

    # --- Graphite grain ---
    rng = np.random.default_rng(999)
    grain = rng.normal(0.0, 0.05, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 0.8)
    density = 1.0 - gray_f
    grain_factor = 1.0 + grain * (0.4 + 0.6 * density)
    sketch = sketch * grain_factor

    # --- Cross-hatching in shadowed regions ---
    dark_mask = np.clip((0.5 - gray_f) / 0.5, 0.0, 1.0)
    hatch1 = _hatch_layer(height, width, spacing=7, angle=35)
    hatch2 = _hatch_layer(height, width, spacing=7, angle=125)
    cross = np.maximum(hatch1, hatch2)
    cross = cv2.GaussianBlur(cross, (3, 3), 0.8)
    hatch_strength = dark_mask * 0.35
    sketch = sketch * (1.0 - cross * hatch_strength)

    # --- Paper texture ---
    paper = _paper_texture(height, width)
    sketch = sketch * paper

    sketch = np.clip(sketch, 0.0, 1.0)
    sketch = (sketch * 255.0).round().astype(np.uint8)

    # Gentle contrast polish.
    sketch = cv2.convertScaleAbs(sketch, alpha=1.05, beta=-3)

    ok = cv2.imwrite(output_path, sketch)
    if not ok:
        raise RuntimeError("Failed to save the pencil-sketch image to disk.")
    return output_path
