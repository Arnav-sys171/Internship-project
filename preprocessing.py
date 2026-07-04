"""
preprocessing.py — Image Preprocessing for CAPTCHA Solver v2

Generates multiple cleaned variants of a CAPTCHA image to maximise
OCR accuracy across different CAPTCHA styles.

Variants:
  V1 – CLAHE + adaptive threshold  (uneven lighting)
  V2 – Otsu binarisation           (clean high-contrast)
  V3 – denoise → sharpen → Otsu    (blurry / noisy)
  V4 – inverted Otsu               (dark background)
  V5 – HSV saturation mask          (coloured characters on white)
  V6 – darkest pixels only          (dark-ink on any bg)
  V7 – Morphological cleanup        (overlapping / connected text)
  V8 – Bilateral filter + Otsu      (heavy distortion / noise)

NEW in v2:
  • Hough-line removal runs BEFORE variant generation so that
    strike-through / diagonal lines are erased from all variants.
  • Character segmentation via vertical projection for per-char OCR.
"""

from __future__ import annotations

import os
import cv2
import numpy as np


# ── helpers ───────────────────────────────────────────────────

def _upscale_if_small(img: np.ndarray, min_h: int = 120) -> np.ndarray:
    """Upscale images so OCR models have enough pixels to work with.
    
    Raised to 120px to ensure that CAPTCHAs with mixed character
    sizes (tiny superscripts/subscripts alongside normal chars)
    have enough resolution for OCR engines to detect the small ones.
    """
    h, w = img.shape[:2]
    if h < min_h:
        scale = min_h / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


def _super_upscale(img: np.ndarray) -> np.ndarray:
    """4× upscale with sharpening for CAPTCHAs with very small characters.
    
    This creates a high-resolution version where even the tiniest
    characters (subscripts, superscripts, mixed-size fonts) get
    enough pixels for the OCR engines to recognize them.
    For a 50px CAPTCHA: base upscale → 120px, then 4× → 480px.
    A tiny 8px char becomes ~76px — well within OCR range.
    """
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    # Sharpen to restore edges lost during upscaling
    sharp_kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    big = cv2.filter2D(big, -1, sharp_kernel)
    return big


# ── NEW: Hough-line removal ──────────────────────────────────

def remove_lines(img: np.ndarray) -> np.ndarray:
    """
    Detect and erase straight lines (strike-throughs, diagonals)
    using the Probabilistic Hough Line Transform.

    We paint over detected lines with the local median background
    colour so character strokes underneath are partially recovered.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape[:2]

    # Edge detection tuned for thin lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines — minLineLength relative to image width
    min_len = max(int(w * 0.25), 20)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_len, maxLineGap=5)

    if lines is None:
        return img

    result = img.copy()
    # Compute background colour (median of bright pixels)
    if len(result.shape) == 3:
        bg_color = tuple(int(v) for v in np.median(result.reshape(-1, 3), axis=0))
    else:
        bg_color = int(np.median(result))

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip very short lines and near-vertical lines (could be character strokes)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < min_len:
            continue

        # Skip lines that are likely character strokes (near vertical)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx > 0 and dy / dx > 3.0:
            continue  # very steep → probably a character like 'l', '1', '|'

        # Paint over the line with background colour
        thickness = 3  # slightly thicker than the line to fully cover it
        cv2.line(result, (x1, y1), (x2, y2), bg_color, thickness)

    return result


# ── Variant generation ───────────────────────────────────────

def preprocess_variants(image_path: str) -> list[str]:
    """
    Generate 8+1 preprocessing variants from a CAPTCHA image.

    V1–V8: Standard variants on the ORIGINAL image (no line removal).
    V9:    Line-removed + Otsu — an extra vote that helps when
           strike-through lines are present, without corrupting the
           original character data in V1–V8.

    Returns a list of file paths to the generated variant images.
    """
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]

    img = _upscale_if_small(img)

    # Do NOT apply line removal globally — it destroys character strokes
    # on many CAPTCHA styles (ECI, IRCTC).  Line removal is only used
    # for a separate V9 variant below.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = image_path.replace(".png", "")
    paths: list[str] = []

    def _save(tag: str, arr: np.ndarray) -> None:
        p = f"{base}_{tag}.png"
        cv2.imwrite(p, arr)
        paths.append(p)

    # V1 — CLAHE + adaptive threshold
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v1 = clahe.apply(gray)
    v1 = cv2.adaptiveThreshold(v1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    _save("v1", v1)

    # V2 — Otsu binarisation
    _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save("v2", v2)

    # V3 — denoise → sharpen → Otsu
    v3 = cv2.fastNlMeansDenoising(gray, h=10)
    sharp_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    v3 = cv2.filter2D(v3, -1, sharp_k)
    _, v3 = cv2.threshold(v3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save("v3", v3)

    # V4 — inverted Otsu
    _save("v4", cv2.bitwise_not(v2))

    # V5 — HSV saturation extraction (coloured characters on white)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_mask = ((sat > 40) | (val < 100)).astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)
    color_mask = cv2.dilate(color_mask, kernel, iterations=1)
    v5 = cv2.bitwise_not(color_mask)
    _save("v5", v5)

    # V6 — darkest 30% of pixels → black, rest → white
    thresh_val = int(np.percentile(gray, 30))
    _, v6 = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    _save("v6", v6)

    # V7 — Morphological cleanup for overlapping / connected characters
    v7 = v2.copy()
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    v7 = cv2.morphologyEx(v7, cv2.MORPH_OPEN, kernel_open, iterations=1)
    v7 = cv2.morphologyEx(v7, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    _save("v7", v7)

    # V8 — Bilateral filter + aggressive Otsu
    v8 = cv2.bilateralFilter(gray, 9, 75, 75)
    _, v8 = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save("v8", v8)

    # V9 — Line-removed + Otsu  (NEW: additive vote, doesn't corrupt V1–V8)
    try:
        cleaned = remove_lines(img)
        gray_clean = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if len(cleaned.shape) == 3 else cleaned
        _, v9 = cv2.threshold(gray_clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _save("v9", v9)
    except Exception:
        pass  # If line removal fails, just skip V9

    # V10 — Super-upscaled + sharpened + Otsu (for tiny / mixed-size characters)
    try:
        big = _super_upscale(img)
        big_gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if len(big.shape) == 3 else big
        _, v10 = cv2.threshold(big_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _save("v10", v10)
    except Exception:
        pass

    return paths


# ── Diagnostic helpers ───────────────────────────────────────

def save_diagnostic_images(image_path: str, attempt: int, debug_dir: str, log) -> None:
    """Save raw CAPTCHA and all preprocessing variants for manual inspection."""
    import shutil
    try:
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            log(f"[diag] Could not read {image_path} for diagnostics")
            return

        # Save raw
        raw_file = os.path.join(debug_dir, f"attempt_{attempt}_00_RAW.png")
        cv2.imwrite(raw_file, raw_img)
        log(f"[diag] Saved raw image: {raw_file}")

        # Save preprocessed variants
        variant_paths = preprocess_variants(image_path)
        for i, vpath in enumerate(variant_paths, 1):
            dst = os.path.join(debug_dir, f"attempt_{attempt}_{i:02d}_V{i}.png")
            try:
                shutil.copy2(vpath, dst)
            except Exception:
                pass

        # Log image properties
        h, w = raw_img.shape[:2]
        log(f"[diag] Image: {w}x{h}px, {len(variant_paths)} variants saved")

    except Exception as e:
        log(f"[diag] Diagnostic save failed: {e}")
