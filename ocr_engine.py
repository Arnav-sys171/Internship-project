"""
ocr_engine.py — Visual OCR Pipeline (Tier 1)

Runs the ddddocr ensemble (base + beta × 8 variants) and EasyOCR,
then combines all reads through the MSA consensus engine.

Returns (text, confidence) for use by the main orchestrator.
"""

from __future__ import annotations

import io
import os
import re
import sys
import threading
from typing import Optional

from preprocessing import preprocess_variants
from consensus import per_char_case_vote

# ── ddddocr (suppress ads) ──────────────────────────────────

_ocr: Optional[object] = None
_ocr_beta: Optional[object] = None

_saved_out, _saved_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
try:
    import ddddocr as _ddddocr_mod
    _ocr = _ddddocr_mod.DdddOcr(use_gpu=False, show_ad=False)
    _ocr_beta = _ddddocr_mod.DdddOcr(beta=True, use_gpu=False, show_ad=False)
except Exception:
    pass
finally:
    sys.stdout, sys.stderr = _saved_out, _saved_err

print("ddddocr ready" if _ocr else "ddddocr unavailable")


# ── EasyOCR (background boot) ────────────────────────────────

_easy_reader = None
_easy_ready = threading.Event()


def _boot_easyocr():
    global _easy_reader
    try:
        import easyocr
        _easy_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready")
    except ImportError:
        print("EasyOCR unavailable")
    except Exception as exc:
        print(f"EasyOCR boot failed: {exc}")
    finally:
        _easy_ready.set()


threading.Thread(target=_boot_easyocr, daemon=True).start()


# ── ddddocr helpers ──────────────────────────────────────────

def _ddddocr_read(image_path: str, use_beta: bool = False) -> str:
    engine = _ocr_beta if use_beta else _ocr
    if engine is None:
        return ""
    try:
        with open(image_path, "rb") as f:
            return engine.classification(f.read())
    except Exception:
        return ""


def _clean_math(text: str) -> str:
    text = text.replace('f', '+').replace('F', '+').replace('t', '+').replace('T', '+')
    text = text.replace('x', '+').replace('X', '+').replace('H', '+').replace('h', '+')
    text = text.replace('K', '+').replace('k', '+')
    text = text.replace('o', '0').replace('O', '0')
    text = text.replace('l', '1').replace('I', '1')
    text = text.replace('s', '5').replace('S', '5')
    text = text.replace('z', '2').replace('Z', '2')
    return text

def _ddddocr_vote(image_path: str, log, is_math_context: bool = False) -> tuple[list[str], list[str]]:
    """
    Run BOTH ddddocr models on all 8 variants.
    Returns (base_reads, beta_reads).
    """
    all_variants = preprocess_variants(image_path)

    # Build variant index
    variant_indices: dict[int, str] = {}
    for p in all_variants:
        if "_v" in p:
            try:
                idx = int(p.rsplit("_", 1)[-1].replace(".png", "").replace("v", ""))
                variant_indices[idx] = p
            except Exception:
                pass

    base_votes: list[str] = []
    beta_votes: list[str] = []

    # Strip non-printable/Unicode characters (like Chinese) but keep visible ASCII special chars
    _strip_re = re.compile(r"[^\x21-\x7E]")

    for v_idx, path in sorted(variant_indices.items()):
        base_text = _strip_re.sub("", _ddddocr_read(path, use_beta=False))
        beta_text = _strip_re.sub("", _ddddocr_read(path, use_beta=True))

        if is_math_context:
            base_text = _clean_math(base_text)
            beta_text = _clean_math(beta_text)

        if len(base_text) >= 3:
            base_votes.append(base_text)
            log(f"  [dddd:base:v{v_idx}] '{base_text}'")
        else:
            log(f"  [dddd:base:v{v_idx}] too short → '{base_text}'")

        if len(beta_text) >= 3:
            beta_votes.append(beta_text)
            log(f"  [dddd:beta:v{v_idx}] '{beta_text}'")

    return base_votes, beta_votes


# ── EasyOCR helpers ──────────────────────────────────────────

def _easyocr_read(image_path: str, log, wait_s: float = 30.0) -> tuple[str, float]:
    if not _easy_ready.wait(timeout=wait_s):
        log("  [easy] timed out")
        return "", 0.0
    if _easy_reader is None:
        log("  [easy] unavailable")
        return "", 0.0

    try:
        # Test Original, V5 (colour), V7 (morphological), V8 (bilateral), V10 (super-upscaled)
        v5_path = image_path.replace(".png", "_v5.png")
        v7_path = image_path.replace(".png", "_v7.png")
        v8_path = image_path.replace(".png", "_v8.png")
        v10_path = image_path.replace(".png", "_v10.png")

        best_text, best_conf = "", 0.0
        paths_to_test = [image_path]
        if os.path.exists(v5_path):
            paths_to_test.append(v5_path)
        if os.path.exists(v7_path):
            paths_to_test.append(v7_path)
        if os.path.exists(v8_path):
            paths_to_test.append(v8_path)
        if os.path.exists(v10_path):
            paths_to_test.append(v10_path)

        for path in paths_to_test:
            try:
                hits = _easy_reader.readtext(path, detail=1)
                if not hits:
                    continue
                text = "".join(h[1] for h in hits)
                conf = sum(h[2] for h in hits) / len(hits)
                if conf > best_conf:
                    best_conf = conf
                    best_text = text
            except Exception:
                continue

        if best_text:
            log(f"  [easy] '{best_text}'  conf={best_conf:.2f}")
        return best_text, best_conf

    except Exception as exc:
        log(f"  [easy] failed: {exc}")
        return "", 0.0


# ── Public API ───────────────────────────────────────────────

def ensemble_ocr(image_path: str, log, is_math_context: bool = False) -> tuple[str, float]:
    """
    Run the full visual OCR pipeline:
      1. ddddocr base + beta across 8 preprocessed variants
      2. EasyOCR on original + select variants
      3. MSA consensus with case voting

    Returns (text, confidence) where confidence is 0.0–1.0.
    """
    log("[OCR] ── ddddocr ensemble ──")
    base_reads, beta_reads = _ddddocr_vote(image_path, log, is_math_context)

    if is_math_context:
        # If it's a math equation, any read without an operator is a hallucination.
        valid_base = [r for r in base_reads if any(op in r for op in '+-*=')]
        valid_beta = [r for r in beta_reads if any(op in r for op in '+-*=')]
        if valid_base or valid_beta:
            base_reads = valid_base
            beta_reads = valid_beta
            log(f"[OCR] Filtered to {len(base_reads)+len(beta_reads)} reads containing math operators")

    log("[OCR] ── EasyOCR ──")
    easy_raw, easy_conf = _easyocr_read(image_path, log)
    easy = re.sub(r"[^\x21-\x7E]", "", easy_raw)
    log(f"[OCR] easy='{easy}'  easy_conf={easy_conf:.2f}")

    if not base_reads and not easy:
        log("[OCR] both engines returned nothing")
        return "", 0.0

    if not base_reads:
        log(f"[OCR] only EasyOCR → '{easy}'")
        return easy, easy_conf

    result, confidence = per_char_case_vote(base_reads, beta_reads, easy, log, easy_conf=easy_conf)
    return result, confidence
