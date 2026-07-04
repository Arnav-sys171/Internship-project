"""
utils.py — Shared Utilities for CAPTCHA Solver v2

Handles:
  • CAPTCHA image download (page-clip screenshot, element screenshot, fetch fallback)
  • Input typing (single box, multi-box with JS injection)
  • CAPTCHA refresh (button click, JS src change)
  • Image blank detection
"""

from __future__ import annotations

import os
import re
import time
import random

import cv2
import numpy as np


# ── Image blank detection ────────────────────────────────────

def is_blank_image(path: str) -> bool:
    """Check if a saved image is blank (all white / black / uniform)."""
    try:
        img = cv2.imread(path)
        if img is None:
            return True
        return float(np.std(img)) < 5.0
    except Exception:
        return True


# ── CAPTCHA image download ───────────────────────────────────

def download_captcha_image(pg, cap_el, save_path: str, log) -> bool:
    """
    Capture the CAPTCHA image that is currently DISPLAYED on the page.

    CRITICAL: We must NOT re-fetch from the server (e.g. Captcha.ashx)
    because each fetch generates a NEW captcha and updates the session.

    Strategy (in order of preference):
      1. Page-level screenshot with clip — captures exactly what's on screen.
      2. Element screenshot — works for <img>, may be blank for <embed>.
      3. fetch() — LAST RESORT, with a warning that it regenerates.
    """
    # Method 1: Page-level screenshot cropped to element bounds
    try:
        box = cap_el.bounding_box()
        if box and box["width"] > 10 and box["height"] > 10:
            pg.screenshot(path=save_path, clip={
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
            })
            if not is_blank_image(save_path):
                fsize = os.path.getsize(save_path)
                log(f"[download] Page-clip screenshot ({fsize} bytes) → {save_path}")
                return True
            else:
                log("[download] Page-clip screenshot is blank, trying element screenshot")
    except Exception as e:
        log(f"[download] Page-clip screenshot failed: {e}")

    # Method 2: Direct element screenshot
    try:
        cap_el.screenshot(path=save_path)
        if not is_blank_image(save_path):
            fsize = os.path.getsize(save_path)
            log(f"[download] Element screenshot ({fsize} bytes) → {save_path}")
            return True
        else:
            log("[download] Element screenshot is blank")
    except Exception as e:
        log(f"[download] Element screenshot failed: {e}")

    # Method 3 (LAST RESORT): Fetch from server
    tag_name = ""
    try:
        tag_name = pg.evaluate("el => el.tagName.toLowerCase()", cap_el)
    except Exception:
        pass

    if tag_name in ("embed", "object"):
        src = cap_el.get_attribute("src") or cap_el.get_attribute("data") or ""
        if not src:
            log("[download] No src on embed/object element")
            return False

        log(f"[download] ⚠ LAST RESORT: fetching embed src (regenerates captcha!): {src}")
        try:
            abs_url = pg.evaluate(f"""() => {{
                const a = document.createElement('a');
                a.href = {repr(src)};
                return a.href;
            }}""")
            log(f"[download] Absolute URL: {abs_url}")

            b64_data = pg.evaluate("""async (url) => {
                const resp = await fetch(url, { credentials: 'include' });
                if (!resp.ok) return null;
                const blob = await resp.blob();
                return new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result.split(',')[1]);
                    reader.readAsDataURL(blob);
                });
            }""", abs_url)

            if b64_data:
                import base64
                img_bytes = base64.b64decode(b64_data)
                with open(save_path, "wb") as f:
                    f.write(img_bytes)
                log(f"[download] Saved {len(img_bytes)} bytes → {save_path}")
                return True
            else:
                log("[download] fetch returned null")
        except Exception as e:
            log(f"[download] fetch failed: {e}")

        try:
            abs_url = pg.evaluate(f"""() => {{
                const a = document.createElement('a');
                a.href = {repr(src)};
                return a.href;
            }}""")
            response = pg.context.request.get(abs_url)
            if response.ok:
                with open(save_path, "wb") as f:
                    f.write(response.body())
                log(f"[download] API request OK → {save_path}")
                return True
            else:
                log(f"[download] API request failed: {response.status}")
        except Exception as e:
            log(f"[download] API fallback failed: {e}")

    return False


# ── CAPTCHA refresh ──────────────────────────────────────────

def refresh_captcha(pg, log) -> None:
    """Click the refresh/reload button to get a new CAPTCHA image."""
    selectors = [
        "#reloadimg",
        "a[title='Genrate New Image']",
        "a.reload",
        "a[title*='enerate']",
        "a[title*='enew']",
        "a[title*='efresh']",
        "img[title*='efresh']", "img[src*='efresh']",
        "button[title*='efresh']", "[onclick*='captcha']",
        "[onclick*='refresh' i]", "[onclick*='generate' i]",
        "a:has-text('Refresh')", "button:has-text('Refresh')",
        "a:has-text('New Image')", "a:has-text('Generate')",
        "a[href='#'][title]",
    ]
    for sel in selectors:
        try:
            btn = pg.query_selector(sel)
            if btn and btn.is_visible():
                btn.click()
                log(f"  CAPTCHA refreshed via '{sel}'")
                time.sleep(1.5)
                return
        except Exception:
            continue

    # Fallback: Directly set embed src via JS
    try:
        pg.evaluate("""() => {
            const el = document.getElementById('CaptchaIMG');
            if (el) el.src = 'UserControls/Captcha.ashx?t=' + Date.now();
        }""")
        log("  CAPTCHA refreshed via JS src change")
        time.sleep(1.5)
        return
    except Exception:
        pass

    log("  WARNING: Could not refresh CAPTCHA")


# ── Input typing ─────────────────────────────────────────────

def type_captcha(pg, inputs: list, text: str, log) -> bool:
    """
    Type CAPTCHA text into one or multiple inputs.

    Single input  → clear and type whole string.
    Multiple boxes → type one character per box via JS injection.
    """
    if not inputs:
        log("  No inputs to type into")
        return False

    try:
        if len(inputs) == 1:
            inp = inputs[0]
            inp.click()
            time.sleep(0.2)
            pg.keyboard.press("Control+a")
            time.sleep(0.1)
            inp.type(text, delay=random.randint(60, 140))
            log(f"  Typed '{text}' into single input")
        else:
            chars = list(text)
            log(f"  Typing '{text}' across {len(inputs)} boxes (JS method)")
            try:
                for i, inp in enumerate(inputs):
                    if i >= len(chars):
                        break
                    pg.evaluate("""([el, val]) => {
                        el.value = val;
                        el.dispatchEvent(new Event('input',  {bubbles: true}));
                        el.dispatchEvent(new Event('change', {bubbles: true}));
                    }""", [inp, chars[i]])
                log(f"  JS injection complete for {min(len(chars), len(inputs))} boxes")
            except Exception as js_err:
                log(f"  JS injection failed ({js_err}) — falling back to keyboard")
                for i, inp in enumerate(inputs):
                    if i >= len(chars):
                        break
                    try:
                        inp.click()
                        time.sleep(0.15)
                        pg.keyboard.press("Control+a")
                        time.sleep(0.05)
                        pg.keyboard.press(chars[i])
                        time.sleep(0.15)
                    except Exception as e:
                        log(f"  Box {i} failed: {e}")
        return True

    except Exception as exc:
        log(f"  Keyboard type failed ({exc}) — JS fallback")
        try:
            if len(inputs) == 1:
                pg.evaluate("""([el, val]) => {
                    el.value = val;
                    el.dispatchEvent(new Event('input',  {bubbles:true}));
                    el.dispatchEvent(new Event('change', {bubbles:true}));
                }""", [inputs[0], text])
            else:
                for i, inp in enumerate(inputs):
                    if i >= len(text):
                        break
                    pg.evaluate("""([el, val]) => {
                        el.value = val;
                        el.dispatchEvent(new Event('input',  {bubbles:true}));
                        el.dispatchEvent(new Event('change', {bubbles:true}));
                    }""", [inp, text[i]])
            return True
        except Exception as exc2:
            log(f"  JS fallback failed: {exc2}")
            return False
