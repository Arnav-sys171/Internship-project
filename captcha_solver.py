"""
captcha_solver.py — CAPTCHA Solver v2 — Main Orchestrator + Overlay UI

Tiered solving architecture:
  Tier 1: Visual OCR (ddddocr + EasyOCR + MSA consensus + line removal)
  Tier 2: Audio CAPTCHA (faster-whisper, local, zero API cost)

Smart routing:
  1. Check for text-based "fake image" CAPTCHAs (instant solve)
  2. Detect audio button → route to audio solver (most reliable)
  3. Fall back to visual OCR pipeline
  4. If visual confidence is low and audio is available, retry with audio
"""

from __future__ import annotations

import os
import re
import sys
import time
import ctypes
import threading
import queue
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import logging as hf_logging
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

print("Loading OpenAI CLIP model (this may take a moment)...")
try:
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model ready.")
except Exception as e:
    print(f"✗ CLIP initialization failed: {e}")

import tkinter as tk

# Windows DPI fix
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

# ── Imports from v2 modules ──────────────────────────────────

from detection import (
    find_captcha_element,
    find_input_elements,
    find_audio_button,
    is_botdetect,
)
from utils import (
    download_captcha_image,
    refresh_captcha,
    type_captcha,
)
from preprocessing import save_diagnostic_images
from ocr_engine import ensemble_ocr
from audio_solver import solve_audio_captcha, is_available as whisper_available

# ── Constants ────────────────────────────────────────────────

BG        = "#111827"
BTN_GREEN = "#22c55e"
BTN_RED   = "#ef4444"
BTN_BLUE  = "#2563eb"
TEXT      = "#f3f4f6"

CAPTCHA_SCREENSHOT = "captcha_live.png"
GRID_SCREENSHOT = "recaptcha_grid.png"
CAPTCHA_DEBUG_DIR  = "captcha_debug"

os.makedirs(CAPTCHA_DEBUG_DIR, exist_ok=True)

_cmd_queue  = queue.Queue()
_stop_event = threading.Event()


def _flush_queue():
    while not _cmd_queue.empty():
        try:
            _cmd_queue.get_nowait()
        except queue.Empty:
            break


# ── Confidence threshold ─────────────────────────────────────

VISUAL_CONFIDENCE_THRESHOLD = 0.60  # below this, try audio if available


# ─────────────────────────────────────────────────────────────
# HIGH-LEVEL CAPTCHA SOLVER
# ─────────────────────────────────────────────────────────────

def _solve_image_captcha(pg, log, app=None) -> bool:
    max_attempts = 5

    # ── Fast path: text-based "fake image" CAPTCHAs ──
    try:
        for fake_cap in pg.query_selector_all("input[readonly]"):
            if not fake_cap.is_visible():
                continue
            val = fake_cap.input_value().strip()
            if not val or len(val) < 3:
                continue

            style = (fake_cap.get_attribute("style") or "").lower()
            classes = (fake_cap.get_attribute("class") or "").lower()
            name = (fake_cap.get_attribute("name") or "").lower()
            id_ = (fake_cap.get_attribute("id") or "").lower()

            is_fake = False
            if "letter-spacing" in style or "background" in style or "url" in style:
                is_fake = True
            elif "captcha" in name or "captcha" in id_ or "captcha" in classes:
                is_fake = True

            if is_fake:
                log(f"Detected text-based 'fake image' CAPTCHA with value: '{val}'")
                inputs = find_input_elements(pg, fake_cap)
                if inputs:
                    log(f"Found {len(inputs)} input(s), filling '{val}'")
                    if type_captcha(pg, inputs, val, log):
                        log(f"CAPTCHA filled with '{val}' — ready for user to sign in")
                        return True
                break
    except Exception:
        pass

    # ── Find the CAPTCHA element and/or Audio button ──
    cap_el = None
    audio_btn = None
    for attempt in range(10):
        if app and not app.running: return False
        # Silent for first 9 attempts, log diagnostics on the last
        attempt_log = None if attempt < 9 else log
        cap_el = find_captcha_element(pg, log=attempt_log)
        audio_btn = find_audio_button(pg, cap_el, log=attempt_log)
        if cap_el or audio_btn:
            # Re-run once with logging
            cap_el = find_captcha_element(pg, log)
            audio_btn = find_audio_button(pg, cap_el, log)
            break
        time.sleep(0.5)

    if not cap_el and not audio_btn:
        log("No CAPTCHA element or audio button found")
        return False

    is_math_context = False
    try:
        page_text = pg.evaluate("document.body ? document.body.innerText.toLowerCase() : ''")
        if any(kw in page_text for kw in ['math question', 'math problem', 'solve this simple math', 'add the numbers', 'result of math', 'enter the result']):
            is_math_context = True
    except Exception:
        pass

    # ── Detect input boxes (determines expected length) ──
    inputs = find_input_elements(pg, cap_el, audio_btn)
    expected_len = len(inputs) if len(inputs) >= 3 else 0
    if expected_len:
        log(f"Detected {expected_len} input boxes → expecting {expected_len}-char CAPTCHA")

    # ── Tier 2: Try audio CAPTCHA first (most reliable) ──
    if audio_btn and whisper_available():
        log("[router] Audio button found + Whisper ready → trying audio solver")
        try:
            audio_text = solve_audio_captcha(pg, cap_el, audio_btn, log)
            if audio_text:
                if is_math_context:
                    audio_text_upper = audio_text.upper()
                    if any(w in audio_text_upper for w in ["EQUALS", "IS", "TYPE", "ENTER", "WRITE", "ANSWER"]):
                        numbers = re.findall(r'\d+', audio_text)
                        if numbers:
                            audio_text = numbers[-1]
                            log(f"[audio] Math parsed from speech → '{audio_text}'")
                    elif "PLUS" in audio_text_upper or "MINUS" in audio_text_upper:
                        math_expr = audio_text_upper.replace("PLUS", "+").replace("MINUS", "-")
                        math_expr = re.sub(r'[^0-9\+\-\*\/]', '', math_expr)
                        try:
                            result_val = eval(math_expr, {"__builtins__": None}, {})
                            audio_text = str(int(result_val))
                            log(f"[audio] Math evaluated from speech: {math_expr} = {audio_text}")
                        except Exception:
                            pass
                    else:
                        numbers = re.findall(r'\d+', audio_text)
                        if numbers:
                            audio_text = numbers[-1]
                            log(f"[audio] Extracted math number → '{audio_text}'")

                if audio_text and (len(audio_text) >= 3 or is_math_context):
                    # Enforce expected length if known
                    if expected_len and not is_math_context:
                        if len(audio_text) > expected_len:
                            audio_text = audio_text[:expected_len]
                        elif len(audio_text) < expected_len:
                            log(f"[audio] Too short ({len(audio_text)}/{expected_len}) — falling back to visual")
                            audio_text = None

                if audio_text:
                    # Re-find inputs
                    if not inputs:
                        inputs = find_input_elements(pg, cap_el, audio_btn)
                    if inputs:
                        # BotDetect → uppercase
                        if cap_el and is_botdetect(cap_el):
                            audio_text = audio_text.upper()
                            log("  [detect] BotDetect engine → forcing uppercase")

                        log(f"Found {len(inputs)} input(s), filling '{audio_text}'")
                        if type_captcha(pg, inputs, audio_text, log):
                            log(f"CAPTCHA filled with '{audio_text}' (audio) — ready for user to sign in")
                            return True
        except Exception as e:
            log(f"[audio] Audio solver failed: {e} — falling back to visual")
    elif audio_btn:
        log("[router] Audio button found but Whisper not ready — using visual OCR")
    else:
        log("[router] No audio button found — using visual OCR")

    # ── Tier 1: Visual OCR Pipeline ──
    best_text = ""
    best_conf = 0.0

    for attempt in range(1, max_attempts + 1):
        try:
            if app and not app.running:
                log("Solver stopped by user, exiting image CAPTCHA loop")
                return False

            # Re-find element after refresh
            if attempt > 1:
                cap_el = find_captcha_element(pg, log)
                if not cap_el:
                    log("  Element gone after refresh")
                    break
                time.sleep(0.5)

            # Download CAPTCHA image
            ok = download_captcha_image(pg, cap_el, CAPTCHA_SCREENSHOT, log)
            if not ok:
                log(f"  Failed to download CAPTCHA image (attempt {attempt})")
                if attempt < max_attempts:
                    refresh_captcha(pg, log)
                    time.sleep(1.0)
                continue

            log(f"CAPTCHA captured (attempt {attempt})")

            # Diagnostic save
            save_diagnostic_images(CAPTCHA_SCREENSHOT, attempt, CAPTCHA_DEBUG_DIR, log)

            # Run OCR ensemble
            text, confidence = ensemble_ocr(CAPTCHA_SCREENSHOT, log, is_math_context)
            text = re.sub(r"[^\x21-\x7E]", "", text)

            # BotDetect → uppercase
            if is_botdetect(cap_el):
                text = text.upper()
                log("  [detect] BotDetect engine identified → forcing uppercase")

            if is_math_context or (any(op in text for op in '+-*') and '=' in text):
                is_math_context = True

            if is_math_context:
                # Math cleanup: convert letters that OCR misread
                text = text.replace('f', '+').replace('F', '+').replace('t', '+').replace('T', '+')
                text = text.replace('x', '*').replace('X', '*').replace('H', '+').replace('h', '+')
                text = text.replace('K', '+').replace('k', '+')
                
                # Convert 'o'/'O' to '0' unconditionally in math context
                # Convert 's'/'S' to '5'
                # Convert 'z'/'Z' to '2'
                text = text.replace('o', '0').replace('O', '0')
                text = text.replace('s', '5').replace('S', '5')
                text = text.replace('z', '2').replace('Z', '2')
                
                # Remove = and ? for evaluation
                clean_for_eval = text.replace('=', '').replace('?', '')
                
                math_expr = re.sub(r'[^0-9\+\-\*\/]', '', clean_for_eval)
                if math_expr and any(op in math_expr for op in '+-*/'):
                    try:
                        result_val = eval(math_expr, {"__builtins__": None}, {})
                        text = str(int(result_val))
                        log(f"Math evaluated: {math_expr} = {text}")
                    except Exception as e:
                        log(f"Math eval failed for '{math_expr}': {e}")
                        text = ""
                elif math_expr and 3 <= len(math_expr) <= 4 and ('4' in math_expr or '5' in math_expr):
                    # OCR missed the '+' or '-' and read it as a number or just joined them
                    for i in range(1, len(math_expr) - 1):
                        for op in ['+', '-']:
                            test_eq = math_expr[:i] + op + math_expr[i:]
                            # Also check if a '4' was meant to be '+'
                            if math_expr[i] == '4':
                                test_eq2 = math_expr[:i] + '+' + math_expr[i+1:]
                                try:
                                    res = eval(test_eq2, {"__builtins__": None}, {})
                                    if 0 <= res <= 200:
                                        log(f"Recovered missing '+' from '{math_expr}': {test_eq2}")
                                        text = str(int(res))
                                        break
                                except: pass
                            
                            try:
                                res = eval(test_eq, {"__builtins__": None}, {})
                                if 0 <= res <= 200:
                                    log(f"Recovered missing '{op}' from '{math_expr}': {test_eq}")
                                    text = str(int(res))
                                    break
                            except Exception:
                                pass
                        if text != clean_for_eval and text != math_expr: break
                else:
                    log(f"Math OCR failed to find an operator in '{text}', forcing retry")
                    text = ""  # Force retry

            log(f"Raw OCR text: '{text}'  confidence={confidence:.2f}")

            # Enforce expected length
            if expected_len and len(text) != expected_len:
                if len(text) > expected_len:
                    text = text[:expected_len]
                    log(f"  Truncated to {expected_len} chars → '{text}'")
                elif len(text) < expected_len:
                    log(f"  Too short ({len(text)}/{expected_len} chars) — retrying")
                    if attempt < max_attempts:
                        refresh_captcha(pg, log)
                        time.sleep(1.0)
                    continue

            if len(text) >= 3 or is_math_context:
                # If confidence is low and audio is available, try audio
                if confidence < VISUAL_CONFIDENCE_THRESHOLD and audio_btn and whisper_available():
                    log(f"  [router] Low confidence ({confidence:.2f}) — trying audio fallback")
                    try:
                        # Refresh to get a fresh captcha with audio
                        refresh_captcha(pg, log)
                        time.sleep(1.5)
                        cap_el = find_captcha_element(pg, log)
                        if cap_el:
                            audio_btn = find_audio_button(pg, cap_el, log)
                            if audio_btn:
                                audio_text = solve_audio_captcha(pg, cap_el, audio_btn, log)
                                if audio_text and (len(audio_text) >= 3 or is_math_context):
                                    text = audio_text
                                    log(f"  [router] Audio fallback succeeded: '{text}'")
                    except Exception as e:
                        log(f"  [router] Audio fallback failed: {e}")

                best_text = text
                best_conf = confidence
                break

            log(f"  Too short ({len(text)} chars) — retrying …")
        except Exception as exc:
            import traceback
            log(f"  OCR error: {exc}\n{traceback.format_exc()}")

        if attempt < max_attempts:
            log(f"  Refreshing CAPTCHA (attempt {attempt}/{max_attempts})…")
            refresh_captcha(pg, log)
            time.sleep(1.0)

    if not best_text:
        log("All OCR attempts exhausted")
        return False

    # Re-find inputs in case page state changed
    if not inputs:
        inputs = find_input_elements(pg, cap_el)
    if not inputs:
        log("Input field(s) not found")
        return False

    # Trim text to match input count for multi-box CAPTCHAs
    if len(inputs) > 1 and len(best_text) > len(inputs):
        best_text = best_text[:len(inputs)]
        log(f"  Trimmed to match {len(inputs)} boxes → '{best_text}'")

    log(f"Found {len(inputs)} input(s), filling '{best_text}'")
    if not type_captcha(pg, inputs, best_text, log):
        return False

    log(f"CAPTCHA filled with '{best_text}' — ready for user to sign in")
    return True



def _human_move_and_click(pg, x: float, y: float):
    steps = random.randint(18, 30)
    viewport_size = pg.viewport_size
    if viewport_size:
        cx = viewport_size["width"] / 2
        cy = viewport_size["height"] / 2
    else:
        cx = pg.evaluate("() => window.innerWidth") / 2
        cy = pg.evaluate("() => window.innerHeight") / 2
    
    mx = (cx + x) / 2 + random.uniform(-40, 40)
    my = (cy + y) / 2 + random.uniform(-40, 40)
    for i in range(steps):
        t  = i / steps
        bx = (1-t)**2 * cx + 2*(1-t)*t * mx + t**2 * x
        by = (1-t)**2 * cy + 2*(1-t)*t * my + t**2 * y
        pg.mouse.move(bx, by)
        time.sleep(random.uniform(0.008, 0.022))
    time.sleep(random.uniform(0.05, 0.15))
    pg.mouse.click(x, y)

# ─────────────────────────────────────────────────────────────
# Image CAPTCHA helpers  (called from PW thread)
# ─────────────────────────────────────────────────────────────
CAPTCHA_KEYWORDS = {
    "captcha": 5, "verify": 3, "verif": 3,
    "security": 2, "code": 2, "validation": 3,
    "challenge": 3, "servlet": 3, "generate": 2,
    "random": 1, "human": 2, "bot": 2,
    "kaptcha": 5, "securimage": 5, "jcaptcha": 5,
    "vcode": 4, "imgcode": 4, "checkcode": 4,
}
NOISE_WORDS = {"logo","banner","icon","arrow","header","footer",
               "avatar","profile","photo"}
MIN_SCORE = 3



def _solve_dom_math(pg, log) -> bool:
    try:
        for frame in pg.frames:
            try:
                math_infos = frame.evaluate(r"""() => {
                    if (!document || !document.body) return [];
                    
                    const resultsSet = [];
                    const allNodes = document.querySelectorAll('div, p, span, h1, h2, h3, h4, h5, h6, label, b, strong, i, em');
                    
                    for (let node of allNodes) {
                        if (node.offsetParent === null && node.tagName.toLowerCase() !== 'label') continue;
                        
                        const text = ((node.innerText || node.textContent) || "").trim();
                        if (text.length > 60 || text.length < 3) continue;
                        const match1 = text.match(/^[\s\S]*?(\d+)\s*([\+\-\*\/\u2212\u2013\u2014\u00D7\u00F7xX])\s*(\d+)\s*[=\?][=\?\s\)]*(?:this field is required\.?)?$/i);
                        const match2 = text.match(/^(?:solve|captcha|what is|calculate|math)[\s\S]*?(\d+)\s*([\+\-\*\/\u2212\u2013\u2014\u00D7\u00F7xX])\s*(\d+)/i);
                        // match3: standalone short math expressions like "9 + 8" without trailing = or ?
                        const match3 = text.length < 15 ? text.match(/^\s*(\d+)\s*([\+\-\*\/\u2212\u2013\u2014\u00D7\u00F7xX])\s*(\d+)\s*$/) : null;
                        const match = match1 || match2 || match3;
                        
                        if (match) {
                            resultsSet.push({node: node, match: match});
                        }
                    }
                    
                    // Keep only the deepest matching nodes
                    const deepestResults = resultsSet.filter(item => {
                        return !resultsSet.some(other => item !== other && item.node.contains(other.node));
                    });
                    
                    const results = [];
                    let index = 0;
                    for (let item of deepestResults) {
                        const node = item.node;
                        const match = item.match;
                        let container = node; // start at the deepest matching node itself
                        let input = null;
                        
                        for (let i=0; i<8; i++) {
                            if (!container) break;
                            const inputs = container.querySelectorAll('input:not([type="hidden"]):not([type="radio"]):not([type="checkbox"])');
                            for (let el of inputs) {
                                if (!el.hasAttribute('data-math-target')) {
                                    input = el;
                                    break;
                                }
                            }
                            if (input) break;
                            container = container.parentElement;
                        }
                        
                        if (!input) {
                            const all_inputs = document.querySelectorAll('input:not([type="hidden"]):not([type="radio"]):not([type="checkbox"])');
                            for (let el of all_inputs) {
                                if (!el.hasAttribute('data-math-target') && el.offsetParent !== null) {
                                    input = el; break;
                                }
                            }
                        }
                        
                        if (input) {
                            const target_id = "math-target-" + index;
                            input.setAttribute('data-math-target', target_id);
                            
                            let verifyBtnId = null;
                            const buttons = container.querySelectorAll('button, input[type="button"], input[type="submit"]');
                            for (let btn of buttons) {
                                let btnText = (btn.innerText || btn.value || '').toUpperCase();
                                if (btnText.includes('VERIFY') || btnText.includes('CONFIRM') || btnText.includes('VALIDATE')) {
                                    verifyBtnId = "math-verify-" + index;
                                    btn.setAttribute('data-math-verify', verifyBtnId);
                                    break;
                                }
                            }

                            let rawOp = match[2];
                            let normOp = rawOp;
                            if (rawOp === 'x' || rawOp === 'X' || rawOp === '\u00D7') normOp = '*';
                            if (rawOp === '\u2212' || rawOp === '\u2013' || rawOp === '\u2014') normOp = '-';
                            if (rawOp === '\u00F7') normOp = '/';

                            results.push({
                                target_id: target_id,
                                verify_id: verifyBtnId,
                                expr: match[0],
                                num1: parseInt(match[1]),
                                op: normOp,
                                num2: parseInt(match[3])
                            });
                            index++;
                        }
                    }
                    return results;
                }""")
                
                if math_infos:
                    solved_any = False
                    for math_info in math_infos:
                        expr = math_info['expr']
                        num1 = math_info['num1']
                        op = math_info['op']
                        num2 = math_info['num2']
                        target_id = math_info['target_id']
                        
                        result = 0
                        if op == '+': result = num1 + num2
                        elif op == '-': result = num1 - num2
                        elif op == '*': result = num1 * num2
                        
                        log(f"Found DOM math CAPTCHA: {expr} = {result}")
                        try:
                            loc = frame.locator(f"[data-math-target='{target_id}']")
                            loc.wait_for(state="visible", timeout=2000)
                            loc.fill(str(result))
                            
                            verify_id = math_info.get('verify_id')
                            if verify_id:
                                vloc = frame.locator(f"[data-math-verify='{verify_id}']")
                                vloc.wait_for(state="visible", timeout=1000)
                                vloc.click(force=True, timeout=2000)
                                log("Clicked verify button")
                            
                            solved_any = True
                        except Exception as e:
                            log(f"Failed to fill DOM math CAPTCHA: {e}")
                    
                    if solved_any:
                        # Check for slider (KooKoo style)
                        try:
                            slider = frame.locator(".slider-thumb").first
                            if slider.count() > 0 and slider.is_visible():
                                log("Found verification slider, dragging to complete...")
                                box = slider.bounding_box()
                                if box:
                                    # Ensure we hover first to trigger any JS event listeners
                                    slider.hover(timeout=2000)
                                    pg.mouse.down()
                                    # Drag horizontally to the right with more steps to simulate real drag
                                    pg.mouse.move(box["x"] + 350, box["y"] + box["height"] / 2, steps=25)
                                    pg.mouse.up()
                                    time.sleep(1)
                                    log("Slider dragged successfully.")
                        except Exception as e:
                            log(f"Error handling slider: {e}")
                            
                        return True
            except Exception as e:
                # ignore cross-origin errors
                pass
    except Exception as e:
        log(f"DOM math solver error: {e}")
    return False



def _detect_tiles(image_path: str, prompt: str, grid_size: int) -> list:
    try:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        tile_w = w / grid_size
        tile_h = h / grid_size
        
        tiles_images = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * tile_w
                top = row * tile_h
                right = (col + 1) * tile_w
                bottom = (row + 1) * tile_h
                cropped = image.crop((left, top, right, bottom))
                tiles_images.append(cropped)
                
        PROMPT_MAP = {
            "bicycles": "bicycle",
            "buses": "bus",
            "cars": "car",
            "crosswalks": "crosswalk",
            "fire hydrants": "fire hydrant",
            "motorcycles": "motorcycle",
            "stairs": "stairs",
            "traffic lights": "traffic light",
            "bridges": "bridge",
            "chimneys": "chimney",
            "palm trees": "palm tree",
            "tractors": "tractor"
        }
        target_item = PROMPT_MAP.get(prompt.lower(), prompt.lower())
        target_label = f"a photo of a {target_item}"
        if target_item == "stairs":
             target_label = "a photo of stairs"
             
        # Explicitly listing all other objects allows CLIP to accurately distribute 
        # probability instead of failing on negation words like 'without'
        ALL_ITEMS = [
            "a bicycle", "a bus", "a car", "a crosswalk", "a fire hydrant", 
            "a motorcycle", "stairs", "a traffic light", "a bridge", 
            "a chimney", "a palm tree", "a tractor", "a truck", "a person", "a van"
        ]
        
        labels = [target_label]
        for item in ALL_ITEMS:
            if item.replace("a ", "") != target_item and item != target_item:
                if item == "stairs":
                    labels.append("a photo of stairs")
                else:
                    labels.append(f"a photo of {item}")
        
        labels.append("a photo of a plain empty street")
        labels.append("a photo of a plain background, sky, or wall")
        
        inputs = _clip_processor(text=labels, images=tiles_images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        tiles = set()
        for idx, prob in enumerate(probs):
            target_prob = prob[0].item()
            # If the target is present with at least 16% confidence across 17 choices,
            # it's a very strong signal. 16% prevents false positives on blurry background tiles.
            if target_prob > 0.16:
                tiles.add(idx)
                
        return sorted(tiles)
    except Exception as e:
        print(f"CLIP detection error: {e}")
        return []

# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# Frame helpers
# ─────────────────────────────────────────────────────────────

def _type_into_input(pg, inp, text: str, log):
    try:
        inp.click(timeout=2000)
        time.sleep(0.1)
        pg.keyboard.press("Control+a")
        time.sleep(0.1)
        pg.keyboard.up("Control")
        inp.type(text, delay=random.randint(60, 140), timeout=5000)
    except Exception as exc:
        log(f"Keyboard type failed ({exc}) — using JS fallback")
        try:
            inp.evaluate(
                """(el, val) => {
                    el.value = val;
                    el.dispatchEvent(new Event('input',  {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }""",
                text,
            )
        except Exception as exc2:
            log(f"JS fallback also failed: {exc2}")
            return False
    return True

_RECAPTCHA_IFRAME_SELECTORS = [
    "iframe[src*='google.com/recaptcha']",
    "iframe[src*='recaptcha.net']",
    "iframe[src*='recaptcha']",
    "iframe[title='reCAPTCHA']",
    "iframe[title*='recaptcha' i]",
]

def _trigger_recaptcha_render(pg, log):
    try:
        for sel in ["input[type='text']", "input[type='email']",
                    "input[name*='user' i]", "input[name*='login' i]",
                    "input[name*='reg' i]", "input:not([type='hidden'])"]:
            # Quick check without waiting
            try:
                inps = pg.query_selector_all(sel)
                inp = inps[0] if inps else None
            except Exception:
                inp = None
            if inp and inp.is_visible():
                try:
                    inp.click(force=True, timeout=1000)
                except Exception:
                    inp.evaluate("el => el.click()")
                log(f"Clicked form field ({sel}) to trigger reCAPTCHA render")
                time.sleep(1.5)
                break
    except Exception as exc:
        log(f"Field click failed: {exc}")

    try:
        container = pg.query_selector(".g-recaptcha, [data-sitekey]")
        if container:
            container.scroll_into_view_if_needed()
            log("Scrolled .g-recaptcha into view")
            time.sleep(1.0)
    except Exception as exc:
        log(f"Scroll failed: {exc}")

    try:
        rendered = pg.evaluate("""() => {
            if (typeof grecaptcha === 'undefined') return 'no_grecaptcha';
            const el = document.querySelector('.g-recaptcha, [data-sitekey]');
            if (!el) return 'no_container';
            if (el.querySelector('iframe')) return 'already_rendered';
            const key = el.getAttribute('data-sitekey');
            if (key) {
                try { grecaptcha.render(el, {sitekey: key}); return 'rendered'; }
                catch(e) { return 'render_error:' + e.message; }
            }
            return 'no_sitekey';
        }""")
        log(f"grecaptcha render probe: {rendered}")
        if rendered in ("rendered", "already_rendered"):
            time.sleep(1.5)
    except Exception as exc:
        log(f"JS render probe failed: {exc}")

def _dump_frames(pg, log):
    try:
        urls = [f.url for f in pg.frames if f.url]
        log(f"[v0] Current frames: {urls}")
    except Exception:
        pass

def _wait_for_recaptcha_iframe(pg, log, timeout: float = 15.0) -> bool:
    _trigger_recaptcha_render(pg, log)
    deadline = time.time() + timeout
    attempt  = 0
    log(f"[v0] Waiting for reCAPTCHA iframe (timeout: {timeout}s)")
    while time.time() < deadline:
        attempt += 1
        for sel in _RECAPTCHA_IFRAME_SELECTORS:
            try:
                pg.wait_for_selector(sel, timeout=1_500, state="attached")
                log(f"[v0] reCAPTCHA iframe found via '{sel}' (attempt {attempt})")
                time.sleep(1.2)
                return True
            except Exception as e:
                pass
        if attempt % 4 == 1:
            log(f"[v0] Iframe not found yet (attempt {attempt})...")
            _dump_frames(pg, log)
        time.sleep(0.5)

    log(f"[v0] reCAPTCHA iframe not found after {timeout}s — dumping frames for diagnosis:")
    _dump_frames(pg, log)
    return False



def _get_anchor_frame(pg, timeout: float = 8.0, log=None):
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            for f in pg.frames:
                url = f.url or ""
                if ("recaptcha" in url or "recaptcha.net" in url) and "anchor" in url:
                    try:
                        if f.query_selector("#recaptcha-anchor"):
                            if log:
                                log(f"[v0] Anchor frame with checkbox found on attempt {attempt}: {url[:80]}")
                            return f
                    except Exception:
                        pass
        except Exception as e:
            if log:
                log(f"[v0] Error checking frames: {e}")
        
        if log and attempt % 4 == 1:
            try:
                all_urls = [f.url[:60] for f in pg.frames if f.url and "about:blank" not in f.url]
                if all_urls:
                    log(f"[v0] Waiting for anchor frame, current frames: {all_urls}")
            except Exception:
                pass
        time.sleep(0.3)
    if log:
        log("[v0] Anchor frame not found after timeout")
    return None



def _get_challenge_frame(pg, timeout: float = 8.0, log=None):
    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            for f in pg.frames:
                url = f.url or ""
                if ("recaptcha" in url or "recaptcha.net" in url) and "bframe" in url:
                    if log:
                        log(f"[v0] Challenge frame found on attempt {attempt}")
                    return f
        except Exception as e:
            if log:
                log(f"[v0] Error checking challenge frames: {e}")
        time.sleep(0.3)
    if log:
        log("[v0] Challenge frame not found")
    return None

# ─────────────────────────────────────────────────────────────
# Solver functions
# ─────────────────────────────────────────────────────────────


def _solve_grid(pg, log, app=None) -> bool:
    for round_num in range(1, 10):
        if app and not app.running: return False
        cf = _get_challenge_frame(pg, timeout=5.0, log=log)
        if not cf:
            log("Grid challenge gone — passed")
            return True
        
        # Try multiple selectors for prompt text
        prompt = None
        prompt_selectors = [
            ".rc-imageselect-desc-wrapper strong",
            ".rc-imageselect-desc strong",
            ".rc-imageselect-desc-no-canonical",
            ".rc-imageselect-desc",
            ".rc-imageselect-desc-wrapper",
            "div.rc-imageselect-prompt strong",
            "div.rc-imageselect-prompt",
            "[role='dialog'] strong",
            "[role='dialog'] .rc-imageselect-desc-wrapper",
        ]
        
        for selector in prompt_selectors:
            try:
                prompt_el = cf.query_selector(selector)
                if prompt_el:
                    raw_text = prompt_el.inner_text()
                    if raw_text and raw_text.strip():
                        prompt = raw_text.strip()
                        log(f"[v0] Prompt found via selector '{selector}': '{prompt}'")
                        break
            except Exception as e:
                log(f"[v0] Selector '{selector}' failed: {e}")
                continue
        
        # Fallback: extract all visible text and look for prompt-like content
        if not prompt:
            try:
                all_text = cf.evaluate("""() => {
                    const desc = document.querySelector('.rc-imageselect-desc-wrapper') ||
                                 document.querySelector('.rc-imageselect-desc');
                    if (desc) {
                        const strong = desc.querySelector('strong');
                        if (strong) return strong.innerText;
                        return desc.innerText;
                    }
                    return '';
                }""")
                if all_text and all_text.strip():
                    prompt = all_text.strip()
                    log(f"[v0] Prompt found via JS fallback: '{prompt}'")
            except Exception as e:
                log(f"[v0] JS fallback failed: {e}")
        
        if not prompt:
            log("Could not read challenge prompt after all attempts")
            return False
        
        log(f"[Grid round {round_num}] Target: '{prompt}'")
        
        grid_el = cf.query_selector(
            ".rc-imageselect-challenge, table.rc-imageselect-table"
        )
        if not grid_el:
            log("Grid element not found")
            return False
        grid_el.screenshot(path=GRID_SCREENSHOT)
        img = cv2.imread(GRID_SCREENSHOT)
        
        all_tiles = cf.query_selector_all(
            "td.rc-imageselect-tile, .rc-imageselect-target td"
        )
        grid_size = 4 if len(all_tiles) == 16 else 3
        log(f"Grid: {grid_size}x{grid_size} (Detected from {len(all_tiles)} tiles)")
        
        tile_indices = _detect_tiles(GRID_SCREENSHOT, prompt, grid_size)
        log(f"Tiles to click: {tile_indices}")
        
        if tile_indices:
            for idx in tile_indices:
                if idx >= len(all_tiles): continue
                try:
                    all_tiles[idx].click()
                    log(f"  Clicked tile {idx}")
                    time.sleep(random.uniform(0.5, 1.1))
                except Exception as e:
                    log(f"  Tile {idx} failed: {e}")
            time.sleep(random.uniform(0.8, 1.4))
            
            # Re-check for prompt/status after clicking
            desc_el = cf.query_selector(".rc-imageselect-desc-wrapper")
            full_desc = desc_el.inner_text().lower() if desc_el else prompt.lower()
            is_dynamic = "none left" in full_desc
            if is_dynamic:
                log("Dynamic challenge — waiting for new tiles to load...")
                time.sleep(2.5)
                continue
                
        verify_btn = cf.query_selector("#recaptcha-verify-button")
        if verify_btn:
            try:
                verify_btn.click(force=True)
                log("Verify clicked")
            except Exception as e:
                log(f"Failed to click verify: {e}")
                verify_btn.evaluate("el => el.click()")
        time.sleep(2.0)
        if not _get_challenge_frame(pg, timeout=2.0, log=log):
            log("Grid solved!")
            return True
        af = _get_anchor_frame(pg, timeout=2.0, log=log)
        if af and af.query_selector(".recaptcha-checkbox-checked"):
            log("reCAPTCHA passed!")
            return True
        err = cf.query_selector(".rc-imageselect-incorrect-response")
        if err and err.is_visible():
            log("Wrong tiles — retrying...")
            time.sleep(1.0)
    log("Grid solver exhausted")
    return False



def _check_custom_robot_button(pg, log) -> bool:
    """Find and click custom fake 'I'm not a robot' buttons that reveal hidden captchas."""
    try:
        clicked = pg.evaluate("""() => {
            let btns = Array.from(document.querySelectorAll('button, a, div, span, label')).filter(el => {
                if (!el.innerText || el._robotClicked) return false;
                let text = el.innerText.toUpperCase().replace(/[^A-Z]/g, '');
                return text.includes('IMNOTAROBOT') || text.includes('IAMNOTAROBOT');
            });
            
            // Keep only the deepest elements to avoid clicking ancestor wrappers
            btns = btns.filter(el => !btns.some(child => el !== child && el.contains(child)));
            btns = btns.filter(b => b.offsetParent !== null);
            
            window._robotClickedSignatures = window._robotClickedSignatures || new Set();
            
            let toClick = btns.filter(b => {
                let text = b.innerText.toUpperCase().replace(/[^A-Z]/g, '');
                let sig = 'ROBOT_BTN|' + text;
                if (window._robotClickedSignatures.has(sig)) return false;
                return true;
            });

            if (toClick.length > 0) {
                toClick.forEach(b => {
                    let text = b.innerText.toUpperCase().replace(/[^A-Z]/g, '');
                    let sig = 'ROBOT_BTN|' + text;
                    window._robotClickedSignatures.add(sig);
                    b._robotClicked = true;
                    b.click();
                });
                return true;
            }
            return false;
        }""")
        if clicked:
            log("Clicked custom 'I'm not a robot' button")
            return True
    except Exception:
        pass
    return False

def _solve_hcaptcha(pg, log, app=None) -> bool:
    if app and not app.running: return False
    log("[hCaptcha] Starting solver")
    
    af = None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        for f in pg.frames:
            url = f.url or ""
            if "hcaptcha" in url:
                try:
                    if f.query_selector("#checkbox"):
                        af = f
                        break
                except Exception:
                    pass
        if af: break
        time.sleep(0.5)

    if not af:
        log("[hCaptcha] Anchor frame not found")
        return False
        
    log("[hCaptcha] Checkbox frame found")
    
    try:
        checkbox = af.wait_for_selector("#checkbox", state="attached", timeout=5000)
    except Exception as e:
        log(f"[hCaptcha] Checkbox element not found: {e}")
        return False
        
    try:
        log("[hCaptcha] Clicking checkbox...")
        checkbox.click(force=True)
    except Exception as e:
        log(f"[hCaptcha] Click failed: {e}")
        try:
            checkbox.evaluate("el => el.click()")
        except Exception:
            pass

    log("[hCaptcha] Checkbox clicked. Waiting for resolution...")
    
    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            checked = af.evaluate("() => document.querySelector('#checkbox')?.getAttribute('aria-checked') === 'true'")
            if checked:
                log("[hCaptcha] Checkbox passed automatically! (1-click)")
                return True
        except Exception: pass
        
        try:
            chal_f = None
            for f in pg.frames:
                url = f.url or ""
                if "hcaptcha" in url and "challenge" in url:
                    chal_f = f
                    break
            
            if chal_f:
                chal = chal_f.query_selector(".challenge-container, .challenge")
                if chal and chal.is_visible():
                    log("[hCaptcha] A visual challenge appeared.")
                    
                    # Check if text challenge is already active
                    inp = chal_f.query_selector("input[type='text'], input[type='number'], input:not([type='hidden'])")
                    if not inp:
                        # Try to click the accessibility button
                        acc_btn = chal_f.query_selector(".accessibility-button, [title*='Access' i], [title*='Text' i], .help-button")
                        if acc_btn and acc_btn.is_visible():
                            log("[hCaptcha] Found Accessibility button. Clicking...")
                            try:
                                acc_btn.click(force=True, timeout=2000)
                            except Exception as e:
                                log(f"[hCaptcha] Click failed, trying evaluate: {e}")
                                try:
                                    acc_btn.evaluate("el => el.click()")
                                except Exception: pass
                            time.sleep(1)
                            
                            # Click the menu item that appears
                            acc_menu_item = chal_f.query_selector("text='Accessibility Challenge'")
                            if acc_menu_item and acc_menu_item.is_visible():
                                log("[hCaptcha] Clicking 'Accessibility Challenge' menu item...")
                                try:
                                    acc_menu_item.click(force=True, timeout=2000)
                                except Exception:
                                    try: acc_menu_item.evaluate("el => el.click()")
                                    except: pass
                                time.sleep(2)
                        
                    # Try to solve the text challenge
                    try:
                        solved_one = _solve_hcaptcha_text(pg, log, chal_f)
                        if solved_one:
                            # It solved one question. The captcha might have multiple questions.
                            # Extend the deadline and loop again to check if the checkbox passed or if there's another question.
                            deadline = time.time() + 8.0
                            continue
                        else:
                            log("[hCaptcha] Failed to solve text challenge in this iteration.")
                    except Exception as e:
                        if "Target page, context or browser has been closed" in str(e):
                            log("[hCaptcha] Frame closed during solve (likely solved automatically or user closed).")
                        else:
                            log(f"[hCaptcha] Solve attempt error: {e}")
                    
                    # We broke out of the solve attempt, but don't return False immediately.
                    # The frame might have closed because the CAPTCHA was solved successfully.
                    # We will loop back up, and the 'checked' attribute check might pass.
                    time.sleep(1.0)
                    continue
                    
        except Exception as e:
            log(f"[hCaptcha] Challenge wait error: {e}")
        time.sleep(1.0)
        
    log("[hCaptcha] Timed out waiting for checkbox resolution")
    return False

def _solve_hcaptcha_text(pg, log, f) -> bool:
    try:
        from hcaptcha_text_solver import solve_hcaptcha_text
    except ImportError:
        log("[hCaptcha] hcaptcha_text_solver.py not found.")
        return False

    log("[hCaptcha] Attempting text logic solver...")
    try:
        # Extract prompt: target the challenge container directly
        chal_container = f.query_selector(".challenge-container, body")
        if chal_container:
            # First dump all textual elements directly using JavaScript to ensure we capture the dynamic math spans
            extracted_text = chal_container.evaluate("el => el.innerText")
            all_text = extracted_text.split('\n')
            prompt_texts = [t.strip() for t in all_text if len(t.strip()) > 5]
        else:
            prompt_texts = []
            
        full_prompt = " ".join(prompt_texts)
        log(f"[hCaptcha] Extracted prompt: {full_prompt}")
        
        answer = solve_hcaptcha_text(full_prompt)
        if not answer:
            log("[hCaptcha] Heuristic engine could not solve this prompt.")
            return False
            
        log(f"[hCaptcha] Heuristic engine solved! Answer: {answer}")
        
        # Input answer
        inp = f.query_selector("input[type='text'], input[type='number'], input:not([type='hidden'])")
        if not inp:
            log("[hCaptcha] Could not find input box.")
            return False
            
        inp.fill(answer)
        try:
            inp.evaluate(f"el => el.value = '{answer}'")
        except: pass
        time.sleep(0.5)
        
        # Submit
        submit_btn = f.query_selector(".button-submit")
        if submit_btn and submit_btn.is_visible():
            submit_btn.click(force=True)
        else:
            inp.press("Enter")
        time.sleep(2.0)
        
        # Check if verified (we must check if the error is ACTUALLY visible, as it exists in DOM by default with opacity: 0)
        err = f.query_selector(".display-error, .error-text, .challenge-error")
        if err:
            try:
                opacity = err.evaluate("el => window.getComputedStyle(el).opacity")
                display = err.evaluate("el => window.getComputedStyle(el).display")
                if float(opacity) > 0 and display != 'none':
                    log(f"[hCaptcha] Answer rejected: {err.inner_text()}")
                    return False
            except:
                pass
            
        log("[hCaptcha] Submitted answer.")
        return True
    except Exception as e:
        log(f"[hCaptcha] Text solver error: {e}")
        return False

def _solve_recaptcha(pg, log, app=None) -> bool:
    if app and not app.running: return False
    try:
        widget_present = pg.evaluate("""() =>
            !!(document.querySelector('.g-recaptcha')          ||
               document.querySelector('[data-sitekey]')        ||
               document.querySelector('iframe[src*=\"recaptcha\"]') ||
               document.querySelector('iframe[title=\"reCAPTCHA\"]'))
        """)
        if not widget_present:
            log("[v0] No reCAPTCHA widget in DOM — skipping reCAPTCHA solver")
            return False
        else:
            log("[v0] reCAPTCHA widget detected, proceeding with solve")
    except Exception as e:
        log(f"[v0] Widget detection error: {e}")
        pass

    if not _wait_for_recaptcha_iframe(pg, log):
        return False

    af = _get_anchor_frame(pg, log=log, timeout=12.0)
    if not af:
        log("reCAPTCHA anchor frame not found after iframe appeared")
        _dump_frames(pg, log)
        return False

    # Wait for checkbox to be ready (max 5 seconds)
    try:
        checkbox = af.wait_for_selector("#recaptcha-anchor", state="attached", timeout=5000)
        log("Checkbox found inside anchor frame")
    except Exception as e:
        log(f"Checkbox not found inside anchor frame after retries: {e}")
        return False

    log("Clicking checkbox...")

    frame_el = None
    for sel in _RECAPTCHA_IFRAME_SELECTORS:
        try:
            frame_el = pg.query_selector(sel + "[src*='anchor']")
            if frame_el:
                log(f"[v0] Found anchor frame via: {sel}")
                break
        except Exception:
            pass
    if not frame_el:
        try:
            frame_el = pg.query_selector("iframe[title='reCAPTCHA']")
            if frame_el:
                log("[v0] Found reCAPTCHA iframe via title selector")
        except Exception:
            pass

    try:
        # Ensure checkbox is enabled
        is_enabled = af.evaluate("""() => {
            const el = document.querySelector('#recaptcha-anchor');
            return el && !el.disabled && el.offsetParent !== null;
        }""")
        
        if not is_enabled:
            log("[v0] Checkbox appears disabled or hidden, waiting and retrying...")
            time.sleep(2)
            is_enabled = af.evaluate("""() => {
                const el = document.querySelector('#recaptcha-anchor');
                return el && !el.disabled && el.offsetParent !== null;
            }""")
        
        log(f"[v0] Checkbox enabled state: {is_enabled}")
        
        box = checkbox.bounding_box()
        log(f"[v0] Checkbox box in frame coords: {box}")
        
        # Scroll checkbox into view
        af.evaluate("document.querySelector('#recaptcha-anchor').scrollIntoView(true);")
        time.sleep(0.5)
        
        # Get updated box after scroll
        box = checkbox.bounding_box()
        log(f"[v0] Checkbox box after scroll: {box}")
        
        # If we have the iframe element, get its position on the main page
        if frame_el:
            fb = frame_el.bounding_box()
            cx = box["x"] + box["width"] / 2 + random.uniform(-3, 3)
            cy = box["y"] + box["height"] / 2 + random.uniform(-3, 3)
            log(f"[v0] Frame position: {fb}, will click at: ({cx}, {cy})")
        else:
            # Fallback: assume checkbox is roughly at its reported coordinates
            cx = box["x"] + box["width"] / 2 + random.uniform(-3, 3)
            cy = box["y"] + box["height"] / 2 + random.uniform(-3, 3)
            log(f"[v0] No frame found, will click at: ({cx}, {cy})")
        
        # Try JS click first (more reliable)
        try:
            af.evaluate("""() => {
                const el = document.querySelector('#recaptcha-anchor');
                if (el) {
                    el.click();
                    el.dispatchEvent(new MouseEvent('click', {bubbles: true}));
                }
            }""")
            log("[v0] Tried JS click on checkbox")
            time.sleep(1.5)
        except Exception as e:
            log(f"[v0] JS click failed: {e}, trying human click...")
            _human_move_and_click(pg, cx, cy)
        
        time.sleep(2.5)
    except Exception as e:
        log(f"[v0] Error during checkbox click: {e}")
        import traceback
        log(traceback.format_exc())

    af = _get_anchor_frame(pg, log=log, timeout=3.0)
    if af and af.query_selector(".recaptcha-checkbox-checked"):
        log("reCAPTCHA passed on checkbox!")
        return True

    log("Trying audio challenge solver first...")
    if app and not app.running:
        return False
    cf = _get_challenge_frame(pg, timeout=5.0, log=log)
    if cf:
        audio_btn = cf.query_selector("#recaptcha-audio-button")
        if audio_btn and audio_btn.is_visible():
            log("Audio button found, switching to audio challenge...")
            import tempfile, os
            intercepted_audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio_intercepted.mp3")
            try:
                if os.path.exists(intercepted_audio_path):
                    try: os.remove(intercepted_audio_path)
                    except: pass
                    
                try:
                    # Just click it, don't expect_response because it might not auto-fetch payload
                    audio_btn.evaluate("el => el.click()")
                    log("Clicked audio button via JS")
                except Exception as e:
                    log(f"Failed to click audio button: {e}")
                    try:
                        ab = audio_btn.bounding_box()
                        if ab:
                            cx = ab["x"] + ab["width"] / 2
                            cy = ab["y"] + ab["height"] / 2
                            _human_move_and_click(pg, cx, cy)
                    except: pass
                time.sleep(1.0)
                
                play_btn = None
                err = None
                for _ in range(10):
                    cf = _get_challenge_frame(pg, timeout=1.0)
                    if not cf: continue
                    err = cf.query_selector(".rc-doscaptcha-header-text")
                    if err and err.is_visible(): break
                    play_btn = cf.query_selector(".rc-audiochallenge-play-button")
                    if play_btn: break
                    time.sleep(0.5)
                
                if err and err.is_visible() and "automated queries" in err.inner_text().lower():
                    log("Audio challenge blocked by anti-bot.")
                else:
                    import audio_solver
                    last_audio_url = None
                    for audio_round in range(6):
                        if app and not app.running: return False
                        
                        cf = _get_challenge_frame(pg, timeout=2.0)
                        if not cf: break
                        play_btn = cf.query_selector(".rc-audiochallenge-play-button")
                        if not play_btn: break
                        
                        text = audio_solver.solve_audio_captcha(pg, cf, play_btn, log, is_recaptcha=True, challenge_frame=cf)
                        if app and not app.running: return False
                        
                        if text:
                            # Strip punctuation that Whisper might add to phrases
                            import re
                            text = re.sub(r'[.,!?;:\'"()]', '', text).strip()
                            
                            # Type answer directly into the challenge frame via JS
                            # This avoids cross-frame keyboard issues with pg.keyboard
                            try:
                                cf.evaluate("""(answer) => {
                                    const inp = document.getElementById('audio-response');
                                    if (inp) {
                                        inp.focus();
                                        inp.value = answer;
                                        inp.dispatchEvent(new Event('input', {bubbles: true}));
                                        inp.dispatchEvent(new Event('change', {bubbles: true}));
                                    }
                                }""", text)
                                
                                # Verify the value was actually set
                                actual_val = cf.evaluate("""() => {
                                    const inp = document.getElementById('audio-response');
                                    return inp ? inp.value : null;
                                }""")
                                log(f"[audio] Typed answer: '{actual_val}' (expected: '{text}')")
                                
                                if actual_val != text:
                                    log("[audio] WARNING: Input value mismatch!")
                                    # Fallback: try element handle typing
                                    inp = cf.query_selector("#audio-response")
                                    if inp:
                                        _type_into_input(pg, inp, text, log)
                                
                            except Exception as e:
                                log(f"[audio] JS typing failed: {e}, trying element handle")
                                inp = cf.query_selector("#audio-response")
                                if inp:
                                    _type_into_input(pg, inp, text, log)
                            
                            time.sleep(0.3)
                            
                            # Click verify via JS within the frame
                            try:
                                cf.evaluate("""() => {
                                    const btn = document.getElementById('recaptcha-verify-button');
                                    if (btn) btn.click();
                                }""")
                            except Exception:
                                verify_btn = cf.query_selector("#recaptcha-verify-button")
                                if verify_btn:
                                    verify_btn.click(force=True)
                            
                            time.sleep(5.0)
                            
                            af2 = _get_anchor_frame(pg, timeout=3.0, log=log)
                            if af2 and af2.query_selector(".recaptcha-checkbox-checked"):
                                log("reCAPTCHA passed on audio!")
                                return True
                            
                            # Re-acquire challenge frame (may have been replaced)
                            cf = _get_challenge_frame(pg, timeout=3.0, log=log)
                            if not cf:
                                log("Challenge frame gone after verify — checking if solved")
                                af3 = _get_anchor_frame(pg, timeout=2.0, log=log)
                                if af3 and af3.query_selector(".recaptcha-checkbox-checked"):
                                    log("reCAPTCHA passed on audio!")
                                    return True
                                break
                            
                            # Check for anti-bot block
                            err = cf.query_selector(".rc-doscaptcha-header-text")
                            if err and err.is_visible() and "automated queries" in err.inner_text().lower():
                                log("Audio challenge blocked by anti-bot.")
                                break
                                
                            err_msg = cf.query_selector(".rc-audiochallenge-error-message")
                            if err_msg and err_msg.is_visible():
                                log(f"Audio challenge feedback: {err_msg.inner_text()}")
                            
                            
                            # Reload the challenge to get a fresh audio
                            reload_btn = cf.query_selector("#recaptcha-reload-button")
                            if reload_btn:
                                log("Reloading to get a fresh audio challenge...")
                                # Clear only the intercepted URL array (don't touch the audio element!)
                                try:
                                    cf.evaluate("""() => {
                                        const win = window;
                                        if (win._interceptedAudioUrls) win._interceptedAudioUrls = [];
                                    }""")
                                except Exception:
                                    pass
                                try:
                                    import tempfile, os
                                    intercepted_audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio_intercepted.mp3")
                                    if os.path.exists(intercepted_audio_path):
                                        try: os.remove(intercepted_audio_path)
                                        except: pass
                                        
                                    with pg.expect_response(
                                        lambda resp: "payload" in resp.url and "p=" in resp.url,
                                        timeout=5000
                                    ) as resp_info:
                                        reload_btn.click(force=True)
                                        
                                    body = resp_info.value.body()
                                    with open(intercepted_audio_path, "wb") as f:
                                        f.write(body)
                                    log(f"Intercepted reload audio payload ({len(body)} bytes)")
                                except Exception as e:
                                    log(f"Failed to intercept reload payload: {e}")
                                    try: reload_btn.click(force=True)
                                    except: pass
                                # Wait for new challenge to load: input gets cleared and play button reappears
                                for _wait in range(10):
                                    time.sleep(0.5)
                                    try:
                                        input_val = cf.evaluate("""() => {
                                            const inp = document.getElementById('audio-response');
                                            return inp ? inp.value : '';
                                        }""")
                                        if not input_val:
                                            log("[audio] New audio challenge loaded")
                                            break
                                    except Exception:
                                        break
                        else:
                            break
            except Exception as e:
                log(f"Error during audio challenge: {e}")
                
    if app and not app.running:
        return False

    # Switch back to image challenge before trying grid solver
    try:
        cf = _get_challenge_frame(pg, timeout=2.0, log=log)
        if cf:
            img_btn = cf.query_selector("#recaptcha-image-button")
            if img_btn and img_btn.is_visible():
                img_btn.click(force=True)
                time.sleep(1.0)
    except Exception:
        pass

    log("Audio solver exhausted or blocked, trying CLIP grid solver fallback...")
    if _solve_grid(pg, log, app):
        return True

    return False

# ─────────────────────────────────────────────────────────────
# Playwright worker thread
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# PLAYWRIGHT WORKER THREAD
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# DRISSIONPAGE TURNSTILE SOLVER (Phase 1 of Hybrid Engine)
# ─────────────────────────────────────────────────────────────

def _solve_turnstile_dp(dp_page, log, app=None) -> bool:
    """Solve Cloudflare Turnstile using DrissionPage's native browser control."""
    try:
        log("Checking for Turnstile widget...")
        cf_iframe = None
        
        def search_container(container):
            for selector in [
                "css:.cf-turnstile",
                "css:#cf-turnstile",
                "css:iframe[src*='challenges.cloudflare.com']",
                "css:iframe[src*='turnstile']",
                "css:iframe[title*='cloudflare' i]",
                "css:iframe[title*='challenge' i]",
                "css:div[style*='display: grid']"
            ]:
                try:
                    el = container.ele(selector, timeout=0.5)
                    if el:
                        return el
                except:
                    pass
            return None

        # 1. Search main page
        cf_iframe = search_container(dp_page)
        
        # 2. Search inside all child frames recursively if not found
        if not cf_iframe:
            def search_all_frames(parent_container):
                try:
                    for iframe in parent_container.get_frames():
                        # First check the frame itself
                        el = search_container(iframe)
                        if el:
                            return el
                        # Then recursively check its children
                        child_el = search_all_frames(iframe)
                        if child_el:
                            return child_el
                except Exception as e:
                    pass
                return None
                
            cf_iframe = search_all_frames(dp_page)
            
        if not cf_iframe:
            log("No Turnstile widget found on page!")
            return False
            
        log("Found Cloudflare Turnstile widget, clicking offset...")
        
        # Click 30 pixels from the left edge of the target element.
        try:
            offset_y = int(cf_iframe.rect.size[1] / 2)
            cf_iframe.click.at(30, offset_y)
            log("Click executed successfully.")
        except Exception as e:
            log(f"Offset click failed, falling back to basic click: {e}")
            try:
                cf_iframe.click()
                log("Basic click executed.")
            except Exception as e2:
                log(f"Basic click also failed: {e2}")
                return False
            
        # We clicked it successfully! Return True to let the outer wait loop handle the rest.
        return True

    except Exception as e:
        log(f"Turnstile solve error: {e}")
    return False


def _wait_for_turnstile_clear(dp_page, log, timeout=20) -> bool:
    """Wait for Turnstile to fully clear (page redirect or token generation)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # Check 1: Token generated (form will auto-submit)
            try:
                token_el = dp_page.ele("css:input[name='cf-turnstile-response']", timeout=0.5)
                if token_el:
                    val = token_el.attr("value")
                    if val and len(val) > 10:
                        log("Turnstile token detected! Forcing form submit...")
                        try:
                            dp_page.run_js('let f=document.getElementById("challenge-form"); if(f) f.submit();')
                        except Exception:
                            pass
                        time.sleep(2)
                        return True
            except Exception:
                pass
                
            # Check 2: Is the Turnstile iframe gone?
            cf_iframe = None
            for selector in [
                "css:iframe[src*='challenges.cloudflare.com']",
                "css:iframe[title*='Cloudflare']",
                "css:iframe[title*='challenge']"
            ]:
                cf_iframe = dp_page.ele(selector, timeout=0.2)
                if cf_iframe:
                    break
                    
            if not cf_iframe:
                wrapper = dp_page.ele("#turnstile-wrapper", timeout=0.2) or dp_page.ele("#challenge-stage", timeout=0.2)
                if wrapper:
                    cf_iframe = wrapper.ele("tag:iframe", timeout=0.2)
                    
            if not cf_iframe:
                # The widget is gone. Wait for the page to navigate away from the challenge page.
                # If we hand off to Playwright too quickly, the title will still be 'Just a moment...' 
                # and Playwright will think the challenge returned.
                log("Turnstile widget disappeared — waiting for page redirect to stabilize...")
                start_wait = time.time()
                while time.time() - start_wait < 5:
                    try:
                        title = dp_page.title.lower() if dp_page.title else ""
                        if "just a moment" not in title and "security verification" not in title and "cloudflare" not in title:
                            break
                    except Exception:
                        pass
                    time.sleep(0.5)
                    
                log("Verification complete!")
                return True
                
        except Exception:
            pass
        time.sleep(1)
    return False


# ─────────────────────────────────────────────────────────────
# HYBRID ENGINE: DrissionPage (Turnstile) + Playwright (OCR)
# ─────────────────────────────────────────────────────────────

def _playwright_worker(app) -> None:
    """Hybrid engine:
    Phase 1 — DrissionPage launches Chrome and bypasses Cloudflare Turnstile.
    Phase 2 — Playwright connects to the same browser via CDP for OCR/reCAPTCHA."""
    
    def log(msg):
        app.log(msg)

    dp_page = None
    pw_browser = None
    pw_context = None
    pg = None
    _pw_instance = None
    _dp_port = None

    try:
        import os
        import socket
        import tempfile
        import uuid
        from playwright.sync_api import sync_playwright
        
        profile_dir = os.path.join(tempfile.gettempdir(), f"cs2_solver_profile_{uuid.uuid4().hex[:8]}")
        
        # Wait for the first goto command to decide the engine
        current_url = None
        while app.running and not _stop_event.is_set():
            try:
                cmd, payload = _cmd_queue.get(timeout=0.5)
                if cmd == "goto":
                    current_url = payload
                    break
                elif cmd == "stop":
                    return
            except queue.Empty:
                pass
                
        if not current_url or not app.running or _stop_event.is_set():
            return
            
        # User requested specific sites to only open using Playwright
        if "janaadhaar.rajasthan.gov.in" in current_url or "forest.rajasthan.gov.in" in current_url:
            USE_DRISSIONPAGE = False
        else:
            # Default to the great hybrid model
            USE_DRISSIONPAGE = True

        SKIP_PLAYWRIGHT = False
        if "cedarparktexas.gov" in current_url or "flaglerclerk.gov" in current_url:
            SKIP_PLAYWRIGHT = True

        if USE_DRISSIONPAGE:
            from DrissionPage import ChromiumPage, ChromiumOptions
            # Pick a free port for the CDP debugging connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                _dp_port = s.getsockname()[1]
            
            log("Launching browser via DrissionPage (stealth engine) …")
            co = ChromiumOptions()
            co.set_local_port(_dp_port)
            co.set_user_data_path(profile_dir)
            co.set_argument("--start-maximized")
            co.set_argument("--disable-blink-features=AutomationControlled")
            co.set_argument("--disable-dev-shm-usage")
            co.set_argument("--no-first-run")
            co.set_argument("--no-default-browser-check")
            
            dp_page = ChromiumPage(addr_or_opts=co)
            log("DrissionPage browser ready")
        else:
            log("Launching browser via Playwright (stealth) …")
            from playwright_stealth import Stealth
            _pw_instance = sync_playwright().start()
            pw_context = _pw_instance.chromium.launch_persistent_context(
                user_data_dir=profile_dir,
                headless=False,
                args=[
                    "--start-maximized",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-first-run",
                    "--no-default-browser-check"
                ],
                ignore_default_args=["--enable-automation"]
            )
            pg = pw_context.pages[0] if pw_context.pages else pw_context.new_page()
            Stealth().apply_stealth_sync(pg)
            log("Playwright browser ready")

        # ── PHASE 1: DrissionPage handles navigation + Turnstile ──
        last_attempt = time.time()
        stop_flag = False
        turnstile_cleared = not USE_DRISSIONPAGE  # Skip Turnstile logic if not using DrissionPage
        pw_connected = not USE_DRISSIONPAGE       # Already connected if pure Playwright
        
        def safe_dp_get(page, url, timeout=15):
            import threading
            def _get():
                try: page.get(url, timeout=timeout)
                except Exception: pass
            t = threading.Thread(target=_get)
            t.start()
            t.join(timeout)
            if t.is_alive():
                log(f"Navigation to {url} hung for {timeout}s, forcing stop_loading()...")
                try: page.stop_loading()
                except Exception: pass

        # Manually process the first URL
        log(f"Opening {current_url}")
        if USE_DRISSIONPAGE:
            turnstile_cleared = False
            pw_connected = False
            if pw_browser:
                try: pw_browser.close()
                except Exception: pass
            pw_browser = None
            pg = None
            pw_context = None
            if _pw_instance:
                try: _pw_instance.stop()
                except Exception: pass
                _pw_instance = None
            
            safe_dp_get(dp_page, current_url, timeout=15)
            last_attempt = time.time() + 2
        else:
            try: pg.goto(current_url, timeout=60000)
            except Exception as e: log(f"Navigation timeout/error: {e}")
            last_attempt = time.time()

        dp_html_errors = 0
        while not stop_flag and app.running and not _stop_event.is_set():
            try:
                cmd, payload = _cmd_queue.get_nowait()
            except queue.Empty:
                cmd, payload = None, None

            if cmd == "stop":
                stop_flag = True
                continue

            if cmd == "goto":
                current_url = payload
                if USE_DRISSIONPAGE:
                    turnstile_cleared = False
                    pw_connected = False
                    # Disconnect Playwright if it was connected from a previous URL
                    if pg:
                        try: pw_browser.close()
                        except Exception: pass
                        pg = None
                        pw_browser = None
                        pw_context = None
                    if _pw_instance:
                        try: _pw_instance.stop()
                        except Exception: pass
                        _pw_instance = None
                    
                    log(f"Navigating to {payload}")
                    safe_dp_get(dp_page, payload, timeout=15)
                    log("Page loaded")
                    last_attempt = time.time()
                else:
                    try:
                        log(f"Navigating to {payload}")
                        pg.goto(payload, timeout=60000)
                        log("Page loaded")
                        last_attempt = time.time()
                    except Exception as exc:
                        log(f"Navigation timeout/error: {exc}")

            if time.time() - last_attempt < 0.2:
                time.sleep(0.1)
                continue

            try:
                # ── Turnstile detection and solving (DrissionPage phase) ──
                if not turnstile_cleared:
                    try:
                        page_html = dp_page.html.lower() if dp_page.html else ""
                        page_title = dp_page.title.lower() if dp_page.title else ""
                        dp_html_errors = 0
                    except Exception as e:
                        if dp_html_errors == 0:
                            log(f"DrissionPage HTML read error: {e}. Retrying...")
                        dp_html_errors += 1
                        if dp_html_errors > 10:
                            log("Too many HTML read errors, skipping to Playwright...")
                            turnstile_cleared = True
                        time.sleep(0.5)
                        continue

                    has_cf_element = bool(dp_page.ele("css:#turnstile-wrapper, #challenge-stage, iframe[src*='challenges.cloudflare.com'], iframe[title*='Cloudflare']", timeout=1))
                    
                    is_turnstile = has_cf_element or "cf-turnstile" in page_html
                    is_challenge_page = "cf-browser-verification" in page_html or "just a moment" in page_title or (has_cf_element and "security verification" in page_title)

                    if is_challenge_page:
                        # ── CLOUDFLARE INTERSTITIAL CHALLENGE PAGE ──
                        # This is the full-page "Just a moment..." blocker.
                        # We must solve Turnstile and wait for the page to redirect.
                        already_has_token = False
                        try:
                            token_el = dp_page.ele("css:input[name='cf-turnstile-response']", timeout=1)
                            if token_el:
                                val = token_el.attr("value")
                                if val and len(val) > 10:
                                    already_has_token = True
                                    log("Turnstile token already present, submitting form...")
                                    try:
                                        dp_page.run_js('let f=document.getElementById("challenge-form"); if(f) f.submit();')
                                    except Exception:
                                        pass
                                    time.sleep(2)
                        except Exception:
                            pass
                        
                        if not already_has_token:
                            # Check if currently verifying
                            try:
                                verifying = dp_page.ele("css:.loading-verifying", timeout=0.5)
                                if verifying and verifying.states.is_displayed:
                                    log("Turnstile is verifying, waiting...")
                                    time.sleep(2)
                                    last_attempt = time.time()
                                    continue
                            except Exception:
                                pass

                            log("Cloudflare challenge page detected — solving via DrissionPage …")
                            last_attempt = time.time()
                            ok = _solve_turnstile_dp(dp_page, log, app)
                            log(f"Turnstile solve result: {ok}")
                            if ok:
                                if _wait_for_turnstile_clear(dp_page, log, timeout=15):
                                    turnstile_cleared = True
                                    log("✓ Turnstile bypassed! Switching to Playwright for CAPTCHA solving …")
                                else:
                                    log("Turnstile click registered but page didn't redirect. Retrying...")
                                    last_attempt = time.time() + 3
                            else:
                                last_attempt = time.time() + 3
                        else:
                            if _wait_for_turnstile_clear(dp_page, log, timeout=10):
                                turnstile_cleared = True
                                log("✓ Turnstile cleared via existing token!")
                            last_attempt = time.time() + 3
                    elif is_turnstile:
                        # ── EMBEDDED TURNSTILE WIDGET ON A NORMAL PAGE ──
                        # The page itself is the destination (e.g., a contact form).
                        # The Turnstile widget is an inline form element.
                        log("Embedded Turnstile widget detected — attempting solve …")
                        last_attempt = time.time()
                        ok = _solve_turnstile_dp(dp_page, log, app)
                        
                        # Embedded widgets usually don't redirect the page, they just check the box
                        if ok:
                            log("✓ Embedded Turnstile widget clicked/cleared!")
                            turnstile_cleared = True
                        else:
                            log("Embedded Turnstile widget not solved (or self-cleared). Proceeding anyway...")
                            turnstile_cleared = True
                    else:
                        # No Turnstile on this page, proceed directly
                        turnstile_cleared = True
                        log("No Turnstile detected — proceeding to Playwright phase")

                # ── PHASE 2: Connect Playwright for OCR/reCAPTCHA ──
                if SKIP_PLAYWRIGHT:
                    if turnstile_cleared:
                        log("Website doesn't require Playwright. Stopping here.")
                        time.sleep(5)
                        break
                elif turnstile_cleared and not pw_connected:
                    try:
                        from playwright.sync_api import sync_playwright
                        if not _pw_instance:
                            _pw_instance = sync_playwright().start()
                        pw_browser = _pw_instance.chromium.connect_over_cdp(f"http://127.0.0.1:{_dp_port}")
                        pw_context = pw_browser.contexts[0]
                        # Find the correct Playwright page that matches DrissionPage's URL
                        pg = None
                        for p in pw_context.pages:
                            if p.url != "about:blank" and not p.url.startswith("chrome-extension"):
                                pg = p
                                break
                        if not pg:
                            pg = pw_context.pages[0] if pw_context.pages else pw_context.new_page()

                        # Reload the page if we need to discover cross-origin iframes
                        # (reCAPTCHA, MTCaptcha, hCaptcha, etc.)
                        # Skip for single-use image CAPTCHAs that 404 on second request
                        dp_html = (dp_page.html or "").lower()
                        needs_reload = (
                            "google.com/recaptcha" in dp_html
                            or "mtcaptcha" in dp_html
                            or "mtcap" in dp_html
                            or "hcaptcha" in dp_html
                            or "imperva" in dp_html
                            or "incapsula" in dp_html
                        )
                        if needs_reload:
                            try:
                                pg.reload(wait_until="load", timeout=30_000)
                            except Exception as reload_err:
                                log(f"Playwright connected (reload warning: {reload_err})")
                        
                        log("Playwright connected to browser via CDP — full OCR engine active")
                        pw_connected = True
                        last_attempt = time.time()
                    except Exception as e:
                        log(f"Playwright CDP connection error: {e}")
                        last_attempt = time.time() + 5
                        continue

                # ── PHASE 2 continued: Standard CAPTCHA solving via Playwright ──
                if pw_connected and pg:
                    try:
                        if pg.is_closed():
                            log("Page closed unexpectedly")
                            break

                        try:
                            content = pg.content().lower()
                        except Exception as e:
                            if "navigating" in str(e).lower():
                                time.sleep(0.5)
                                continue
                            raise

                        try:
                            title = pg.title().lower()
                        except Exception:
                            title = ""
                            
                        has_cf_element = False
                        try:
                            has_cf_element = pg.locator("#turnstile-wrapper, #challenge-stage, iframe[src*='challenges.cloudflare.com'], iframe[title*='Cloudflare']").count() > 0
                        except Exception:
                            pass
                            
                        is_turnstile_again = (
                            has_cf_element
                            or "cf-browser-verification" in content 
                            or "just a moment" in title 
                            or "security verification" in title
                            or "cf-turnstile" in content
                        )
                        
                        if is_turnstile_again and USE_DRISSIONPAGE:
                            log("Turnstile reappeared (or arrived late) — switching back to DrissionPage")
                            turnstile_cleared = False
                            pw_connected = False
                            try:
                                pw_browser.close()
                            except Exception:
                                pass
                            
                            # Give DrissionPage a moment to re-assert control
                            time.sleep(2)
                            
                            pg = None
                            pw_browser = None
                            pw_context = None
                            if _pw_instance:
                                try:
                                    _pw_instance.stop()
                                except Exception:
                                    pass
                                _pw_instance = None
                            last_attempt = 0  # Force immediate DrissionPage check
                            continue

                        is_recaptcha = False
                        is_hcaptcha = False
                        
                        for f in pg.frames:
                            furl = (f.url or "").lower()
                            if "recaptcha" in furl: is_recaptcha = True
                            if "hcaptcha" in furl: is_hcaptcha = True
                                
                        has_captcha = "captcha" in content or "geetest" in content

                        already_solved = False
                        if has_captcha or is_recaptcha or is_hcaptcha:
                            try:
                                already_solved = pg.evaluate("window.__captchaSolved === true")
                            except Exception:
                                pass

                        if not already_solved:
                            if _check_custom_robot_button(pg, log):
                                time.sleep(1.0)
                                last_attempt = time.time()
                                continue
                            
                            if _solve_dom_math(pg, log):
                                last_attempt = time.time() + 8
                                try: pg.evaluate("window.__captchaSolved = true")
                                except: pass
                            elif is_hcaptcha:
                                log("[Attempt 1/10] hCaptcha detected — Playwright mode")
                                last_attempt = time.time()
                                ok = _solve_hcaptcha(pg, log, app)
                                log(f"hCaptcha solve result: {ok}")
                                if ok:
                                    log("✓ CAPTCHA filled — waiting for user to sign in")
                                    last_attempt = time.time() + 8
                                    try: pg.evaluate("window.__captchaSolved = true")
                                    except: pass
                                else:
                                    last_attempt = time.time() + 3
                            elif has_captcha:
                                log("Image CAPTCHA detected — solving …")
                                last_attempt = time.time()
                                ok = _solve_image_captcha(pg, log, app)
                                log(f"Solve {'succeeded' if ok else 'failed'}")
                                if ok:
                                    try:
                                        pg.evaluate("window.__captchaSolved = true")
                                    except Exception:
                                        pass
                                    log("✓ CAPTCHA filled — waiting for user to sign in")
                                    last_attempt = time.time() + 8
                                else:
                                    if is_recaptcha:
                                        if USE_DRISSIONPAGE:
                                            log("[Attempt 1/10] reCAPTCHA detected (fallback) — disconnecting Playwright for stealth")
                                            last_attempt = time.time()
                                            pw_connected = False
                                            try: pw_browser.close()
                                            except: pass
                                            pg = None
                                            
                                            from recaptcha_dp import solve_recaptcha_drission
                                            ok2 = solve_recaptcha_drission(dp_page, log, app)
                                        else:
                                            log("[Attempt 1/10] reCAPTCHA detected (fallback) — Playwright mode")
                                            last_attempt = time.time()
                                            ok2 = _solve_recaptcha(pg, log, app)
                                            
                                        log(f"reCAPTCHA solve result: {ok2}")
                                        if ok2:
                                            log("✓ CAPTCHA filled — waiting for user to sign in")
                                            last_attempt = time.time() + 8
                                        else:
                                            last_attempt = time.time() + 3
                                    else:
                                        last_attempt = time.time() + 3
                            elif is_recaptcha:
                                if USE_DRISSIONPAGE:
                                    log("[Attempt 1/10] reCAPTCHA detected — disconnecting Playwright for stealth")
                                    last_attempt = time.time()
                                    pw_connected = False
                                    try: pw_browser.close()
                                    except: pass
                                    pg = None
                                    
                                    from recaptcha_dp import solve_recaptcha_drission
                                    ok = solve_recaptcha_drission(dp_page, log, app)
                                else:
                                    log("[Attempt 1/10] reCAPTCHA detected — Playwright mode")
                                    last_attempt = time.time()
                                    ok = _solve_recaptcha(pg, log, app)
                                    
                                log(f"reCAPTCHA solve result: {ok}")
                                if ok:
                                    log("✓ CAPTCHA filled — waiting for user to sign in")
                                    last_attempt = time.time() + 8
                                else:
                                    last_attempt = time.time() + 3
                            else:
                                last_attempt = time.time()
                        else:
                            last_attempt = time.time()
                    except Exception as exc:
                        import traceback
                        msg = str(exc).lower()
                        if "closed" in msg:
                            log("Browser closed — exiting")
                            break
                        log(f"Solver error: {exc}\n{traceback.format_exc()}")
                        last_attempt = time.time()

            except Exception as exc:
                import traceback
                log(f"Engine error: {exc}\n{traceback.format_exc()}")
                last_attempt = time.time() + 2

            time.sleep(0.5)

    except Exception as outer:
        import traceback
        app.log(f"CRITICAL: {outer}\n{traceback.format_exc()}")
    finally:
        # Clean up Playwright
        for obj in (pg, pw_context, pw_browser):
            try:
                if obj:
                    obj.close()
            except Exception:
                pass
        if _pw_instance:
            try:
                _pw_instance.stop()
            except Exception:
                pass
        # Clean up DrissionPage
        if dp_page:
            try:
                dp_page.quit()
            except Exception:
                pass
        app.log("Browser closed")


# ─────────────────────────────────────────────────────────────
# OVERLAY UI
# ─────────────────────────────────────────────────────────────

class Overlay(tk.Tk):
    def __init__(self):
        super().__init__()
        self.running = False
        self._pw_thread = None
        self._open_after_id = None

        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.configure(bg=BG)
        self.geometry("380x340+1450+750")

        tk.Label(
            self, text="CAPTCHA SOLVER ",
            bg=BG, fg=TEXT, font=("Consolas", 10, "bold"),
        ).pack(pady=8)

        self.url_entry = tk.Entry(
            self, bg="#1f2937", fg=TEXT,
            insertbackground=TEXT, relief="flat",
            font=("Consolas", 9),
        )
        self.url_entry.pack(fill="x", padx=15, pady=4)
        self.url_entry.insert(
            0,
            "https://ip.webtel.in/testing/(S(l4aejnkulxy3zu3aefeedfdi))"
            "/OnlineCallSheet/login.aspx?COMPANYID=2&serverIP=1",
        )

        self.btn = tk.Button(
            self, text="START", bg=BTN_GREEN, fg="white",
            font=("Consolas", 15, "bold"), relief="flat",
            command=self.toggle,
        )
        self.btn.pack(fill="x", padx=15, pady=8)

        tk.Button(
            self, text="OPEN WEBSITE", bg=BTN_BLUE, fg="white",
            font=("Consolas", 10, "bold"), relief="flat",
            command=self.open_website,
        ).pack(fill="x", padx=15, pady=4)

        self.logbox = tk.Text(
            self, height=12, bg="#1f2937", fg=TEXT,
            relief="flat", font=("Consolas", 8),
        )
        self.logbox.pack(fill="both", expand=True, padx=10, pady=6)

        self.bind("<ButtonPress-1>", self._drag_start)
        self.bind("<B1-Motion>", self._drag_move)
        self.bind("<Escape>", lambda _: self.quit_app())

    def log(self, msg: str) -> None:
        self.logbox.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.logbox.see("end")

    def open_website(self) -> None:
        url = self.url_entry.get().strip()
        if not url:
            return
        if not url.startswith("http"):
            url = "https://" + url
        self.log(f"Opening {url}")
        _cmd_queue.put(("goto", url))

    def toggle(self) -> None:
        self.stop() if self.running else self.start()

    def start(self) -> None:
        _stop_event.set()
        self.running = False
        if self._pw_thread and self._pw_thread.is_alive():
            self.log("Waiting for previous session …")
            self._pw_thread.join(timeout=10)
        _stop_event.clear()
        _flush_queue()
        if self._open_after_id is not None:
            try:
                self.after_cancel(self._open_after_id)
            except Exception:
                pass
            self._open_after_id = None
        self.running = True
        self.btn.config(text="STOP", bg=BTN_RED)
        self.log("Starting …")
        self._pw_thread = threading.Thread(
            target=_playwright_worker, args=(self,), daemon=True,
        )
        self._pw_thread.start()
        self._open_after_id = self.after(3000, self.open_website)

    def stop(self) -> None:
        if self._open_after_id is not None:
            try:
                self.after_cancel(self._open_after_id)
            except Exception:
                pass
            self._open_after_id = None
        self.running = False
        _stop_event.set()
        _cmd_queue.put(("stop", None))
        self.btn.config(text="START", bg=BTN_GREEN)
        self.log("Stopping …")
        if self._pw_thread and self._pw_thread.is_alive():
            self._pw_thread.join(timeout=6)

    def quit_app(self) -> None:
        self.stop()
        self.after(500, self.destroy)

    def _drag_start(self, event) -> None:
        self._mx, self._my = event.x, event.y

    def _drag_move(self, event) -> None:
        self.geometry(
            f"+{self.winfo_x() + event.x - self._mx}"
            f"+{self.winfo_y() + event.y - self._my}"
        )


if __name__ == "__main__":
    Overlay().mainloop()
