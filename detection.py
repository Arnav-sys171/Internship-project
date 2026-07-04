"""
detection.py — CAPTCHA Element Detection

Detects:
  • CAPTCHA image elements (img, embed, object, canvas)
  • Input fields (single or multi-box)
  • Audio CAPTCHA buttons (speaker icons, play buttons)
  • BotDetect engine identification
"""

from __future__ import annotations

import re
import time


# ── Scoring constants ────────────────────────────────────────

_CAPTCHA_KW = {
    "captcha": 5, "kaptcha": 5, "securimage": 5, "jcaptcha": 5,
    "mtcap-image": 20, "mtcaptcha": 3, "mtcap": 2,
    "verify": 3, "verif": 3, "validation": 3, "challenge": 3, "servlet": 3,
    "security": 2, "code": 2, "generate": 2, "vcode": 4,
    "imgcode": 4, "checkcode": 4, "human": 2, "bot": 2, "random": 1,
}
_NOISE_KW = {
    "logo", "banner", "icon", "arrow", "header", "footer",
    "avatar", "profile", "photo",
    # App store badges & social media
    "store", "android", "apple", "google", "play", "download",
    "badge", "appstore", "playstore", "facebook", "twitter",
    "instagram", "linkedin", "youtube", "social",
    # Common UI elements
    "flag", "country", "language", "payment", "card", "qr",
    "advertisement", "sponsor", "promo", "slider", "carousel",
}
_MIN_SCORE = 6


# ── Context check ────────────────────────────────────────────

def _has_captcha_context(el) -> bool:
    try:
        return bool(el.evaluate("""node => {
            const c = node.closest('tr,div,td,form,section') || node.parentElement;
            if (!c) return false;
            const txt = c.innerText.toLowerCase();
            return c.querySelectorAll('input').length > 0
                && /captcha|verify|code|vcode|enter|type/.test(txt);
        }"""))
    except Exception:
        return False


# ── Element scoring ──────────────────────────────────────────

def _score_el(el) -> int:
    try:
        box = el.bounding_box()
        if not box:
            return -1
        w, h = box["width"], box["height"]
    except Exception:
        return -1
    if w < 60 or h < 20:
        return -1

    score = 0
    aspect = w / h if h else 0
    if 2.0 <= aspect <= 8.0:
        score += 3
    elif 1.2 <= aspect <= 12.0:
        score += 1
    if 60 <= w <= 350 and 20 <= h <= 120:
        score += 2

    attrs = " ".join(filter(None, [
        el.get_attribute("src") or "",
        el.get_attribute("id") or "",
        el.get_attribute("class") or "",
        el.get_attribute("alt") or "",
        el.get_attribute("name") or "",
        el.get_attribute("title") or "",
    ])).lower()

    for kw, pts in _CAPTCHA_KW.items():
        if kw in attrs:
            score += pts
    for nw in _NOISE_KW:
        if nw in attrs:
            score -= 3

    src = (el.get_attribute("src") or "").lower()
    if "data:" in src:
        score += 2
    if any(x in src for x in ("servlet", "generate", "captcha", "verify", "random")):
        score += 3
    if re.search(r"\?.*=", src):
        score += 1
    if score >= 1 and _has_captcha_context(el):
        score += 4

    return score


# ── Find CAPTCHA image element ───────────────────────────────

def find_captcha_element(pg, log=None):
    """Find the CAPTCHA image element on the page using a scoring system. [v3-patched]"""
    candidates = []

    selector = "embed:visible, object:visible, img:visible, canvas:visible, .mtcap-image:visible, [id*='mtcap-image']:visible"
    for frame in pg.frames:
        try:
            candidates.extend(frame.query_selector_all(selector))
        except Exception as exc:
            if log:
                log(f"[detect-diag] frame query error: {exc}")

    if log:
        log(f"[detect-diag] {len(candidates)} candidate(s) after selector scan")

    best_el = None
    best_score = -1
    seen_boxes = set()

    for el in candidates:
        try:
            box = el.bounding_box()
            if not box:
                continue
            box_key = (int(box["x"]), int(box["y"]),
                       int(box["width"]), int(box["height"]))
            if box_key in seen_boxes:
                continue
            seen_boxes.add(box_key)

            score = _score_el(el)

            if log:
                try:
                    tag = el.evaluate("el => el.tagName").lower()
                    eid = el.get_attribute("id") or el.get_attribute("class") or ""
                    log(f"[detect-diag] <{tag}> id/cls={eid[:40]} box={box_key} score={score}")
                except Exception:
                    pass

            # Boost for known webtel element
            if (el.get_attribute("id") == "CaptchaIMG"
                    and el.evaluate("el => el.tagName").lower() == "embed"):
                score += 50

            if score > best_score:
                best_score = score
                best_el = el
            elif score == best_score and score >= _MIN_SCORE and best_el:
                # Tie-breaker: prefer the image that has a valid input field nearby,
                # and heavily prefer input fields that have CAPTCHA keywords in their attributes!
                best_inputs = find_input_elements(pg, best_el)
                el_inputs = find_input_elements(pg, el)
                
                kw = {"captcha", "code", "verify", "security", "text", "answer", "cap"}
                def has_kw(inps):
                    if not inps: return False
                    for i in inps:
                        try:
                            attrs = " ".join(filter(None, [i.get_attribute("id"), i.get_attribute("name"), i.get_attribute("placeholder"), i.get_attribute("class")])).lower()
                            if any(k in attrs for k in kw): return True
                        except: pass
                    return False
                
                best_has_kw = has_kw(best_inputs)
                el_has_kw = has_kw(el_inputs)
                
                if not best_has_kw and el_has_kw:
                    best_el = el
                elif not best_inputs and el_inputs and not best_has_kw:
                    best_el = el
        except Exception:
            continue

    if best_el and best_score >= _MIN_SCORE:
        if log:
            try:
                src = best_el.get_attribute("src") or best_el.get_attribute("id") or "canvas/object"
                tag = best_el.evaluate("el => el.tagName").lower()
                log(f"[detect] Found best CAPTCHA element (score: {best_score}) → <{tag}> {src[:80]}")
            except Exception:
                log(f"[detect] Found best CAPTCHA element (score: {best_score})")
        return best_el

    if log:
        log(f"[detect] No CAPTCHA element found with sufficient score (best={best_score}, min={_MIN_SCORE}, candidates={len(candidates)})")
    return None


# ── Find input elements ──────────────────────────────────────

def find_input_elements(pg, cap_el, audio_btn=None) -> list:
    """
    Returns a list of input elements to type into.

    If the CAPTCHA uses individual character boxes (like webtel.in),
    returns them sorted left-to-right.
    Otherwise returns a single-element list with the normal text input.
    """
    ref_el = cap_el or audio_btn
    ref_frame = ref_el.owner_frame() if ref_el else pg.main_frame
    
    # Priority 1: webtel.in specific
    try:
        cap_inputs = ref_frame.query_selector_all(".captcha-input input[maxlength='1']")
        if cap_inputs:
            visible = [inp for inp in cap_inputs if inp.is_visible()]
            if len(visible) >= 3:
                return visible
    except Exception:
        pass

    # Also try by ID pattern cap1, cap2, ...
    try:
        ordered = []
        for i in range(1, 10):
            inp = ref_frame.query_selector(f"#cap{i}")
            if inp and inp.is_visible():
                ordered.append(inp)
            else:
                break
        if len(ordered) >= 3:
            return ordered
    except Exception:
        pass

    # Priority 2: Multiple small inputs on the same row
    try:
        if ref_el:
            cap_box = ref_el.bounding_box()
            if cap_box:
                cap_cy = cap_box["y"] + cap_box["height"] / 2
                all_inputs = ref_frame.query_selector_all(
                    "input[type='text'], input[type='tel'], input[type='number'], input[type='email'], input[type='search'], input[maxlength='1'], input:not([type]), input.mtcap-inputtext"
                )
                row_inputs = []
                for inp in all_inputs:
                    if not inp.is_visible():
                        continue
                    if inp.get_attribute("readonly"):
                        continue
                    b = inp.bounding_box()
                    if not b:
                        continue
                    inp_cy = b["y"] + b["height"] / 2
                    if abs(inp_cy - cap_cy) < 80:
                        row_inputs.append((b["x"], b["width"], inp))

                row_inputs.sort(key=lambda x: x[0])
                narrow = [(x, w, el) for x, w, el in row_inputs if w < 55]
                if len(narrow) >= 3:
                    return [el for _, _, el in narrow]
    except Exception:
        pass

    # Form-based fallback
    try:
        if ref_el:
            fh = ref_frame.evaluate_handle("el => el.closest('form')", ref_el)
            form = fh.as_element() if fh else None
            if form:
                kw = {"captcha", "code", "verify", "security", "text", "answer", "cap"}
                inputs = form.query_selector_all(
                    "input[type='text'], input[type='tel'], input[type='number'], input[type='email'], input[type='search'], input[maxlength='1'], input:not([type]), input.mtcap-inputtext"
                )
                for inp in inputs:
                    if inp.get_attribute("readonly"):
                        continue
                    attrs = " ".join(filter(None, [
                        inp.get_attribute("id") or "",
                        inp.get_attribute("name") or "",
                        inp.get_attribute("placeholder") or "",
                        inp.get_attribute("class") or "",
                    ])).lower()
                    if any(k in attrs for k in kw) and inp.is_visible():
                        return [inp]
                for inp in reversed(inputs):
                    if inp.is_visible():
                        return [inp]
    except Exception:
        pass

    # Distance fallback
    try:
        if ref_el:
            cap_box = ref_el.bounding_box()
            if cap_box:
                best, best_dist = None, float("inf")
                for inp in ref_frame.query_selector_all(
                    "input[type='text'], input[type='tel'], input[type='number'], input[type='email'], input[type='search'], input[maxlength='1'], input:not([type]), input.mtcap-inputtext"
                ):
                    if not inp.is_visible() or inp.get_attribute("readonly"):
                        continue
                    b = inp.bounding_box()
                    if not b:
                        continue
                    dx = abs(b["x"] - cap_box["x"])
                    dy = abs(b["y"] - cap_box["y"])
                    
                    if dx > 600 or dy > 250:
                        continue
                        
                    d = dx + (dy * 3)
                    if d < best_dist:
                        best_dist, best = d, inp
                if best:
                    return [best]
    except Exception:
        pass

    return []


# ── NEW: Audio CAPTCHA button detection ──────────────────────

def find_audio_button(pg, cap_el, log=None) -> object | None:
    """
    Detect an audio CAPTCHA button near the CAPTCHA image.

    Looks for speaker icons, audio play buttons, or links with
    audio-related text/attributes near the CAPTCHA element.
    """
    cap_box = None
    if cap_el:
        try:
            cap_box = cap_el.bounding_box()
        except Exception:
            pass

    cap_cx = cap_box["x"] + cap_box["width"] / 2 if cap_box else None
    cap_cy = cap_box["y"] + cap_box["height"] / 2 if cap_box else None

    # Selectors for audio buttons (ordered by specificity)
    audio_selectors = [
        # Explicit text buttons
        "button:has-text('Play Audio')",
        "button:has-text('Play audio')",
        "button:has-text('Audio CAPTCHA')",
        "a:has-text('Play Audio')",
        "a:has-text('Audio CAPTCHA')",
        # BotDetect audio button
        "a[id*='SoundLink']",
        "a[href*='get=sound']",
        # MTCaptcha audio button
        "[class*='mtcap-audio' i]",
        # Common audio captcha patterns
        "[id*='audio' i][id*='captcha' i]",
        "[class*='audio' i][class*='captcha' i]",
        "button[aria-label*='audio' i]",
        "a[title*='audio' i]",
        "a[title*='sound' i]",
        "a[title*='listen' i]",
        "button[title*='audio' i]",
        "button[title*='sound' i]",
        "button[title*='listen' i]",
        # Speaker icon images
        "img[alt*='audio' i]",
        "img[alt*='sound' i]",
        "img[alt*='speaker' i]",
        "img[alt*='listen' i]",
        "img[src*='audio' i]",
        "img[src*='sound' i]",
        "img[src*='speaker' i]",
        # Generic icon buttons near captcha
        "[class*='speaker' i]",
        "[class*='sound' i]",
        # ECI-style refresh/audio
        "img[src*='reload' i]",
    ]

    combined_selector = ", ".join(audio_selectors)
    
    for frame in pg.frames:
        try:
            elements = frame.query_selector_all(combined_selector)
            for el in elements:
                if not el.is_visible():
                    continue
                try:
                    el_box = el.bounding_box()
                    if not el_box:
                        continue
                    
                    # Instead of checking the specific string that matched (which we lost by combining), 
                    # we evaluate the element's outerHTML or text content to see if it's highly specific
                    outer_html = (el.evaluate("el => el.outerHTML") or "").lower()
                    text_content = (el.evaluate("el => el.innerText") or "").lower()
                    
                    is_highly_specific = "play audio" in text_content or "audio captcha" in text_content or "soundlink" in outer_html or ("audio" in outer_html and "captcha" in outer_html)
                    
                    if not cap_el or is_highly_specific:
                        if log:
                            log(f"[detect] Found audio button")
                        return el

                    # Otherwise, must be close to the CAPTCHA (within 300px)
                    el_cx = el_box["x"] + el_box["width"] / 2
                    el_cy = el_box["y"] + el_box["height"] / 2
                    dist = abs(el_cx - cap_cx) + abs(el_cy - cap_cy)
                    if dist < 300:
                        if log:
                            log(f"[detect] Found audio button nearby")
                        return el
                except Exception:
                    continue
        except Exception:
            continue

    # Also check for <audio> elements on the page
    try:
        audios = pg.query_selector_all("audio")
        for audio in audios:
            src = audio.get_attribute("src") or ""
            if "captcha" in src.lower() or "sound" in src.lower():
                if log:
                    log(f"[detect] Found audio element: {src[:60]}")
                return audio
    except Exception:
        pass

    return None


# ── BotDetect detection ──────────────────────────────────────

def is_botdetect(cap_el) -> bool:
    """Check if the CAPTCHA is from BotDetect (always uppercase)."""
    try:
        src = (cap_el.get_attribute("src") or "").lower()
        return "botdetect" in src
    except Exception:
        return False
