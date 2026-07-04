"""
audio_solver.py — Audio CAPTCHA Solver (Tier 2)

Uses faster-whisper (local Whisper inference) to transcribe audio
CAPTCHA challenges.  Runs 100% locally with zero API cost.

Flow:
  1. Detect audio button near CAPTCHA element
  2. Trigger audio playback / download
  3. Capture or download the audio file
  4. Transcribe with Whisper 'small' model
  5. Clean and return the transcription
"""

from __future__ import annotations

import os
import re
import time
import tempfile
import threading
from typing import Optional

# ── Whisper model (lazy-loaded) ──────────────────────────────

_whisper_model = None
_whisper_lock = threading.Lock()
_whisper_ready = threading.Event()

WHISPER_MODEL_SIZE = "small"  # ~500MB, high accuracy for short clips


def _load_whisper():
    """Load the Whisper model in a background thread."""
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        print(f"faster-whisper ({WHISPER_MODEL_SIZE}) ready")
    except ImportError:
        print("faster-whisper not installed — audio solver unavailable")
    except Exception as exc:
        print(f"Whisper load failed: {exc}")
    finally:
        _whisper_ready.set()


# Boot whisper in background
threading.Thread(target=_load_whisper, daemon=True).start()


# ── Audio download helpers ───────────────────────────────────

def _download_audio_from_button(pg, audio_btn, log, challenge_frame=None) -> Optional[str]:
    """
    Click the audio button and capture the resulting audio.

    Returns the path to the downloaded audio file, or None.
    """
    # Check if button has a direct href/src to an audio file
    href = audio_btn.get_attribute("href") or ""
    src = audio_btn.get_attribute("src") or ""
    audio_url = ""

    def is_likely_audio(url: str) -> bool:
        lower_url = url.lower()
        if any(ext in lower_url for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js"]):
            return False
        return "sound" in lower_url or "audio" in lower_url or ".wav" in lower_url or ".mp3" in lower_url

    if href and is_likely_audio(href):
        audio_url = href
    elif src and is_likely_audio(src):
        audio_url = src

    # BotDetect pattern: href contains "get=sound"
    if "get=sound" in href.lower():
        audio_url = href

    if audio_url:
        return _fetch_audio_url(pg, audio_url, log, challenge_frame=challenge_frame)

    # Check if we ALREADY intercepted the audio payload during the challenge load
    try:
        intercepted_audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio_intercepted.mp3")
        if os.path.exists(intercepted_audio_path):
            import time, shutil
            # If the file was written recently, it's our payload
            if time.time() - os.path.getmtime(intercepted_audio_path) < 30:
                audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio.mp3")
                shutil.copy2(intercepted_audio_path, audio_path)
                os.remove(intercepted_audio_path)
                log(f"[audio] Using pre-intercepted audio payload ({os.path.getsize(audio_path)} bytes) → {audio_path}")
                return audio_path
    except Exception as e:
        log(f"[audio] Error checking intercepted audio: {e}")

    # Click the button and listen for network audio response
    try:
        # Set up a listener for audio responses before clicking
        audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio.mp3")

        # Try to intercept the audio via page.expect_response
        with pg.expect_response(
            lambda resp: (
                any(ct in (resp.headers.get("content-type", "") or "") for ct in ("audio/", "application/octet-stream"))
                or ("audio" in resp.url.lower() and "image" not in (resp.headers.get("content-type", "") or ""))
                or ("sound" in resp.url.lower() and "image" not in (resp.headers.get("content-type", "") or ""))
                or resp.url.lower().endswith(".wav")
                or resp.url.lower().endswith(".mp3")
                or ("ashx" in resp.url.lower() and "image" not in (resp.headers.get("content-type", "") or "") and "text/html" not in (resp.headers.get("content-type", "") or ""))
            ),
            timeout=5000,
        ) as resp_info:
            audio_btn.click(force=True)
            log("[audio] Clicked audio button, waiting for audio response...")

        response = resp_info.value
        if response:
            body = response.body()
            with open(audio_path, "wb") as f:
                f.write(body)
            log(f"[audio] Downloaded audio ({len(body)} bytes) → {audio_path}")
            return audio_path

    except Exception as e:
        # Before failing, check if TTS was intercepted!
        try:
            tts_text = pg.evaluate("window._interceptedSpeech")
            if tts_text:
                log(f"[audio] Intercepted SpeechSynthesis TTS: '{tts_text}'")
                return "TTS:" + tts_text
        except Exception:
            pass
        
        log(f"[audio] Response capture failed: {e}")

    # Fallback: Intercept play() and check DOM within the correct frame
    try:
        # Inject interceptor into the correct frame
        audio_btn.evaluate("""el => {
            const win = el.ownerDocument.defaultView;
            if (!win._audioIntercepted) {
                win._interceptedAudioUrls = [];
                const origPlay = win.HTMLAudioElement.prototype.play;
                win.HTMLAudioElement.prototype.play = function() {
                    try {
                        if (this.src) {
                            win._interceptedAudioUrls.push(this.src);
                        } else {
                            const source = this.querySelector('source');
                            if (source && source.src) win._interceptedAudioUrls.push(source.src);
                        }
                    } catch(e) {}
                    return origPlay.apply(this, arguments);
                };
                win._audioIntercepted = true;
            } else {
                win._interceptedAudioUrls = [];
            }
        }""")
        
        audio_btn.click(force=True)
        time.sleep(1.5)

        # 1. Check intercepted URLs in the correct frame
        intercepted = audio_btn.evaluate("el => el.ownerDocument.defaultView._interceptedAudioUrls")
        if intercepted and len(intercepted) > 0:
            log(f"[audio] Intercepted JS audio play: {intercepted[0][:80]}")
            return _fetch_audio_url(pg, intercepted[0], log, challenge_frame=challenge_frame)

        # 2. Check DOM for <audio> elements in the correct frame
        audio_src = audio_btn.evaluate("""el => {
            const audio = el.ownerDocument.querySelector('audio');
            if (audio) {
                if (audio.src) return audio.src;
                const source = audio.querySelector('source');
                if (source && source.src) return source.src;
            }
            return null;
        }""")
        if audio_src:
            log(f"[audio] Found <audio> tag in DOM: {audio_src[:80]}")
            return _fetch_audio_url(pg, audio_src, log, challenge_frame=challenge_frame)
                
    except Exception as e:
        log(f"[audio] Fallback audio detection failed: {e}")

    return None


def _fetch_audio_url(pg, url: str, log, challenge_frame=None) -> Optional[str]:
    """Download audio from a URL using the page's fetch context."""
    audio_path = os.path.join(tempfile.gettempdir(), "captcha_audio.mp3")

    try:
        # Make the URL absolute
        abs_url = pg.evaluate(f"""() => {{
            const a = document.createElement('a');
            a.href = {repr(url)};
            return a.href;
        }}""")

        log(f"[audio] Fetching audio from: {abs_url[:80]}")

        # Try fetching from the challenge frame first (same origin as reCAPTCHA audio)
        # This avoids CORS issues that occur when fetching from the main page
        fetch_contexts = []
        if challenge_frame is not None:
            fetch_contexts.append(("challenge frame", challenge_frame))
        fetch_contexts.append(("main page", pg))

        for ctx_name, ctx in fetch_contexts:
            try:
                b64_data = ctx.evaluate("""async (url) => {
                    const resp = await fetch(url, { credentials: 'include', cache: 'force-cache' });
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
                    audio_bytes = base64.b64decode(b64_data)
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    log(f"[audio] Downloaded {len(audio_bytes)} bytes via {ctx_name} → {audio_path}")
                    return audio_path
                else:
                    log(f"[audio] fetch from {ctx_name} returned null")
            except Exception as e:
                log(f"[audio] fetch from {ctx_name} failed: {e}")
    except Exception as e:
        log(f"[audio] URL resolution failed: {e}")

    # Fallback: Playwright API request (no iframe cookies, last resort)
    try:
        response = pg.context.request.get(url if not 'abs_url' in dir() else abs_url)
        if response.ok:
            with open(audio_path, "wb") as f:
                f.write(response.body())
            log(f"[audio] API request OK → {audio_path}")
            return audio_path
    except Exception as e:
        log(f"[audio] API fallback failed: {e}")

    return None


# ── Whisper transcription ────────────────────────────────────

def _transcribe(audio_path: str, log, is_recaptcha: bool = False) -> str:
    """Transcribe an audio file using faster-whisper."""
    if not _whisper_ready.wait(timeout=60):
        log("[audio] Whisper model timed out loading")
        return ""

    if _whisper_model is None:
        log("[audio] Whisper model not available")
        return ""

    try:
        segments, info = _whisper_model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            word_timestamps=False,
            vad_filter=False,  # Disabled: CAPTCHA audio has long pauses between letters
                               # which VAD misinterprets as "end of speech", cutting off early
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        raw_text = " ".join(text_parts).strip()
        log(f"[audio] Raw transcription: '{raw_text}'")

        if is_recaptcha:
            log(f"[audio] Bypassing string cleanup for reCAPTCHA phrase")
            return raw_text

        # Clean: CAPTCHAs are typically spelled out character by character
        # "R E P B V" → "REPBV"
        # "are ee pee bee vee" → need phonetic mapping
        cleaned = _clean_transcription(raw_text)
        log(f"[audio] Cleaned: '{cleaned}'")

        return cleaned

    except Exception as e:
        log(f"[audio] Transcription failed: {e}")
        return ""


def _clean_transcription(raw: str) -> str:
    """
    Clean up Whisper transcription of a CAPTCHA audio.

    CAPTCHAs typically spell out characters one at a time.
    Whisper may output them as:
      - Single characters separated by spaces: "R E P B V"
      - Phonetic words: "are ee pee bee vee"
      - Mixed: "R E P bee vee"
    """
    # Phonetic → character mapping
    phonetic_map = {
        "ay": "a", "ay.": "a",
        "bee": "b", "be": "b",
        "see": "c", "sea": "c", "cee": "c",
        "dee": "d", "de": "d",
        "ee": "e",
        "eff": "f", "ef": "f",
        "gee": "g", "jee": "g",
        "aitch": "h", "ach": "h", "age": "h", "aych": "h",
        "eye": "i", "aye": "i",
        "jay": "j",
        "kay": "k", "key": "k",
        "el": "l", "ell": "l",
        "em": "m",
        "en": "n",
        "oh": "o",
        "pee": "p", "pe": "p",
        "queue": "q", "cue": "q", "kew": "q",
        "are": "r", "ar": "r",
        "ess": "s", "es": "s",
        "tee": "t", "te": "t",
        "you": "u", "yu": "u",
        "vee": "v", "ve": "v",
        "double you": "w", "double u": "w", "doubleyou": "w",
        "ex": "x",
        "why": "y", "wye": "y",
        "zee": "z", "zed": "z", "zee.": "z",
        # Numbers
        "zero": "0", "oh": "0",
        "one": "1", "won": "1",
        "two": "2", "to": "2", "too": "2",
        "three": "3",
        "four": "4", "for": "4", "fore": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8", "ate": "8",
        "nine": "9",
        "and": "nd", # Common Whisper hallucination for "N D"
    }

    raw = raw.replace(".", " ").replace(",", " ").replace("-", " ")
    raw = raw.strip().lower()

    # First try: if it's single characters separated by spaces
    parts = raw.split()
    if all(len(p) == 1 and p.isalnum() for p in parts):
        return "".join(parts).upper()

    # Second try: phonetic mapping
    result = []
    i = 0
    words = raw.split()
    while i < len(words):
        # Try two-word combinations first (e.g., "double you")
        if i + 1 < len(words):
            two_word = words[i] + " " + words[i + 1]
            if two_word in phonetic_map:
                result.append(phonetic_map[two_word])
                i += 2
                continue

        word = words[i].rstrip(".,!?")

        if word in phonetic_map:
            result.append(phonetic_map[word])
        elif len(word) == 1 and word.isalnum():
            result.append(word)
        elif word.isalnum():
            # Might be the actual text
            result.append(word)
        i += 1

    text = "".join(result)

    # Remove any remaining non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", "", text)

    return text.upper()


# ── Public API ───────────────────────────────────────────────

def solve_audio_captcha(pg, cap_el, audio_btn, log, is_recaptcha: bool = False, challenge_frame=None) -> Optional[str]:
    """
    Attempt to solve an audio CAPTCHA by downloading the audio file
    and passing it through local Whisper.
    Also intercepts SpeechSynthesis (TTS) API for sites that use local speech.
    """
    if not is_available():
        log("[audio] whisper module not installed, skipping audio solver")
        return None

    # Inject TTS interceptor before doing anything
    try:
        pg.evaluate("""() => {
            if (!window._ttsIntercepted) {
                window._interceptedSpeech = null;
                if (window.speechSynthesis) {
                    const originalSpeak = window.speechSynthesis.speak;
                    window.speechSynthesis.speak = function(utterance) {
                        window._interceptedSpeech = utterance.text;
                    };
                }
                window._ttsIntercepted = true;
            } else {
                window._interceptedSpeech = null;
            }
        }""")
    except Exception:
        pass

    log("[audio] ── Audio CAPTCHA solver ──")

    audio_path = _download_audio_from_button(pg, audio_btn, log, challenge_frame=challenge_frame)
    if not audio_path:
        # Final check if TTS was caught during fallback
        try:
            tts_text = pg.evaluate("window._interceptedSpeech")
            if tts_text:
                log(f"[audio] Intercepted SpeechSynthesis TTS: '{tts_text}'")
                audio_path = "TTS:" + tts_text
        except Exception:
            pass
            
    if not audio_path:
        log("[audio] Failed to download audio")
        return None

    if audio_path.startswith("TTS:"):
        text = audio_path[4:]
        log(f"[audio] Bypassing Whisper, using TTS intercept: '{text}'")
    else:
        log("[audio] Running local Whisper model...")
        text = _transcribe(audio_path, log, is_recaptcha=is_recaptcha)
        log(f"[audio] Raw transcription: '{text}'")

    if not text:
        return None

    # UP Gov sites often have broken audio buttons that just read the static boilerplate
    if "e.g. for 1" in text.lower() and "enter" in text.lower() and not re.search(r'\d+\s*[\+\-\*]\s*\d+', text):
        # But wait, the text was "E.g. for 1+3,". The regex \d+ \+ \d+ WOULD MATCH "1+3"!
        # Let's just check if it ends with "for 1+3, " or if the last numbers are exactly 1 and 3.
        pass

    # A better check: if the audio text is just the exact broken string from UP sites
    if "solve this simple math problem and enter the result. e.g. for 1+3" in text.lower().replace("  ", " "):
        # Check if there are any OTHER numbers besides 1, 3, and 4 in the text
        nums = re.findall(r'\d+', text)
        if nums == ['1', '3'] or nums == ['1', '3', '4']:
            log("[audio] Detected broken UP Gov audio (reads boilerplate only) -> falling back to visual OCR")
            return None

    if not audio_path.startswith("TTS:"):
        # Cleanup temp file
        try:
            os.remove(audio_path)
        except Exception:
            pass

    log(f"[audio] ✓ Solved: '{text}'")
    return text


def is_available() -> bool:
    """Check if the Whisper model is loaded and ready."""
    return _whisper_model is not None
