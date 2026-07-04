import time
import os
import re

_RECAPTCHA_IFRAME_SELECTORS = [
    "iframe[src*='google.com/recaptcha']",
    "iframe[src*='recaptcha.net']",
    "iframe[src*='recaptcha']",
    "iframe[title='reCAPTCHA']",
    "iframe[title*='recaptcha' i]",
]

def _trigger_recaptcha_render(page, log):
    try:
        for sel in ["input[type='text']", "input[type='email']",
                    "input[name*='user' i]", "input[name*='login' i]",
                    "input[name*='reg' i]", "input:not([type='hidden'])"]:
            inp = page.ele(f"css:{sel}", timeout=1)
            if inp and inp.states.is_displayed:
                try: inp.click(by_js=True)
                except Exception: pass
                log(f"Clicked form field ({sel}) to trigger reCAPTCHA render")
                time.sleep(1.5)
                break
    except Exception as exc: pass

    try:
        container = page.ele("css:.g-recaptcha, [data-sitekey]", timeout=1)
        if container:
            container.scroll.to_see()
            time.sleep(1.0)
    except Exception as exc: pass

    try:
        rendered = page.run_js("""return (function() {
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
        })();""")
        if rendered in ("rendered", "already_rendered"):
            time.sleep(1.5)
    except Exception as exc: pass

def _wait_for_recaptcha_iframe(page, log, timeout: float = 15.0) -> bool:
    _trigger_recaptcha_render(page, log)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for sel in _RECAPTCHA_IFRAME_SELECTORS:
            try:
                el = page.ele(f"css:{sel}", timeout=1)
                if el:
                    time.sleep(1.2)
                    return True
            except Exception: pass
        time.sleep(0.5)
    return False

def _get_challenge_frame(page, timeout=3.0, log=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for f in page.get_frames():
                try:
                    if f.ele("css:#recaptcha-audio-button", timeout=0.1) or f.ele("css:.rc-imageselect-instructions", timeout=0.1):
                        return f
                except Exception: pass
        except Exception: pass
        time.sleep(0.5)
    return None

def _get_anchor_frame(page, timeout=3.0, log=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for f in page.get_frames():
                try:
                    if f.ele("css:#recaptcha-anchor", timeout=0.1):
                        return f
                except Exception: pass
        except Exception: pass
        time.sleep(0.5)
    return None

def solve_recaptcha_drission(page, log, app) -> bool:
    """Solve reCAPTCHA entirely using DrissionPage for maximum stealth."""
    
    log("[dp-stealth] Starting DrissionPage reCAPTCHA solver...")
    
    if not _wait_for_recaptcha_iframe(page, log):
        return False
        
    # Wait for anchor
    af = _get_anchor_frame(page, timeout=10.0, log=log)
    if not af:
        log("[dp-stealth] Anchor frame not found")
        return False
        
    try:
        checkbox = af.ele("css:#recaptcha-anchor", timeout=2)
        if not checkbox:
            return False
            
        checked = af.ele("css:.recaptcha-checkbox-checked", timeout=1)
        if checked:
            log("[dp-stealth] Already checked!")
            return True
            
        log("[dp-stealth] Clicking checkbox...")
        checkbox.click()
        time.sleep(2)
        
        checked = af.ele("css:.recaptcha-checkbox-checked", timeout=1)
        if checked:
            log("[dp-stealth] Passed with 1-click!")
            return True
            
    except Exception as e:
        log(f"[dp-stealth] Checkbox click failed: {e}")
        return False

    # Need to solve challenge
    cf = _get_challenge_frame(page, timeout=5.0, log=log)
    if not cf:
        return False
        
    try:
        audio_btn = cf.ele("css:#recaptcha-audio-button", timeout=2)
        if audio_btn and audio_btn.states.is_displayed:
            log("[dp-stealth] Clicking audio challenge button...")
            audio_btn.click(by_js=True)
            time.sleep(2)
    except Exception as e:
        pass

    for audio_round in range(6):
        if app and not app.running: return False
        
        cf = _get_challenge_frame(page, timeout=3.0, log=log)
        if not cf: break
        
        play_btn = cf.ele("css:.rc-audiochallenge-play-button", timeout=2)
        if not play_btn: break
        
        log("[dp-stealth] Solving audio...")
        
        intercepted_audio_path = "captcha_audio.mp3"
        try:
            if os.path.exists(intercepted_audio_path):
                os.remove(intercepted_audio_path)
        except: pass
        
        # Listen for the actual audio file
        page.listen.start('audio.mp3')
        play_btn.click(by_js=True)
        
        packet = page.listen.wait(timeout=5)
        audio_saved = False
        if packet:
            body = packet.response.body
            if isinstance(body, str):
                body = body.encode('latin-1')
            with open(intercepted_audio_path, "wb") as f:
                f.write(body)
            audio_saved = True
            log(f"[dp-stealth] Intercepted audio payload ({len(body)} bytes)")
        page.listen.stop()
        
        if not audio_saved:
            log("[dp-stealth] Failed to intercept audio! Trying alternative method...")
            try:
                # Fallback: get the audio source URL and download it directly
                audio_el = cf.ele("css:#audio-source", timeout=1)
                if audio_el:
                    src = audio_el.attr("src")
                    if src:
                        log(f"[dp-stealth] Found audio source: {src[:50]}...")
                        import urllib.request
                        req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=5) as response:
                            with open(intercepted_audio_path, "wb") as f:
                                f.write(response.read())
                        audio_saved = True
                        log("[dp-stealth] Downloaded audio via fallback!")
            except Exception as e:
                log(f"[dp-stealth] Fallback download failed: {e}")
                
        if not audio_saved:
            break
            
        from audio_solver import _transcribe
        text = _transcribe(intercepted_audio_path, log, is_recaptcha=True)
        if app and not app.running: return False
        
        if text:
            text = re.sub(r'[.,!?;:\'"()]', '', text).strip()
            try:
                inp = cf.ele("css:#audio-response", timeout=2)
                if inp:
                    inp.input(text, clear=True)
                    log(f"[dp-stealth] Typed answer natively: '{text}'")
            except Exception as e:
                log(f"[dp-stealth] Native typing failed: {e}")
                
            time.sleep(1.0)
            
            try:
                verify_btn = cf.ele("css:#recaptcha-verify-button", timeout=2)
                if verify_btn:
                    verify_btn.click()
                    log("[dp-stealth] Clicked Verify button")
            except Exception as e:
                log(f"[dp-stealth] Verify click failed: {e}")
            
            time.sleep(5.0)
            
            af2 = _get_anchor_frame(page, timeout=3.0, log=log)
            if af2:
                if af2.ele("css:.recaptcha-checkbox-checked", timeout=1):
                    log("[dp-stealth] reCAPTCHA passed on audio!")
                    return True
                    
            cf = _get_challenge_frame(page, timeout=3.0, log=log)
            if not cf:
                return True
                
            err = cf.ele("css:.rc-doscaptcha-header-text", timeout=1)
            if err and err.states.is_displayed and "automated queries" in (err.text or "").lower():
                log("[dp-stealth] Audio challenge blocked by anti-bot.")
                return False
                
            reload_btn = cf.ele("css:#recaptcha-reload-button", timeout=1)
            if reload_btn:
                log("[dp-stealth] Reloading for fresh audio challenge...")
                reload_btn.click(by_js=True)
                time.sleep(2)
        else:
            break
            
    return False
