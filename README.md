# Universal CAPTCHA Solver

A robust, automated CAPTCHA solver built with Python, Playwright, and DrissionPage. This project is designed to seamlessly detect, intercept, and solve a wide variety of modern CAPTCHA and bot-protection challenges.

## Features & Supported CAPTCHAs

* **Image & Text CAPTCHAs**: Solves standard visual alphanumeric and math challenges using an ensemble OCR engine (EasyOCR/Tesseract).
* **Audio Fallback**: Automatically attempts to solve CAPTCHAs via their audio alternative using Whisper if the visual solve confidence is too low.
* **reCAPTCHA (v2 & v3)**: Uses stealth techniques via DrissionPage and Playwright to bypass Google's reCAPTCHA.
* **hCaptcha**: Automatically solves hCaptcha challenges, including text-logic and image selection variants.
* **Cloudflare Turnstile**: Detects and clears Turnstile security checkpoints.
* **Geetest**: Framework in place to route and handle Geetest puzzle challenges.
* **DOM Math CAPTCHAs**: Detects inline math verifications and slider combinations.

## Architecture

The solver uses a dual-engine approach to maximize success rates and stealth:
1. **Playwright (CDP)**: Used for deep DOM manipulation, complex element interactions, and running the primary solver logic.
2. **DrissionPage**: Used as a stealth engine to bypass advanced anti-bot fingerprinting (e.g., Cloudflare Turnstile) before handing control back to Playwright.

### Key Files
* `captcha_solver.py`: The main entry point and routing engine. Determines the type of CAPTCHA on the page and delegates to the appropriate subsystem.
* `ocr_engine.py`: Handles image preprocessing, noise reduction, and text extraction.
* `audio_solver.py`: Downloads audio payloads and transcribes them for text input.
* `recaptcha_dp.py` & `hcaptcha_solver.py`: Dedicated routines for bypassing specific third-party providers.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/universal-captcha-solver.git
   cd universal-captcha-solver
   ```

2. **Install dependencies:**
   Make sure you have Python 3.8+ installed. 
   *(Note: Ensure you have your `requirements.txt` populated with the necessary packages)*
   ```bash
   pip install playwright DrissionPage Pillow easyocr
   ```

3. **Install Playwright Browsers:**
   ```bash
   playwright install chromium
   ```

## Usage

Import the solver into your automation scripts and pass your active Playwright page instance to the solver router.

```python
from captcha_solver import _solve_image_captcha

# Assuming 'pg' is your Playwright page object and 'log' is a logging function
success = _solve_image_captcha(pg, log)
if success:
    print("CAPTCHA bypassed successfully!")
```

## Disclaimer

This project is intended for educational purposes, accessibility research, and automated testing of your own systems. Ensure you comply with the Terms of Service of any website you interact with.
