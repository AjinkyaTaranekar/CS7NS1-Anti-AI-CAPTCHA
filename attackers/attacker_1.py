import asyncio
import base64
import logging
import os
import random
import tempfile
from time import time

import cv2
import easyocr
import requests
# urllib.parse not needed here
from playwright.async_api import async_playwright
from utils import (append_attack_result, draw_captcha, human_like_move_between,
                   human_like_scroll)

# Silenzia i log inutili / Mute logs
os.environ["OMP_NUM_THREADS"] = "1"

def get_all_channels(img):
    """
    Estrae 9 canali colore diversi dall'immagine originale
    Extracts 9 different color channels from the original image
    """
    channels = {}
    
    # 1. Spazio RGB
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
    else:
        # Se l'immagine è già grayscale / If already grayscale
        return {'Gray': img}

    channels['RGB_Blue'] = b
    channels['RGB_Green'] = g
    channels['RGB_Red'] = r
    
    # 2. Spazio HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    channels['HSV_Hue'] = h
    channels['HSV_Sat'] = s
    channels['HSV_Val'] = v
    
    # 3. Spazio LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    channels['LAB_L'] = l
    channels['LAB_A'] = a
    channels['LAB_B'] = bb
    
    return channels

def preprocess_channel(gray_img, invert=False):
    """
    Pulisce il singolo canale per renderlo leggibile.
    Cleans the single channel to make it readable.
    """
    # Upscale per definire meglio i bordi / Upscale for better edges
    scale = 3
    img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Inversione opzionale / Optional inversion
    if invert:
        img = cv2.bitwise_not(img)
        
    # Aumento Contrasto Locale (CLAHE) / Local Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Binarizzazione (Otsu) / Thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Pulizia Morfologica / Morphological Cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Rimuovi rumore piccolissimo (opzionale) / Remove tiny noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    # Bordo bianco obbligatorio per EasyOCR / Mandatory white border
    binary = cv2.copyMakeBorder(binary, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    
    return binary

def solve_captcha(img_path_or_url, reader=None):
    """
    Solve a captcha from either a local file path, an http(s) URL or a data URI.

    Parameters:
      img_path_or_url: local filesystem path, http(s) URL or data:image/... base64 URI
      reader: optional easyocr.Reader — if not provided one will be created
    """
    tmp_file = None

    # Accept data URLs, HTTP(S) URLs, or local paths
    path = img_path_or_url
    is_url = False
    if isinstance(img_path_or_url, str):
        if img_path_or_url.startswith('data:image'):
            # data URI — decode into a temporary file
            header, b64 = img_path_or_url.split(',', 1)
            try:
                data = base64.b64decode(b64)
            except Exception:
                return "Error"
            fd, tmp_file = tempfile.mkstemp(suffix='.png')
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            path = tmp_file
        elif img_path_or_url.startswith('http'):
            # remote file — download to temp file
            is_url = True
            try:
                resp = requests.get(img_path_or_url, timeout=7)
                resp.raise_for_status()
            except Exception:
                return "Error"
            fd, tmp_file = tempfile.mkstemp(suffix='.png')
            with os.fdopen(fd, 'wb') as f:
                f.write(resp.content)
            path = tmp_file

    if not os.path.exists(path):
        if tmp_file:
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        return "Error"

    img = cv2.imread(path)
    if img is None: return "Error"
    
    filename = os.path.basename(path)
    print(f"Processing {filename}...", end=" ", flush=True)

    # Ottieni tutti i canali / Get all channels
    raw_channels = get_all_channels(img)
    
    best_text = "?????"
    best_score = -100
    best_method = ""
    
    # Prova ogni canale, sia normale che invertito
    # Try every channel, both normal and inverted
    for name, gray in raw_channels.items():
        for invert in [False, True]:
            variant_name = f"{name}_{'Inv' if invert else 'Norm'}"
            
            # Preprocessing
            processed = preprocess_channel(gray, invert=invert)
            
            # Debug: salva l'immagine processata (opzionale)
            # cv2.imwrite(f"debug_{filename}_{variant_name}.png", processed)
            
            # Ensure we have a reader object
            if reader is None:
                try:
                    reader = easyocr.Reader(['en'], gpu=False)
                except Exception:
                    reader = None

            try:
                if reader is None:
                    # No OCR reader available, skip this variant
                    continue
                # allowlist include SIA lettere CHE numeri
                # allowlist includes BOTH letters AND numbers
                results = reader.readtext(
                    processed, 
                    detail=1, 
                    allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
            except:
                continue
                
            text_accum = ""
            conf_accum = 0.0
            count = 0
            
            for res in results:
                t = res[1].strip()
                c = res[2] # Confidence score (0.0 - 1.0)
                if t:
                    text_accum += t
                    conf_accum += c
                    count += 1
            
            if count == 0: continue
            
            # Pulizia base (solo caratteri validi, niente replace forzati!)
            # Basic cleanup (valid chars only, NO forced replacements!)
            clean = ''.join(ch for ch in text_accum.lower() if ch.isalnum())
            
            # Calcolo Punteggio / Scoring
            avg_conf = conf_accum / count
            length = len(clean)
            
            score = avg_conf
            
            # Bonus lunghezza esatta (5 caratteri)
            # Exact length bonus
            if length == 5: score += 1.5 # Priorità massima
            elif length == 4 or length == 6: score += 0.4
            else: score -= 1.0 # Penalità forte per lunghezza errata
            
            # Bonus extra se il testo contiene numeri E lettere
            # Extra bonus if text has mixed numbers AND letters
            has_digit = any(c.isdigit() for c in clean)
            has_alpha = any(c.isalpha() for c in clean)
            if has_digit and has_alpha and length == 5:
                score += 0.2
            
            # Aggiorna il vincitore / Update winner
            if score > best_score:
                best_score = score
                best_text = clean
                best_method = variant_name
                
    # Formattazione finale / Final formatting
    if len(best_text) > 5: best_text = best_text[:5]
    while len(best_text) < 5: best_text += '?'
    
    print(f"-> {best_text} (via {best_method})")

    # cleanup temp file if we downloaded/decoded one
    if tmp_file:
        try:
            os.remove(tmp_file)
        except Exception:
            pass
    return best_text

async def attack_website(base_url: str, full_name: str, email: str, password: str,
                         gemini_api_key: str, record_video: bool = False,
                         video_dir: str = 'recordings', show_browser: bool = False):
    async with async_playwright() as p:
        # If you want to see the browser while the script runs, set show_browser=True
        # to launch in headful mode (headless=False). For CI or fast runs keep it headless.
        browser = await p.chromium.launch(headless=(not show_browser))
        # Optionally create a context that records video. Playwright saves
        # videos per page into the record_video_dir; creating a unique
        # subdirectory per run helps avoid collisions.
        context = None
        if record_video:
            import os
            import time
            subdir = f"run_attacker_1_{int(time.time())}"
            full_video_dir = os.path.join(video_dir, subdir)
            # Ensure path exists - Playwright will create files into it
            os.makedirs(full_video_dir, exist_ok=True)
            context = await browser.new_context(record_video_dir=full_video_dir)
            page = await context.new_page()
            print(f"Recording enabled — videos will be written to: {full_video_dir}")
        else:
            page = await browser.new_page()
        
        print("Navigating to the website...")
        await page.goto(base_url)
        
        gdpr_handled = False
        # Handling the GDPR consent modal if it appears
        try:
            await page.wait_for_selector('#gdprOverlay', state='visible', timeout=5000)
            print("GDPR consent modal detected, accepting...")
            await human_like_move_between(page, from_selector='body', to_selector='#gdprAccept', steps=10)
            await page.click('#gdprAccept')
            gdpr_handled = True
            await asyncio.sleep(1)  # wait a moment for modal to close
        except Exception:
            # Modal did not appear, continue
            pass


        # print("Filling form fields with human-like typing...")
        # move to the name field and click
        await human_like_move_between(page, from_selector='body', to_selector='#fullName', steps=10)
        await page.click('#fullName')
        await page.type('#fullName', full_name, delay=random.uniform(80, 150))
        
        # move from name to email before clicking (human-like)
        await human_like_move_between(page, from_selector='#fullName', to_selector='#email', steps=8)
        await page.click("#email")
        await page.type('#email', email, delay=random.uniform(80, 150))
        
        # move from email to password
        await human_like_move_between(page, from_selector='#email', to_selector='#password', steps=8)
        await page.click('#password')
        await page.type('#password', password, delay=random.uniform(80, 150))
        
        # move from password to confirmPassword
        await human_like_move_between(page, from_selector='#password', to_selector='#confirmPassword', steps=8)
        await page.click('#confirmPassword')
        await page.type('#confirmPassword', password, delay=random.uniform(80, 150))
        
        # # Ignore honeypot fields like company and website
        # await page.fill('#company', '')
        # await page.fill('#website', '')

        print("Handling CAPTCHA...")
        # Move the mouse around the form and then scroll like a human
        try:
            # move from the confirm password field toward the CAPTCHA area
            await human_like_move_between(page, from_selector="#confirmPassword", to_selector="#captchaContainer img", steps=16)
        except Exception:
            # selector might not exist yet; attempt a gentle move relative to the page
            try:
                await human_like_move_between(page, from_point=(400, 400), to_point=(400, 700), steps=12)
            except Exception:
                pass

        # perform a human-like scroll towards the bottom so CAPTCHA becomes visible
        await human_like_scroll(page, total_px=800, step_px=110)

        # Wait for CAPTCHA to load
        await page.wait_for_selector('#captchaContainer img', state='visible')
        
        # Get CAPTCHA image — capture bytes in-page to avoid a separate HTTP request
        img_locator = page.locator('#captchaContainer img')
        img_src = await img_locator.get_attribute('src')
        if not img_src:
            raise RuntimeError('CAPTCHA image src not found')

        # create a temporary file to hold the image bytes and pass to solve_captcha
        tmp_file = None
        path_for_solver = None
        # If src is a data URI, decode it directly
        if isinstance(img_src, str) and img_src.startswith('data:image'):
            header, b64 = img_src.split(',', 1)
            try:
                data = base64.b64decode(b64)
            except Exception:
                raise RuntimeError('Failed to decode data URI')
            fd, tmp_file = tempfile.mkstemp(suffix='.png')
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            path_for_solver = tmp_file
        else:
            # Prefer element screenshot to capture the image bytes directly from the current browser context
            try:
                img_bytes = await img_locator.screenshot()
            except Exception:
                # Fall back to constructing a URL if screenshot isn't possible
                if not img_src.startswith('http'):
                    img_src = base_url.rstrip('/') + '/' + img_src.lstrip('/')
                path_for_solver = img_src
            else:
                fd, tmp_file = tempfile.mkstemp(suffix='.png')
                with os.fdopen(fd, 'wb') as f:
                    f.write(img_bytes)
                path_for_solver = tmp_file

        # Create or reuse an easyocr reader and pass to the solver.
        # Creating the reader is somewhat expensive so we do it only when needed.
        try:
            reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            reader = None

        captcha_text = solve_captcha(path_for_solver, reader)

        # remove the local temp file we created (solve_captcha may also cleanup its own temp)
        if tmp_file:
            try:
                os.remove(tmp_file)
            except Exception:
                pass

        
        print(f"Extracted CAPTCHA text: {captcha_text}")
        # move to the drawing canvas and replay recorded strokes
        try:
            await human_like_move_between(page, from_selector='#captchaContainer img', to_selector='#drawingCanvas', steps=12)
        except Exception:
            # best-effort fallback: small pause before drawing
            await page.wait_for_timeout(250)

        await draw_captcha(page, '#drawingCanvas', captcha_text)
    
        print("Submitting the form...")

        # move to the submit button in a human-looking way then click
        try:
            await human_like_move_between(page, from_selector='#drawingCanvas', to_selector='#submitBtn', steps=12)
        except Exception:
            pass

        await asyncio.sleep(0.5 + random.uniform(0, 0.5))

        # Wait for PoW completion
        await page.wait_for_function("""
            () => {
                const btn = document.querySelector('#submitBtn');
                return !btn.disabled;
            }
        """)

        # Submit form
        await page.click('#submitBtn')
        
        # Wait for potential response
        await asyncio.sleep(2)
        print("Form submitted")

        # --- Final status check using #errorMessage and #successMessage ---
        error_text = ""
        success_text = ""
        captcha_id = None
        # Try to extract captcha ID from image src or page variable
        try:
            if img_src:
                # e.g. '/captcha/<id>.png' or full URL containing '/captcha/<id>.png'
                if '/captcha/' in img_src:
                    filename = img_src.rsplit('/', 1)[-1]
                    if filename and '.' in filename:
                        captcha_id = filename.rsplit('.', 1)[0]
        except Exception:
            captcha_id = None

        if not captcha_id:
            try:
                # currentCaptchaId is set by the page JS when loading a challenge
                val = await page.evaluate("typeof currentCaptchaId !== 'undefined' ? currentCaptchaId : null")
                if val:
                    captcha_id = val
            except Exception:
                captcha_id = None
        try:
            success_loc = page.locator('#successMessage')
            if await success_loc.count() > 0:
                t = await success_loc.text_content()
                success_text = (t or "").strip()
        except Exception:
            pass

        try:
            error_loc = page.locator('#errorMessage')
            if await error_loc.count() > 0:
                t = await error_loc.text_content()
                error_text = (t or "").strip()
        except Exception:
            pass

        if success_text:
            append_attack_result('attacker_1.py', True, success_text, captcha_id=captcha_id)
            print(f"Attack result: SUCCESS — {success_text}")
        elif error_text:
            append_attack_result('attacker_1.py', False, error_text, captcha_id=captcha_id)
            print(f"Attack result: FAILURE — {error_text}")
        else:
            append_attack_result('attacker_1.py', False, 'no status message', captcha_id=captcha_id)
            print("Attack result: UNKNOWN — no status messages found")

        await asyncio.sleep(2)
        
        # Close the context first if present to ensure Playwright finalizes
        # and saves any in-progress video files, then close the browser.
        try:
            if context:
                await context.close()
        finally:
            await browser.close()

if __name__ == '__main__':

    user = {
        "full_name": "John Doe",
        "email": "john.doe",
        "password": "StrongPass123!"
    }

    start_time = time()
    for i in range(3):
        print(f"\n=== Starting attack iteration {i+1} ===")
        asyncio.run(
            attack_website(
                'http://localhost:5174',
                user["full_name"] + f" {i}",
                user["email"] + f"_{i}@example.com",
                user["password"],
                'AIzaSyD0gWnYFRfUXQgC2zd-RWjguXoy4WjNzrE',
                record_video=True,
                video_dir='attack-recordings',
                show_browser=True,
            )
        )
        end_time = time()
        total_duration = end_time - start_time
        print(f"Attack iteration {i+1} took {total_duration:.2f} seconds")