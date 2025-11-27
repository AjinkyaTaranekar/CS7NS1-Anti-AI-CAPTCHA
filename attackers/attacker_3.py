import asyncio
import os
import random
from time import time

from playwright.async_api import async_playwright
from utils import (append_attack_result, draw_captcha,
                   extract_text_from_image_using_llm, human_like_move_between,
                   human_like_scroll, image_processing, image_processing_bytes)


async def attack_website(base_url: str, full_name: str, email: str, password: str,
                         api_key: str, llm_model: str, record_video: bool = False,
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
            subdir = f"run_attacker_3_{int(time.time())}"
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
        
        # Get CAPTCHA image and avoid making a separate HTTP request.
        # Some captcha images are protected or require session cookies — a direct
        # requests.get(image_url) can return 403. Instead we retrieve the image
        # bytes directly from the browser context using Playwright.
        img_locator = page.locator('#captchaContainer img')
        img_src = await img_locator.get_attribute('src')
        if not img_src:
            raise RuntimeError('CAPTCHA image src not found')

        if isinstance(img_src, str) and img_src.startswith('data:'):
            # If the img src is a data URL, decode it.
            import base64 as _b64
            img_base64_payload = img_src.split(',', 1)[1]
            img_bytes = _b64.b64decode(img_base64_payload)
        else:
            # Element screenshot will capture the rendered image bytes in the current
            # browser session — this bypasses cross-origin/server protection.
            img_bytes = await img_locator.screenshot()

        # Extract text using Gemini via litellm from preprocessed bytes
        img_base64 = image_processing_bytes(img_bytes)
        print("Extracting text from CAPTCHA image using LLM...")
        llm_result = extract_text_from_image_using_llm(
            img_base64, api_key, llm_model
        )

        # extract_text_from_image_using_llm now returns (captcha_text, confidence)
        if isinstance(llm_result, dict):
            captcha_text = llm_result.get('captcha', '')
            confidence = llm_result.get('confidence', 0.0)
        else:
            captcha_text = llm_result
            confidence = 0.0

        print(f"Extracted CAPTCHA text: {captcha_text} (confidence={confidence})")
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
        """, timeout=60000)

        # Submit form
        await page.click('#submitBtn')
        
        # Wait for potential response
        await asyncio.sleep(2)
        print("Form submitted")

        # --- Final status check using #errorMessage and #successMessage ---
        error_text = ""
        success_text = ""
        captcha_id = None
        # Try to extract captcha ID from image src (if available) or page JS
        try:
            if isinstance(img_src, str) and '/captcha/' in img_src:
                filename = img_src.rsplit('/', 1)[-1]
                if filename and '.' in filename:
                    captcha_id = filename.rsplit('.', 1)[0]
        except Exception:
            captcha_id = None

        if not captcha_id:
            try:
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
            append_attack_result('attacker_3.py', True, success_text, captcha_id=captcha_id)
            print(f"Attack result: SUCCESS — {success_text}")
        elif error_text:
            append_attack_result('attacker_3.py', False, error_text, captcha_id=captcha_id)
            print(f"Attack result: FAILURE — {error_text}")
        else:
            append_attack_result('attacker_3.py', False, 'no status message', captcha_id=captcha_id)
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

    url = 'http://localhost:5174'
    user = {
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "password": "StrongPass123!"
    }
    api_key = os.getenv('API_KEY', 'YOUR_API_KEY_HERE')
    llm_model = os.getenv('LLM_MODEL', 'MODEL_NAME_HERE')


    for i in range(3):
        print("Starting attack for the iteration:", i + 1)

        start_time = time()
        asyncio.run(
            attack_website(
                url,
                user["full_name"],
                user["email"],
                user["password"],
                api_key,
                llm_model,
                record_video=True,
                video_dir='attack-recordings',
                show_browser=True,
            )
        )
        end_time = time()
        total_duration = end_time - start_time
        print(f"Attack took {total_duration:.2f} seconds for iteration {i + 1}\n")