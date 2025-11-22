import asyncio
import random
from time import time

from playwright.async_api import async_playwright
from utils import (draw_captcha, extract_text_from_image_using_llm,
                   human_like_move_between, human_like_scroll,
                   image_processing, load_symbol_mouse_data)


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
            subdir = f"run_{int(time.time())}"
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
        
        # Get CAPTCHA image URL
        img_src = await page.locator('#captchaContainer img').get_attribute('src')
        if not img_src:
            raise RuntimeError('CAPTCHA image src not found')
        if not img_src.startswith('http'):
            img_src = base_url.rstrip('/') + '/' + img_src.lstrip('/')

        # Extract text using Gemini via litellm
        img_base64 = image_processing(img_src)
        print("Extracting text from CAPTCHA image using LLM...")
        llm_result = extract_text_from_image_using_llm(
            img_base64, gemini_api_key, model="gemini/gemini-2.5-flash-lite"
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
        """)

        # Submit form
        await page.click('#submitBtn')
        
        # Wait for potential response
        await asyncio.sleep(2)
        print("Form submitted")

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
    for i in range(10):
        print(f"\n=== Starting attack iteration {i+1} ===")
        asyncio.run(
            attack_website(
                'http://localhost:8000',
                user["full_name"] + f" {i}",
                user["email"] + f"_{i}@example.com",
                user["password"],
                'AIzaSyD0gWnYFRfUXQgC2zd-RWjguXoy4WjNzrE',
                record_video=True,
                video_dir='attack-recordings',
                show_browser=False,
            )
        )
        end_time = time()
        total_duration = end_time - start_time
        print(f"Attack iteration {i+1} took {total_duration:.2f} seconds")