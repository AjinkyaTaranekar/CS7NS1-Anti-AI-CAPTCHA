# Developed by Marta
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import asyncio
import os
import random
import sys
import tempfile
import time
from time import time as get_time

from playwright.async_api import async_playwright
from ultralytics import YOLO
from utils import (append_attack_result, draw_captcha, human_like_move_between,
                   human_like_scroll)


# ==========================================
# INTERNAL CLASS: YOLO SOLVER
# ==========================================
class YoloCaptchaSolver:
    """
    Wrapper class to handle YOLOv8 inference for variable length CAPTCHAs.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading YOLO model from: {model_path}")
        # Load the model (verbose=False to keep console clean)
        self.model = YOLO(model_path)

    def solve(self, image_path):
        """
        Detects characters and sorts them left-to-right.
        """
        # Run inference
        results = self.model(image_path, verbose=False)
        
        detected_items = []

        # Process results
        for r in results:
            for box in r.boxes:
                # Box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # left edge for sorting order
                x_pos = x1
                
                # Class ID and Name
                cls_id = int(box.cls[0])
                char_name = r.names[cls_id]
                
                # Confidence score
                conf = float(box.conf[0])
                
                # Store candidate
                detected_items.append({'x': x_pos, 'char': char_name, 'conf': conf})

        # SORT LEFT TO RIGHT
        detected_items.sort(key=lambda k: k['x'])
        
        # Join the result directly
        # Unisce direttamente i caratteri trovati
        final_chars = [item['char'] for item in detected_items]
        text = "".join(final_chars)
        
        return text

# ==========================================
# ATTACK LOGIC
# ==========================================
async def attack_website(base_url: str, full_name: str, email: str, password: str,
                         model_path: str, record_video: bool = False,
                         video_dir: str = 'recordings', show_browser: bool = False):
    
    # Initialize the internal solver
    try:
        solver = YoloCaptchaSolver(model_path)
    except Exception as e:
        print(f"Error initializing solver: {e}")
        return

    async with async_playwright() as p:
        # Browser setup
        browser = await p.chromium.launch(headless=(not show_browser))
        context = None
        
        if record_video:
            subdir = f"run_attacker_4_{int(get_time())}"
            full_video_dir = os.path.join(video_dir, subdir)
            os.makedirs(full_video_dir, exist_ok=True)
            context = await browser.new_context(record_video_dir=full_video_dir)
            page = await context.new_page()
            print(f"Recording enabled: {full_video_dir}")
        else:
            page = await browser.new_page()

        print(f"Navigating to {base_url}...")
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

        await human_like_move_between(page, from_selector='#fullName', to_selector='#email', steps=8)
        await page.click('#email')
        await page.type('#email', email, delay=random.uniform(80, 150))

        await human_like_move_between(page, from_selector='#email', to_selector='#password', steps=8)
        await page.click('#password')
        await page.type('#password', password, delay=random.uniform(80, 150))

        await human_like_move_between(page, from_selector='#password', to_selector='#confirmPassword', steps=8)
        await page.click('#confirmPassword')
        await page.type('#confirmPassword', password, delay=random.uniform(80, 150))

        # Locate Captcha
        await human_like_scroll(page, total_px=800, step_px=110)
        await page.wait_for_selector('#captchaContainer img', state='visible')
        img_locator = page.locator('#captchaContainer img')

        # --- Captcha Extraction ---
        fd, tmp_file = tempfile.mkstemp(suffix='.png')
        os.close(fd) 
        
        await img_locator.screenshot(path=tmp_file)

        # --- Solving ---
        # print("Solving CAPTCHA with YOLO...")
        captcha_text = solver.solve(tmp_file)
        print(f"Extracted CAPTCHA text: {captcha_text}")

        # --- Drawing and Submission ---
        try:
            await human_like_move_between(page, from_selector='#captchaContainer img', to_selector='#drawingCanvas', steps=12)
        except:
            await page.wait_for_timeout(250)

        await draw_captcha(page, '#drawingCanvas', captcha_text)

        try:
            await human_like_move_between(page, from_selector='#drawingCanvas', to_selector='#submitBtn', steps=12)
        except: pass

        await asyncio.sleep(0.5 + random.uniform(0, 0.5))

        # Wait for button
        await page.wait_for_function("""
            () => {
                const btn = document.querySelector('#submitBtn');
                return !btn.disabled;
            }
        """)

        await page.click('#submitBtn')
        await asyncio.sleep(2)

        # --- Result Verification ---
        success_text = ''
        error_text = ''
        captcha_id = None 
        
        # Tentativo di estrarre ID
        try:
            img_src = await img_locator.get_attribute('src')
            if img_src and '/captcha/' in img_src:
                filename = img_src.rsplit('/', 1)[-1]
                if filename and '.' in filename:
                    captcha_id = filename.rsplit('.', 1)[0]
        except: pass

        try:
            if await page.locator('#successMessage').count() > 0:
                success_text = await page.locator('#successMessage').text_content()
                success_text = (success_text or '').strip()
        except: pass

        try:
            if await page.locator('#errorMessage').count() > 0:
                error_text = await page.locator('#errorMessage').text_content()
                error_text = (error_text or '').strip()
        except: pass

        if success_text:
            append_attack_result('attacker_yolo.py', True, success_text, captcha_id=captcha_id)
            print(f"Attack result: SUCCESS — {success_text}")
        elif error_text:
            append_attack_result('attacker_yolo.py', False, error_text, captcha_id=captcha_id)
            print(f"Attack result: FAILURE — {error_text}")
        else:
            append_attack_result('attacker_yolo.py', False, 'no status message', captcha_id=captcha_id)
            print('Attack result: UNKNOWN — no status messages found')

        try:
            if context:
                await context.close()
        finally:
            await browser.close()

if __name__ == "__main__":
    # Configuration
    user = {
        'full_name': 'John Doe',
        'email': 'john.doe',
        'password': 'StrongPass123!'
    }
    
    # Modello YOLO
    MODEL_FILE = 'yolo_model.pt'

    # Loop di attacco
    start_time_total = get_time()
    
    for i in range(3):
        print(f"\n=== Starting attack iteration {i+1} ===")
        
        iter_start = get_time()
        
        asyncio.run(
            attack_website(
                'http://localhost:5174',
                user['full_name'] + f" {i}",        
                user['email'] + f"_{i}@example.com", 
                user['password'],
                MODEL_FILE,
                record_video=True,
                video_dir='attack-recordings',
                show_browser=True,
            )
        )
        
        iter_duration = get_time() - iter_start
        print(f"Attack iteration {i+1} took {iter_duration:.2f} seconds")
        print("="*65)

    total_duration = get_time() - start_time_total
    print(f"\nTotal execution time: {total_duration:.2f} seconds")