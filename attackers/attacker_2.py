import argparse
import asyncio
import base64
import os
import random
import sys
import tempfile
from time import time

import numpy as np
import tensorflow as tf
from playwright.async_api import async_playwright
from tensorflow import keras
from tensorflow.keras import layers, models
from utils import (append_attack_result, draw_captcha, human_like_move_between,
                   human_like_scroll)

# ==========================================
# CONFIGURATION (MUST MATCH TRAIN.PY!)
# ==========================================
IMG_WIDTH = 420      
IMG_HEIGHT = 220      
MAX_LENGTH = 5       

# Vocabolario identico al training
# Vocabulary identical to training
characters = sorted(list("abcdefghijklmnopqrstuvwxyz0123456789"))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def build_prediction_model_structure():
    """
    Ricostruisce manualmente l'architettura per caricare i pesi.
    """

    input_img = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name="image", dtype="float32")

    # Same arch as training
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="Conv3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Reshape
    new_shape = ((IMG_WIDTH // 8), (IMG_HEIGHT // 8) * 256)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3)(x)

    # RNN
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3))(x)

    # Output
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)

    # Creates empty model
    model = models.Model(inputs=input_img, outputs=x, name="ocr_prediction")
    return model

def decode_batch_predictions(pred):
    """
    Decodifica l'output della rete neurale in testo.
    Decodes neural network output into text.
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Greedy Search per trovare i caratteri migliori
    # Greedy Search to find best characters
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
    
    output_text = []
    for res in results:
        # Converte numeri in stringa e unisce
        # Converts numbers to string and joins
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        res = res.replace("[UNK]", " ")
        output_text.append(res)
    return output_text

def preprocess_image(img_path):
    """
    Prepara l'immagine esattamente come nel training.
    Prepares the image exactly as in training.
    """
    try:
        img = tf.io.read_file(img_path)
        
        # IMPORTANTE: RGB (3 canali)
        # IMPORTANT: RGB (3 channels)
        img = tf.io.decode_png(img, channels=3) 
        
        img = tf.image.convert_image_dtype(img, tf.float32)
        
        # Resize alle dimensioni del training (420x220)
        # Resize to training dimensions (420x220)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        
        # Trasposizione (Time, Width, Channels)
        # Transpose (Time, Width, Channels)
        img = tf.transpose(img, perm=[1, 0, 2])
        
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("target", help="Path to image or folder")
    parser.add_argument("--model", default="best_model.h5", help="Path to model file")
    args = parser.parse_args()
    
    # path = args.target.replace('"', '').replace("'", "")
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found!")
        return

    print(f"Loading model: {model_path} ...")
    model = build_prediction_model_structure()
    model.load_weights(model_path)

    
    images = []
    filenames = []
    true_labels = []

    # Rilevamento file o cartella
    # Detect single file or folder
    file_list = []
    if os.path.isfile(path):
        file_list = [path]
    elif os.path.isdir(path):
        file_list = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(('.png', '.jpg'))]
    else:
        print("Invalid path.")
        return

    print(f"Processing {len(file_list)} images...")

    for p in file_list:
        img = preprocess_image(p)
        if img is not None:
            images.append(img)
            fname = os.path.basename(p)
            filenames.append(fname)
            
            # Prova ad estrarre la soluzione dal nome file
            # Try to extract label from filename
            if "." in fname:
                label = fname.split(".")[0]
            true_labels.append(label)

    if not images:
        print("No images found.")
        return

    # Predizione (Batch Prediction)
    batch_images = tf.stack(images)
    preds = model.predict(batch_images, verbose=1)
    decoded_texts = decode_batch_predictions(preds)
    
    # Calcolo Risultati / Results Calculation
    correct = 0
    total = len(decoded_texts)

    print("\n" + "="*65)
    print(f" {'FILENAME':<25} | {'PRED':<10} | {'TRUE':<10} | {'STATUS'}")
    print("="*65)

    for i in range(total):
        pred = decoded_texts[i]
        true = true_labels[i]
        
        # Controllo se è corretto (case insensitive)
        # Check correctness (case insensitive)
        is_correct = (pred.lower() == true.lower())
        if is_correct:
            correct += 1
            status = "OK"
        else:
            status = "FAIL"
            
        # Se la label vera non ha lunghezza 5 (es: image.png), non contiamo come errore visivo
        # If true label is not len 5 (e.g., image.png), don't mark visually as fail
        if len(true) != 5: 
            status = "???"
            # Non contiamo questo file nelle statistiche di accuratezza
            # Don't count this file in accuracy stats if label is unknown
            total -= 1 
            correct -= (1 if is_correct else 0) # Annulla incremento se era giusto per caso

        print(f" {filenames[i]:<25} | {pred:<10} | {true:<10} | {status}")

    print("="*65)
    """


async def attack_website(base_url: str, full_name: str, email: str, password: str,
                         model_path: str, record_video: bool = False,
                         video_dir: str = 'recordings', show_browser: bool = False):
    """Playwright attack flow that uses the Keras model for captcha solving.

    This mostly mirrors the flow in attacker_1.py but uses the model inference
    functions already present in this file (preprocess_image + decode_batch_predictions).
    """

    # load model for inference (compile=False recommended for Keras saved models)
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    print(f"Loading model for attack: {model_path}")
    model = build_prediction_model_structure()
    model.load_weights(model_path)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=(not show_browser))
        context = None
        if record_video:
            import time as _time
            subdir = f"run_{int(_time.time())}"
            full_video_dir = os.path.join(video_dir, subdir)
            os.makedirs(full_video_dir, exist_ok=True)
            context = await browser.new_context(record_video_dir=full_video_dir)
            page = await context.new_page()
            print(f"Recording enabled — videos will be written to: {full_video_dir}")
        else:
            page = await browser.new_page()

        await page.goto(base_url)

        # Basic human-like typing and navigation copied from attacker_1 style
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

        # Move toward captcha and make it visible
        try:
            await human_like_move_between(page, from_selector='#confirmPassword', to_selector='#captchaContainer img', steps=16)
        except Exception:
            try:
                await human_like_move_between(page, from_point=(400, 400), to_point=(400, 700), steps=12)
            except Exception:
                pass

        await human_like_scroll(page, total_px=800, step_px=110)

        # Wait for captcha image to appear
        await page.wait_for_selector('#captchaContainer img', state='visible')
        img_locator = page.locator('#captchaContainer img')
        img_src = await img_locator.get_attribute('src')
        if not img_src:
            raise RuntimeError('CAPTCHA image src not found')

        tmp_file = None
        path_for_solver = None

        # Prefer a direct screenshot of the element — this avoids remote fetch permissions
        try:
            img_bytes = await img_locator.screenshot()
        except Exception:
            img_bytes = None

        if img_bytes:
            fd, tmp_file = tempfile.mkstemp(suffix='.png')
            with os.fdopen(fd, 'wb') as f:
                f.write(img_bytes)
            path_for_solver = tmp_file
        elif isinstance(img_src, str) and img_src.startswith('data:image'):
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
            # fallback to constructing a full URL (if src is a relative path)
            if not img_src.startswith('http'):
                img_src = base_url.rstrip('/') + '/' + img_src.lstrip('/')
            path_for_solver = img_src

        # If we have a file path for the solver, preprocess and predict
        if not path_for_solver:
            raise RuntimeError('No captcha image available to solve')

        # The preprocess_image function returns a transposed tensor. We must ensure
        # it behaves properly for a single file path. Use the existing functions.
        img_tensor = preprocess_image(path_for_solver)
        if img_tensor is None:
            captcha_text = "?????"
        else:
            batch = tf.expand_dims(img_tensor, axis=0)
            preds = model.predict(batch, verbose=0)
            decoded = decode_batch_predictions(preds)
            captcha_text = decoded[0] if decoded else "?????"

        # cleanup tempfile if created
        if tmp_file:
            try:
                os.remove(tmp_file)
            except Exception:
                pass

        print(f"Extracted CAPTCHA text: {captcha_text}")

        try:
            await human_like_move_between(page, from_selector='#captchaContainer img', to_selector='#drawingCanvas', steps=12)
        except Exception:
            await page.wait_for_timeout(250)

        await draw_captcha(page, '#drawingCanvas', captcha_text)

        try:
            await human_like_move_between(page, from_selector='#drawingCanvas', to_selector='#submitBtn', steps=12)
        except Exception:
            pass

        await asyncio.sleep(0.5 + random.uniform(0, 0.5))

        await page.wait_for_function("""
            () => {
                const btn = document.querySelector('#submitBtn');
                return !btn.disabled;
            }
        """)

        await page.click('#submitBtn')
        await asyncio.sleep(2)

        # status extraction
        success_text = ''
        error_text = ''
        captcha_id = None
        try:
            img_src = img_src or ''
            if '/captcha/' in img_src:
                filename = img_src.rsplit('/', 1)[-1]
                if filename and '.' in filename:
                    captcha_id = filename.rsplit('.', 1)[0]
        except Exception:
            captcha_id = None

        try:
            val = await page.evaluate("typeof currentCaptchaId !== 'undefined' ? currentCaptchaId : null")
            if val:
                captcha_id = val
        except Exception:
            pass

        try:
            success_loc = page.locator('#successMessage')
            if await success_loc.count() > 0:
                t = await success_loc.text_content()
                success_text = (t or '').strip()
        except Exception:
            pass

        try:
            error_loc = page.locator('#errorMessage')
            if await error_loc.count() > 0:
                t = await error_loc.text_content()
                error_text = (t or '').strip()
        except Exception:
            pass

        if success_text:
            append_attack_result('attacker_2.py', True, success_text, captcha_id=captcha_id)
            print(f"Attack result: SUCCESS — {success_text}")
        elif error_text:
            append_attack_result('attacker_2.py', False, error_text, captcha_id=captcha_id)
            print(f"Attack result: FAILURE — {error_text}")
        else:
            append_attack_result('attacker_2.py', False, 'no status message', captcha_id=captcha_id)
            print('Attack result: UNKNOWN — no status messages found')

        try:
            if context:
                await context.close()
        finally:
            await browser.close()


if __name__ == "__main__":
    # Provide similar CLI convenience to attacker_1 for quick checks
    user = {
        'full_name': 'John Doe',
        'email': 'john.doe',
        'password': 'StrongPass123!'
    }

    start_time = time()
    # run a few attack iterations to exercise the flow
    for i in range(3):
        print(f"\n=== Starting attack iteration {i+1} ===")
        asyncio.run(
            attack_website(
                'http://localhost:5174',
                user['full_name'] + f" {i}",
                user['email'] + f"_{i}@example.com",
                user['password'],
                'best_model.h5',
                record_video=True,
                video_dir='attack-recordings',
                show_browser=False,
            )
        )
        end_time = time()
        total_duration = end_time - start_time
        print(f"Attack iteration {i+1} took {total_duration:.2f} seconds")
    print("="*65)

if __name__ == "__main__":
    main()