# Developed by Marta
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPTCHA solver - EasyOCR con Morphological Top-Hat
Focus: Rimozione aggressiva dello sfondo
Focus: Aggressive background removal
"""

import easyocr
import cv2
import numpy as np
import os
import sys
import logging

# Disabilita i log inutili
# Disable useless logs
logging.getLogger('easyocr.easyocr').setLevel(logging.ERROR)

def preprocess_image(img):

    # 1. Ridimensiona (Resize)
    # 1. Resize
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 2. Converti in HSV ed estrai solo la Luminosità
    # 2. Convert to HSV and extract only Brightness 
    # Questo ignora i colori e si concentra su quanto è "accesa" la luce
    # This ignores colors and focuses on light intensity
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2] # Canale V (0-255) / V Channel (0-255)

    # 3. Morphological Top-Hat
    # Sottrae lo sfondo locale
    # Subtracts local background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    tophat = cv2.morphologyEx(v_channel, cv2.MORPH_TOPHAT, kernel)

    # 4. Aumenta il contrasto 
    # 4. Increase contrast
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Pulizia del rumore
    # 5. Noise cleaning
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    # 6. Dilatazione leggera
    # 6. Light dilation
    clean = cv2.dilate(clean, kernel_clean, iterations=1)

    # 7. Aggiungi un bordo (Padding)
    # 7. Add border (Padding)
    final_img = cv2.copyMakeBorder(clean, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=0)

    # Inversione opzionale
    # Optional inversion
    final_img = cv2.bitwise_not(final_img)

    return final_img

def ocr_captcha(img_path, reader):
    if not os.path.exists(img_path):
        return "ERRORE: File non trovato" # ERROR: File not found

    img = cv2.imread(img_path)
    if img is None: return "Error Img"

    processed = preprocess_image(img)

    # SALVATAGGIO DEBUG
    # DEBUG SAVE
    # debug_name = "debug_" + os.path.basename(img_path)
    # cv2.imwrite(debug_name, processed)
    
    try:
        # Configurazione EasyOCR
        # EasyOCR Configuration
        results = reader.readtext(
            processed, 
            detail=0,
            decoder='beamsearch', 
            beamWidth=5,
            allowlist='abcdefghijklmnopqrstuvwxyz0123456789'
        )
    except Exception as e:
        return f"Error: {e}"

    text = "".join(results)
    text = ''.join(c for c in text.lower() if c.isalnum())

    # Post-processing specifico per errori comuni
    # Specific post-processing for common errors
    text = text.replace('0', 'o') # Spesso confusi / Often confused
    
    if len(text) > 5: text = text[:5]
    while len(text) < 5: text += '?'

    return text

def main():
    reader = easyocr.Reader(['en'], gpu=True) 

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path immagine/cartella: ").strip().replace('"', '').replace("'", "")

    if os.path.isfile(path):
        print(f"{os.path.basename(path)}: {ocr_captcha(path, reader)}")
    elif os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg'))]
        files.sort()
        for f in files:
            print(f"{f}: {ocr_captcha(os.path.join(path, f), reader)}")

if __name__ == "__main__":
    main()