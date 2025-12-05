# Developed by Marta
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPTCHA solver v12 - PURE BRUTE FORCE 
"""

import easyocr
import cv2
import numpy as np
import os
import sys
import logging

# Silenzia i log inutili / Mute logs
logging.getLogger('easyocr.easyocr').setLevel(logging.ERROR)
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

def solve_captcha(img_path, reader):
    if not os.path.exists(img_path): return "Error"
    img = cv2.imread(img_path)
    if img is None: return "Error"
    
    filename = os.path.basename(img_path)
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
            
            try:
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
    return best_text

def main():
    print("--- CAPTCHA SOLVER ---")
    reader = easyocr.Reader(['en'], gpu=False) 
    
    path = ""
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Inserisci path: ").strip().replace('"', '').replace("'", "")
        
    summary = []
    
    if os.path.isfile(path):
        res = solve_captcha(path, reader)
        summary.append((os.path.basename(path), res))
    elif os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg'))])
        for f in files:
            res = solve_captcha(os.path.join(path, f), reader)
            summary.append((f, res))
            
    print("\n" + "="*40)
    print(" FINAL RESULTS")
    print("="*40)
    for f, t in summary:
        print(f"{f:<20} : {t}")

if __name__ == "__main__":
    main()