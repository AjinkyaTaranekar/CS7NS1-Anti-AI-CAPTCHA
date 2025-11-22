#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPTCHA solver - ADAPTIVE BRUTE FORCE

"""

import easyocr
import cv2
import numpy as np
import os
import sys
import logging

# Silenzia i log / Mute logs
logging.getLogger('easyocr.easyocr').setLevel(logging.ERROR)
os.environ["OMP_NUM_THREADS"] = "1"

def get_all_channels(img):
    """
    Estrae tutti i canali colore utili
    Extracts all useful color channels
    """
    channels = {}
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
    else:
        return {'Gray': img}

    channels['RGB_R'] = r
    channels['RGB_B'] = b
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    channels['HSV_S'] = s
    channels['HSV_V'] = v
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, bb = cv2.split(lab)
    channels['LAB_A'] = a
    channels['LAB_B'] = bb
    
    return channels

def preprocess_variant(gray_img, invert=False, morph_level=0):
    """
    Prepara l'immagine con diversi livelli di aggressività morfologica
    Prepares image with different levels of morphological aggression
    """
    # Upscale
    scale = 3
    img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    if invert:
        img = cv2.bitwise_not(img)
        
    # CLAHE (Contrasto locale)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Otsu Thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ADAPTIVE MORPHOLOGY
    if morph_level == 1:
        # Medio: Unisce puntini vicini
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif morph_level == 2:
        # Forte: Incicciottisce
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Bordo bianco
    binary = cv2.copyMakeBorder(binary, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    return binary

def solve_captcha(img_path, reader):
    if not os.path.exists(img_path): return "Error"
    img = cv2.imread(img_path)
    if img is None: return "Error"
    
    filename = os.path.basename(img_path)
    # print(f"Processing {filename}...", end=" ", flush=True)

    raw_channels = get_all_channels(img)
    
    best_text = "?????"
    best_score = -100
    best_method = ""
    
    # CICLO PRINCIPALE: Canali -> Inversione -> Morfologia
    # MAIN LOOP: Channels -> Inversion -> Morphology
    for name, gray in raw_channels.items():
        for invert in [False, True]:
            for morph in [0, 1, 2]: # Prova 3 livelli di "spessore"
                
                variant_name = f"{name}_{'Inv' if invert else 'Norm'}_M{morph}"
                
                processed = preprocess_variant(gray, invert=invert, morph_level=morph)
                
                try:
                    # Beamsearch ON per risolvere ambiguità
                    results = reader.readtext(
                        processed, 
                        detail=1, 
                        allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                        decoder='beamsearch',
                        beamWidth=5
                    )
                except:
                    continue
                    
                text_accum = ""
                conf_accum = 0.0
                count = 0
                
                for res in results:
                    t = res[1].strip()
                    c = res[2]
                    if t:
                        text_accum += t
                        conf_accum += c
                        count += 1
                
                if count == 0: continue
                
                # Pulizia base
                clean = ''.join(ch for ch in text_accum.lower() if ch.isalnum())
                
                # SCORING
                avg_conf = conf_accum / count
                length = len(clean)
                score = avg_conf
                
                # Bonus Lunghezza 5
                if length == 5: score += 2.0
                elif length == 4 or length == 6: score += 0.5
                else: score -= 1.5
                
                # Se contiene SIA numeri CHE lettere, è molto probabile sia corretto
                has_digit = any(c.isdigit() for c in clean)
                has_alpha = any(c.isalpha() for c in clean)
                if has_digit and has_alpha and length == 5:
                    score += 0.5
                
                # Se troviamo una stringa perfetta con confidenza alta, daje
                if score > best_score:
                    best_score = score
                    best_text = clean
                    best_method = variant_name

    # Formattazione
    if len(best_text) > 5: best_text = best_text[:5]
    while len(best_text) < 5: best_text += '?'
    
    # print(f"-> {best_text} (via {best_method})")
    return best_text

def main():
    print("--- CAPTCHA SOLVER ---")
    reader = easyocr.Reader(['en'], gpu=True) 
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path: ").strip().replace('"', '').replace("'", "")
        
    summary = []
    
    if os.path.isfile(path):
        res = solve_captcha(path, reader)
        summary.append((os.path.basename(path), res))
    elif os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg'))])
        print(f"Analisi di {len(files)} immagini...")
        for f in files:
            res = solve_captcha(os.path.join(path, f), reader)
            summary.append((f, res))
            
    print("\n" + "="*40)
    print(" FINAL REPORT v15 ")
    print("="*40)
    for f, t in summary:
        print(f"{f:<20} : {t}")
    print("="*40)

if __name__ == "__main__":
    main()