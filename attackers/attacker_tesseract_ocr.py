#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAPTCHA solver ottimizzato con Tesseract
Versione definitiva: legge l'intera immagine e limita a 5 caratteri
"""

import pytesseract
import cv2
import numpy as np
import os
import sys

def preprocess_image(img):
    """
    Grayscale, equalizzazione, ridimensionamento e thresholding
    Preprocessing: grayscale, equalization, resize and threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # Ridimensiona 3x
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # Threshold adattivo
    processed = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return processed

def ocr_captcha(img_path):
    """
    OCR Tesseract diretto sull'intera immagine
    Direct Tesseract OCR on the whole image
    """
    if not os.path.exists(img_path):
        return f"ERRORE: {img_path} non trovato"

    img = cv2.imread(img_path)
    processed = preprocess_image(img)

    # OCR: whitelist e PSM 7 (single line)
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz0123456789"
    text = pytesseract.image_to_string(processed, config=config)
    text = ''.join(c for c in text.lower() if c.isalnum())

    # Limita a 5 caratteri
    if len(text) > 5:
        text = text[:5]
    # Se più corto, completa con '?'
    while len(text) < 5:
        text += '?'

    return text

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            print(ocr_captcha(path))
        elif os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif')):
                    full_path = os.path.join(path, f)
                    print(f"{f}: {ocr_captcha(full_path)}")
        else:
            print("ERRORE: path non valido / Invalid path")
    else:
        path = input("Inserisci il path dell'immagine o cartella: ").strip()
        if os.path.isfile(path):
            print(ocr_captcha(path))
        elif os.path.isdir(path):
            for f in os.listdir(path):
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.gif')):
                    full_path = os.path.join(path, f)
                    print(f"{f}: {ocr_captcha(full_path)}")
        else:
            print("ERRORE: path non valido / Invalid path")

if __name__ == "__main__":
    main()
