#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR CAPTCHA con EasyOCR
Funziona con singoli file o cartelle di immagini
CAPTCHA length fixed to 5 characters
"""

import easyocr
import cv2
import os
import sys

def ocr_easyocr(image_path):
    """
    OCR con EasyOCR, output limitato a 5 caratteri
    EasyOCR OCR, output limited to 5 characters
    """
    if not os.path.exists(image_path):
        return f"ERRORE: {image_path} non trovato / File not found"
    
    # Inizializza il reader EasyOCR (lingua inglese, con GPU)
    # Initialize EasyOCR reader (English language, yes GPU)
    reader = easyocr.Reader(['en'], gpu=True)

    # Carica immagine con OpenCV
    # Load image with OpenCV
    img = cv2.imread(image_path)
    
    # Riconoscimento testo: solo lettere minuscole e numeri
    # Text recognition: only lowercase letters and numbers
    result = reader.readtext(img, detail=0, allowlist='abcdefghijklmnopqrstuvwxyz0123456789')
    
    # Combina risultati in stringa
    # Combine results into a string
    text = ''.join(result).lower()
    
    # Limita a 5 caratteri, completa con '?' se più corto
    # Limit to 5 characters, pad with '?' if shorter
    if len(text) > 5:
        text = text[:5]
    while len(text) < 5:
        text += '?'

    return text

def process_folder(folder_path):
    """
    Processa tutte le immagini in una cartella
    Processes all images in a folder
    """
    if not os.path.exists(folder_path):
        print(f"ERRORE: cartella {folder_path} non trovata / Folder not found")
        return
    
    # Estensioni immagini supportate
    # Supported image extensions
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
    
    if not files:
        print(f"Nessuna immagine trovata in {folder_path} / No images found in folder")
        return
    
    for filename in sorted(files):
        filepath = os.path.join(folder_path, filename)
        text = ocr_easyocr(filepath)
        print(f"{filename}: {text}")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            print(ocr_easyocr(path))
        elif os.path.isdir(path):
            process_folder(path)
        else:
            print("ERRORE: path non valido / Invalid path")
    else:
        path = input("Inserisci il path dell'immagine o cartella: ").strip()
        if os.path.isfile(path):
            print(ocr_easyocr(path))
        elif os.path.isdir(path):
            process_folder(path)
        else:
            print("ERRORE: path non valido / Invalid path")

if __name__ == "__main__":
    main()
