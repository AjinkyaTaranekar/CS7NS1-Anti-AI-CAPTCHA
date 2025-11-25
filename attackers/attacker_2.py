import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import os
import argparse

# ==========================================
# CONFIGURATION (Must match training!) / CONFIGURAZIONE
# ==========================================
IMG_WIDTH = 128
IMG_HEIGHT = 64
MAX_LENGTH = 5

# Vocabolario identico al training
# Vocabulary identical to training
characters = sorted(list("abcdefghijklmnopqrstuvwxyz0123456789"))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_batch_predictions(pred):
    """
    Decodifica l'output della rete in testo.
    Decodes network output into text.
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Greedy Search per trovare i caratteri migliori
    # Greedy Search to find best characters
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
    
    output_text = []
    for res in results:
        # Converte numeri in stringa
        # Convert numbers to string
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def preprocess_image(img_path):
    """
    Prepara l'immagine (RGB, Resize, Transpose).
    Prepares the image (RGB, Resize, Transpose).
    """
    try:
        img = tf.io.read_file(img_path)
        # IMPORTANTE: channels=3 perché abbiamo addestrato in RGB
        # IMPORTANT: channels=3 because we trained in RGB
        img = tf.io.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    # Gestione argomenti / Argument parsing
    parser = argparse.ArgumentParser(description="Keras Captcha Solver with Accuracy")
    parser.add_argument("target", help="Path to image or folder")
    parser.add_argument("--model", default="best_model.h5", help="Path to .h5 model file")
    
    args = parser.parse_args()
    path = args.target.replace('"', '').replace("'", "")
    model_path = args.model

    # Controllo Modello / Model Check
    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found!")
        print("Run 'train_keras.py' first.")
        return

    print(f"--- CAPTCHA SOLVER ---\nLoading model: {model_path} ...")
    model = keras.models.load_model(model_path, compile=False)
    
    images = []
    filenames = []
    true_labels = []
    
    # Raccogli immagini (File singolo o Cartella)
    # Collect images (Single file or Folder)
    if os.path.isfile(path):
        file_list = [path]
    elif os.path.isdir(path):
        file_list = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        print("Invalid path.")
        return

    print(f"Processing {len(file_list)} images...")

    # Pre-caricamento in memoria
    # Pre-loading into memory
    for file_path in file_list:
        img = preprocess_image(file_path)
        if img is not None:
            images.append(img)
            fname = os.path.basename(file_path)
            filenames.append(fname)
            
            # Tentativo di estrarre la label reale dal nome file
            # Attempt to extract real label from filename
            if "." in fname:
                true_label = fname.split(".")[0]
            true_labels.append(true_label)

    if not images:
        print("No valid images found.")
        return

    # Predizione in Batch
    # Batch Prediction
    batch_images = tf.stack(images)
    preds = model.predict(batch_images, verbose=1)
    decoded_texts = decode_batch_predictions(preds)
    
    # Calcolo Statistiche / Calculate Stats
    correct_count = 0
    total_count = len(decoded_texts)

    print("\n" + "="*60)
    print(f" {'FILENAME':<25} | {'PREDICTED':<10} | {'TRUE':<10} | {'RESULT'}")
    print("="*60)

    for i in range(total_count):
        pred = decoded_texts[i]
        true = true_labels[i]
        fname = filenames[i]
        
        # Verifica correttezza / Check correctness
        # (Ignora case sensitivity per sicurezza)
        is_correct = (pred.lower() == true.lower())
        
        if is_correct:
            correct_count += 1
            status = "OK"
        else:
            status = "FAIL"

        # Stampa riga risultato / Print result row
        # Mostriamo true label solo se sembra valida (lunghezza 5), altrimenti mettiamo "?"
        # Show true label only if it looks valid, otherwise "?"
        display_true = true if len(true) == 5 else "?"
        
        # Se la label vera non è nota (es. file test.png), non contare come errore nel report visivo
        # If true label is unknown, don't mark visually as fail (but logic below handles stats)
        if len(true) != 5: status = "❓ UNKNOWN"

        print(f" {fname:<25} | {pred:<10} | {display_true:<10} | {status}")

    # --- REPORT FINALE / FINAL REPORT ---
    accuracy = (correct_count / total_count) * 100
    
    print("="*60)
    print(f" TOTAL IMAGES: {total_count}")
    print(f" CORRECT:      {correct_count}")
    print(f" ACCURACY:     {accuracy:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()