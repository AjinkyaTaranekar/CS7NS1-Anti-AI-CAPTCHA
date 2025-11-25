import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import os
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Path to image or folder")
    parser.add_argument("--model", default="best_model.h5", help="Path to model file")
    args = parser.parse_args()
    
    path = args.target.replace('"', '').replace("'", "")
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found!")
        return

    print(f"Loading model: {model_path} ...")
    # compile=False è fondamentale per l'inferenza
    # compile=False is critical for inference
    model = keras.models.load_model(model_path, compile=False)
    
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
    if total > 0:
        acc = (correct / total) * 100
        print(f" FINAL ACCURACY: {acc:.2f}% ({correct}/{total} correct)")
    else:
        print(" No labeled data found to calculate accuracy.")
    print("="*65)

if __name__ == "__main__":
    main()