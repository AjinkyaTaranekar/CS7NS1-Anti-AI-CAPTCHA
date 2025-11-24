import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import os
import argparse

# ================================================
# CONFIGURATION (Must match training!!!!!!!!!!!!!)
# ================================================
IMG_WIDTH = 420
IMG_HEIGHT = 220
MAX_LENGTH = 5

# Vocabolario / Vocabulary
characters = sorted(list("abcdefghijklmnopqrstuvwxyz0123456789"))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def preprocess_image(img_path):
    try:
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    # Gestione argomenti linea di comando
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Keras Captcha Solver")
    parser.add_argument("target", help="Path to image or folder")
    parser.add_argument("--model", default="best_model.h5", help="Path to .h5 model file (default: best_model.h5)")
    
    args = parser.parse_args()
    
    path = args.target.replace('"', '').replace("'", "")
    model_path = args.model

    if not os.path.exists(model_path):
        print(f"ERROR: Model '{model_path}' not found!")
        print("Check the path or run 'train_keras.py' first.")
        return

    print(f"Loading model: {model_path} ...")
    model = keras.models.load_model(model_path, compile=False)
    
    images = []
    filenames = []
    
    # Raccogli immagini
    if os.path.isfile(path):
        img = preprocess_image(path)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(path))
    elif os.path.isdir(path):
        print(f"Scanning folder: {path}")
        for f in sorted(os.listdir(path)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = preprocess_image(os.path.join(path, f))
                if img is not None:
                    images.append(img)
                    filenames.append(f)
    
    if not images:
        print("No valid images found.")
        return

    print(f"Processing {len(images)} images...")

    batch_images = tf.stack(images)
    preds = model.predict(batch_images, verbose=1)
    decoded_texts = decode_batch_predictions(preds)
    
    print("\n" + "="*40)
    print(f" RESULTS (Model: {os.path.basename(model_path)})")
    print("="*40)
    for i, text in enumerate(decoded_texts):
        print(f"{filenames[i]:<25} : {text}")
    print("="*40)

if __name__ == "__main__":
    main()