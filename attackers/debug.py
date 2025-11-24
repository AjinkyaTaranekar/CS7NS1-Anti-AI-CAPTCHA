import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# CONFIGURAZIONE (Deve essere IDENTICA al training)
IMG_WIDTH = 128
IMG_HEIGHT = 64
DATA_DIR = "./dataset"

# Vocabolario
characters = sorted(list("abcdefghijklmnopqrstuvwxyz0123456789"))
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def encode_single_sample(img_path, label):
    # Stessa identica pipeline del training
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3) # RGB
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2]) # La rotazione per la RNN
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}

def main():
    # Carica immagini
    images = sorted(list(map(str, list(Path(DATA_DIR).glob("*.png")))))
    if not images:
        print("ERRORE: Nessuna immagine in ./dataset")
        return
        
    labels = [img.split(os.path.sep)[-1].split("_")[0] for img in images]
    
    print(f"Trovate {len(images)} immagini.")
    print("Generazione anteprima debug...")

    # Prendi 16 campioni
    sample_idxs = np.random.choice(len(images), 16, replace=False)
    
    fig = plt.figure(figsize=(10, 10))
    
    for i, idx in enumerate(sample_idxs):
        data = encode_single_sample(images[idx], labels[idx])
        
        # L'immagine è trasposta (ruotata) per la RNN. .
        img_display = tf.transpose(data["image"], perm=[1, 0, 2]).numpy()
        
        # Converti label numerica in testo
        label_nums = data["label"].numpy()
        label_text = tf.strings.reduce_join(num_to_char(label_nums)).numpy().decode("utf-8")
        
        ax = fig.add_subplot(4, 4, i + 1)
        ax.imshow(img_display)
        ax.set_title(f"Label: {label_text}")
        ax.axis("off")
    
    plt.savefig("debug_batch_check.png")
    print("SALVATO: Apri 'debug_batch_check.png' e controlla se le etichette corrispondono alle immagini!")

if __name__ == "__main__":
    main()