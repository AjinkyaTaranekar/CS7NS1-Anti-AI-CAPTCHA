# Developed by Marta
#!/usr/bin/env python3
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm
import shutil

# ==========================================
# CONFIGURAZIONE PATH (Adattata alla tua foto)
# ==========================================
NUM_IMAGES = 10000          
OUTPUT_DIR = "yolo_dataset" # Verrà creata dentro 'attackers'
IMG_WIDTH = 640             
IMG_HEIGHT = 320            

# Percorsi relativi: ".." esce da 'attackers' e entra in 'captcha-system'
# Relative paths: ".." exits 'attackers' and enters 'captcha-system'
BASE_PATH = os.path.join("..", "captcha-system")

BG_DIR = os.path.join(BASE_PATH, "background_images")
OV_DIR = os.path.join(BASE_PATH, "overlay_images")
FONTS_DIR = os.path.join(BASE_PATH, "fonts")

# Classi: 0-9, a-z (36 classi)
CLASSES = list("0123456789abcdefghijklmnopqrstuvwxyz")
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

def check_paths():
    """Controlla che le cartelle esistano prima di iniziare."""
    if not os.path.exists(BG_DIR):
        print(f"ERRORE: Non trovo la cartella: {os.path.abspath(BG_DIR)}")
        return False
    if not os.path.exists(OV_DIR):
        print(f"ERRORE: Non trovo la cartella: {os.path.abspath(OV_DIR)}")
        return False
    if not os.path.exists(FONTS_DIR):
        print(f"ERRORE: Non trovo la cartella: {os.path.abspath(FONTS_DIR)}")
        return False
    return True

def create_yolo_sample(bg_paths, ov_paths, font_paths, img_id, mode="train"):
    # 1. Setup casuale
    text = "".join(random.choices(CLASSES, k=5))
    font_size = int(IMG_HEIGHT * 0.5)
    
    try:
        font = ImageFont.truetype(random.choice(font_paths), font_size)
    except:
        font = ImageFont.load_default()

    # 2. Crea Immagine
    bg_path = random.choice(bg_paths)
    bg = Image.open(bg_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    
    ov_path = random.choice(ov_paths)
    ov = Image.open(ov_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    
    mask = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    
    yolo_labels = []
    
    total_w = sum(int(draw.textlength(ch, font=font)) for ch in text)
    start_x = (IMG_WIDTH - total_w) // 2
    start_y = (IMG_HEIGHT - font_size) // 2
    cur_x = start_x
    
    for char in text:
        draw.text((cur_x, start_y), char, fill=255, font=font)
        
        # --- CALCOLO BOX YOLO ---
        bbox = font.getbbox(char) 
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        
        abs_x = cur_x + bbox[0]
        abs_y = start_y + bbox[1]
        
        # Normalizzazione 0-1
        center_x = (abs_x + char_w / 2) / IMG_WIDTH
        center_y = (abs_y + char_h / 2) / IMG_HEIGHT
        norm_w = char_w / IMG_WIDTH
        norm_h = char_h / IMG_HEIGHT
        
        # Evita coordinate fuori dai bordi
        center_x = min(max(center_x, 0), 1)
        center_y = min(max(center_y, 0), 1)
        norm_w = min(norm_w, 1)
        norm_h = min(norm_h, 1)

        class_id = CLASS_MAP[char]
        yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        cur_x += int(draw.textlength(char, font=font))

    # 3. Effetti
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
    final_img = Image.composite(ov, bg, mask)
    
    noise = np.random.randint(0, 50, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    final_img = Image.blend(final_img, Image.fromarray(noise), alpha=0.1)

    # 4. Salvataggio
    img_filename = f"{img_id}.jpg"
    txt_filename = f"{img_id}.txt"
    
    final_img.save(os.path.join(OUTPUT_DIR, "images", mode, img_filename))
    
    with open(os.path.join(OUTPUT_DIR, "labels", mode, txt_filename), "w") as f:
        f.write("\n".join(yolo_labels))

def main():
    # Controllo preliminare percorsi
    if not check_paths():
        return

    # Setup Cartelle Output
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    
    for t in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", t), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", t), exist_ok=True)

    # Caricamento file
    # Loading files using the new paths
    bg_files = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    ov_files = [os.path.join(OV_DIR, f) for f in os.listdir(OV_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    font_files = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.lower().endswith('.ttf')]

    if not bg_files or not ov_files or not font_files:
        print("ERRORE: Una delle cartelle è vuota! Controlla background_images, overlay_images o fonts.")
        return

    print(f"Generazione Dataset YOLO in '{OUTPUT_DIR}'...")
    print(f"Sfondi: {len(bg_files)}, Overlay: {len(ov_files)}, Fonts: {len(font_files)}")
    
    # Generazione
    for i in tqdm(range(int(NUM_IMAGES * 0.9)), desc="Train Set"):
        create_yolo_sample(bg_files, ov_files, font_files, f"train_{i}", "train")
        
    for i in tqdm(range(int(NUM_IMAGES * 0.1)), desc="Val Set"):
        create_yolo_sample(bg_files, ov_files, font_files, f"val_{i}", "val")

    # File YAML per YOLO
    abs_path = os.path.abspath(OUTPUT_DIR)
    # Windows
    abs_path = abs_path.replace("\\", "/") 
    
    yaml_content = f"""
path: {abs_path}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
    """
    with open("data.yaml", "w") as f:
        f.write(yaml_content)
        
    print("\nDataset Generato!")
    print("File 'data.yaml' creato correttamente.")

if __name__ == "__main__":
    main()