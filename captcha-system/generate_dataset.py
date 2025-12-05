# Optimized version of generate_dataset.py using multiprocessing by Marta
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST DATASET GENERATION WRAPPER (MULTIPROCESSING)
Uses all CPU cores to generate images in parallel.
"""

import os
import multiprocessing
from tqdm import tqdm

# Original script
import generate 

# ==========================================
# CONFIGURATION
# ==========================================
NUM_IMAGES = 20000          # How many images to generate
OUTPUT_DIR = "dataset"      # Destination folder
IMG_WIDTH = 420             # Width
IMG_HEIGHT = 220            # Height

# Asset paths (Must exist)
BG_DIR = "background_images"
OV_DIR = "overlay_images"
FONTS_DIR = "fonts"

# ==========================================
# WORKER FUNCTION (Executed by each core)
# ==========================================
def generate_single_image(index):
    """
    Generates a single CAPTCHA image.
    This function is run in parallel by multiple processes.
    """
    try:
        # Call the function from the original 'generate.py' script
        img, text = generate.generate_camouflage_captcha(
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            bg_dir=BG_DIR,
            ov_dir=OV_DIR,
            symbols="abcdefghijklmnopqrstuvwxyz0123456789",
            fonts_dir=FONTS_DIR,
            # Adapt font size to image height
            font_size=int(IMG_HEIGHT * 0.55), 
            min_length=5, 
            max_length=5,
            blur=0.8,
            bold=2,
            colorblind=False,
            difficulty=0.2
        )

        if img is None:
            return False

        filename = f"{text}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        img.save(save_path)
        return True

    except Exception:
        return False

# ==========================================
# MAIN BLOCK
# ==========================================
def main():
    # 1. Create destination folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Security check: Ensure asset folders exist
    if not os.path.isdir(BG_DIR) or not os.path.isdir(OV_DIR):
        print(f"CRITICAL ERROR: Could not find folders '{BG_DIR}' or '{OV_DIR}'.")
        print("Please create them and add images before starting!")
        return

    # 3. Setup Multiprocessing
    # Count available CPU cores
    num_cores = multiprocessing.cpu_count()
    # Use one less than max to avoid freezing the PC (ihih)
    workers = max(1, num_cores)

    print(f"--- STARTING FAST DATASET GENERATION ---")
    print(f"Target: {NUM_IMAGES} images")
    print(f"Dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Using {workers} CPU cores in parallel")

    # 4. Launch the parallel process pool
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(generate_single_image, range(NUM_IMAGES)), 
            total=NUM_IMAGES, 
            desc="Generating"
        ))

    print(f"\nDONE! Generated {NUM_IMAGES} images in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    # MANDATORY for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
