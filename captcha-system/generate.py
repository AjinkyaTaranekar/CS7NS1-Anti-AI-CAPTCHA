#!/usr/bin/env python3
"""Camouflage CAPTCHA generator with visual obfuscation techniques.

This module generates CAPTCHAs where text is camouflaged into textured backgrounds
using edge blur, character dilation, and strategic color pairing based on color theory.
"""

import argparse
import colorsys
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from tqdm import tqdm


def get_dominant_color(img_path: str) -> Tuple[float, float, float]:
    """Extract the dominant color of an image as HSV values.
    
    Args:
        img_path: Path to the image file.
    
    Returns:
        Tuple of (hue in degrees 0-360, saturation 0-1, value 0-1).
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((50, 50))
        data = np.array(img)
        avg_r = np.mean(data[:,:,0])
        avg_g = np.mean(data[:,:,1])
        avg_b = np.mean(data[:,:,2])
        r, g, b = avg_r/255, avg_g/255, avg_b/255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return (h * 360, s, v)
    except Exception:
        return (0, 0, 0)


def colors_match(hue1: float, hue2: float, mode: str = "complementary") -> bool:
    """Check if two hues match based on color theory.
    
    Args:
        hue1: First hue in degrees (0-360).
        hue2: Second hue in degrees (0-360).
        mode: Matching mode - 'complementary', 'analogous', or 'blend'.
    
    Returns:
        True if the hues match according to the specified mode.
    """
    diff = abs(hue1 - hue2) % 360
    diff = min(diff, 360 - diff)
    if mode == "complementary":
        return 150 <= diff <= 210
    elif mode == "analogous":
        return diff <= 60
    elif mode == "blend":
        return 120 <= diff <= 240
    return False


def create_center_mask(width: int, height: int, text_area_ratio: float = 0.7) -> Image.Image:
    """Create a soft oval mask for center area blurring.
    
    Args:
        width: Width of the mask.
        height: Height of the mask.
        text_area_ratio: Ratio of the text area to the total image size.
    
    Returns:
        PIL Image mask with soft edges.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    center_w = int(width * text_area_ratio)
    center_h = int(height * text_area_ratio)
    left = (width - center_w) // 2
    top = (height - center_h) // 2
    draw.ellipse([left, top, left + center_w, top + center_h], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=width * 0.08))
    return mask


def create_camouflage_text(
    bg_path: Union[str, Image.Image],
    overlay_path: Union[str, Image.Image],
    text: str,
    width: int = 420,
    height: int = 220,
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,
    blur_radius: float = 1.5,
    bold_amount: Optional[int] = None,
    colorblind: bool = False,
    difficulty: float = 0.5,
) -> Image.Image:
    """Generate a camouflage CAPTCHA with text blended into a textured background.
    
    Args:
        bg_path: Path to background image or PIL Image object.
        overlay_path: Path to overlay image or PIL Image object.
        text: Text to render in the CAPTCHA.
        width: Width of output image.
        height: Height of output image.
        font_path: Path to TrueType font file.
        font_size: Font size in pixels.
        blur_radius: Gaussian blur radius for edge softening.
        bold_amount: Dilation pixels for character boldness.
        colorblind: Use high-contrast blue-orange palette.
        difficulty: Noise level (0.0=easy, 1.0=very hard).
        output_path: Path to save the generated CAPTCHA.
    
    Returns:
        Generated PIL Image object.
    """
    if font_size is None:
        font_size = int(height * 0.55)

    try:
        font = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", font_size)
    except Exception:
        bundled = Path(__file__).parent / "FreeSansBold.ttf"
        if bundled.exists():
            font = ImageFont.truetype(str(bundled), font_size)
        else:
            font = ImageFont.load_default()

    def load_and_fit(src) -> Image.Image:
        """Load and fit image to target dimensions by resizing or tiling."""
        if isinstance(src, Image.Image):
            img = src.copy()
        else:
            img = Image.open(src).convert("RGB")

        if img.width > width or img.height > height:
            img = img.resize((width, height), Image.LANCZOS)
        elif img.width < width or img.height < height:
            tx = (width // img.width) + 2
            ty = (height // img.height) + 2
            tiled = Image.new("RGB", (img.width * tx, img.height * ty))
            for i in range(tx):
                for j in range(ty):
                    tiled.paste(img, (i * img.width, j * img.height))
            img = tiled.crop((0, 0, width, height))

        return img

    bg = load_and_fit(bg_path)
    overlay = load_and_fit(overlay_path)

    center_mask = create_center_mask(width, height, text_area_ratio=0.7)
    bg_center = bg.copy()
    bg_center = bg_center.filter(ImageFilter.GaussianBlur(radius=2))
    bg = Image.composite(bg_center, bg, center_mask)

    if colorblind:
        bg_gray = bg.convert("L")
        overlay_gray = overlay.convert("L")
        bg = ImageOps.colorize(bg_gray, black="#003366", white="#ffcc00")
        overlay = ImageOps.colorize(overlay_gray, black="#003366", white="#ffcc00")

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    char_spacing = int(font_size * 0.18)
    total_w = sum(int(draw.textlength(ch, font=font)) for ch in text)
    total_w += char_spacing * (len(text) - 1)

    start_x = (width - total_w) // 2
    start_y = (height - font_size) // 2

    cur_x = start_x
    for ch in text:
        draw.text((cur_x, start_y), ch, fill=255, font=font)
        cur_x += int(draw.textlength(ch, font=font)) + char_spacing

    if blur_radius > 0.0:
        blur = max(0.5, blur_radius)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur))

    if bold_amount is None:
        bold_amount = max(1, int(font_size * 0.03))
    if bold_amount > 0:
        pad = bold_amount + 2
        mask_padded = ImageOps.expand(mask, border=pad, fill=0)
        mask_padded = mask_padded.filter(ImageFilter.MaxFilter(size=3))
        mask = mask_padded.crop((pad, pad, width + pad, height + pad))

    out = Image.composite(overlay, bg, mask)

    if difficulty > 0.0:
        noise = np.random.randint(0, int(255 * difficulty), (height, width, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise).convert("RGB")
        out = Image.blend(out, noise_img, alpha=min(difficulty, 0.3))

    return out


def generate_camouflage_captcha(
    width: int,
    height: int,
    bg_dir: str,
    ov_dir: str,
    symbols_file: str,
    fonts_dir: str,
    font_size: int,
    min_length: int,
    max_length: int,
    blur: float,
    bold: int,
    colorblind: bool,
    difficulty: float,
) -> Tuple[Optional[Image.Image], str]:
    """Generate a single camouflage CAPTCHA and return the image and text."""
    
    if not os.path.exists(symbols_file):
        print(f"Warning: '{symbols_file}' missing → using default alphanum")
        sym = "abcdefghijklmnopqrstuvwxyz0123456789"
    else:
        with open(symbols_file) as f:
            sym = f.read().strip().lower()
        if not sym:
            sym = "abcdefghijklmnopqrstuvwxyz0123456789"
    font_path = None
    if fonts_dir and os.path.isdir(fonts_dir):
        font_files = [
            os.path.join(fonts_dir, f)
            for f in os.listdir(fonts_dir)
            if f.lower().endswith((".ttf", ".otf"))
        ]
        if font_files:
            font_path = random.choice(font_files)

    if not os.path.isdir(bg_dir):
        print(f"Error: background directory '{bg_dir}' missing")
        sys.exit(1)

    bg_paths = [
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".avif"))
    ]
    bg_paths = [p for p in bg_paths if os.path.isfile(p)]
    if len(bg_paths) < 1:
        print("Error: no usable background images")
        sys.exit(1)

    bg_colors = {p: get_dominant_color(p) for p in bg_paths}

    if not os.path.isdir(ov_dir):
        print(f"Error: overlay directory '{ov_dir}' missing")
        sys.exit(1)

    ov_paths = [
        os.path.join(ov_dir, f)
        for f in os.listdir(ov_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".avif"))
    ]
    ov_paths = [p for p in ov_paths if os.path.isfile(p)]
    if len(ov_paths) < 1:
        print("Error: no usable overlay images")
        sys.exit(1)

    ov_colors = {p: get_dominant_color(p) for p in ov_paths}

    length = random.randint(min_length, max_length)
    text = "".join(random.choice(sym) for _ in range(length))

    bg = random.choice(bg_paths)
    bg_hue = bg_colors[bg][0]

    candidates = [ov for ov in ov_paths if colors_match(bg_hue, ov_colors[ov][0], "analogous")]
    if not candidates:
        candidates = [ov for ov in ov_paths if colors_match(bg_hue, ov_colors[ov][0], "complementary")]
    if not candidates:
        candidates = ov_paths

    ov = random.choice(candidates)
    captcha = None
    try:
        captcha = create_camouflage_text(
            bg_path=bg,
            overlay_path=ov,
            text=text,
            width=width,
            height=height,
            font_path=font_path,
            font_size=font_size,
            blur_radius=blur,
            bold_amount=bold,
            colorblind=colorblind,
            difficulty=difficulty,
        )
    except Exception as e:
        print(f"\nFailed {text}: {e}")
    return captcha, text

def generate_captchas():
    """Command-line interface for batch CAPTCHA generation."""
    parser = argparse.ArgumentParser(
        description="Camouflage CAPTCHA generator – blur + bold + color-blind"
    )
    parser.add_argument("--width", type=int, default=420)
    parser.add_argument("--height", type=int, default=220)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="camouflage_output")
    parser.add_argument("--bg-dir", type=str, default="background_images")
    parser.add_argument("--ov-dir", type=str, default="overlay_images")
    parser.add_argument("--symbols", type=str, default="symbols.txt")
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=6)
    parser.add_argument("--fonts-dir", type=str, default="fonts",
                        help="Directory containing font files (.ttf/.otf)")
    parser.add_argument("--font-size", type=int, default=120,
                        help="Font size in pixels")
    parser.add_argument("--blur", type=float, default=0.8,
                        help="Gaussian blur radius on character edges (0 = none)")
    parser.add_argument("--bold", type=int, default=5,
                        help="Extra dilation pixels (None = auto)")
    parser.add_argument("--colorblind", action="store_true",
                        help="Use blue-orange high-contrast palette")
    parser.add_argument("--difficulty", type=float, default=0.2,
                        help="0.0 = easy, 1.0 = very hard (adds noise)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm(range(args.count), desc="CAPTCHA"):
        out_file = os.path.join(args.output_dir, f"captcha_{i+1}.png")
        img, text = generate_camouflage_captcha(
            width=args.width,
            height=args.height,
            bg_dir=args.bg_dir,
            ov_dir=args.ov_dir,
            symbols_file=args.symbols,
            fonts_dir=args.fonts_dir,
            font_size=args.font_size,
            min_length=args.min_length,
            max_length=args.max_length,
            blur=args.blur,
            bold=args.bold,
            colorblind=args.colorblind,
            difficulty=args.difficulty,
        )
        if img is None:
            print(f"Failed to generate CAPTCHA {i+1}")
        else:
            img.save(out_file)
            print(f"Generated {text.upper()} → {out_file}")

    print(f"\nDone → {args.output_dir} ({args.count} files)")

if __name__ == "__main__":
    generate_captchas()