# CS7NS1-Anti-AI-CAPTCHA
An Anti-AI CAPTCHA system for CS7NS1 module at Trinity

## Local Development Setup

### 1. Generate Training and Validation Data

#### Generate Samples
```bash
python generate.py --width 128 --height 64 --min-length 5 --max-length 5 --symbols symbols.txt --count 10 --output-dir sample_data --bg-dir background_images --font-path .\fonts\ARIAL.TTF
```

#### Generate 16,000 training images:
```bash
python generate.py --width 128 --height 64 --min-length 5 --max-length 5 --symbols symbols.txt --count 16000 --output-dir training_data --bg-dir background_images --font-path .\fonts\ARIAL.TTF
```

#### Generate 4,000 validation images:
```bash
python generate.py --width 128 --height 64 --min-length 5 --max-length 5 --symbols symbols.txt --count 4000 --output-dir validation_data --bg-dir background_images --font-path .\fonts\ARIAL.TTF
```