# CS7NS1 – Anti-AI CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 at Trinity College Dublin. This repository contains a configurable CAPTCHA generator, a FastAPI backend that enforces several anti-automation signals (mouse/keystroke analysis, honeypots, navigator fingerprints, PoW), an OCR microservice, and several attack suites to evaluate the robustness of the system.

[Anti-AI-CAPTCHA-System-Architecture.pdf](https://github.com/user-attachments/files/23797981/Anti-AI-CAPTCHA-System-Architecture.pdf)

---

## Quick Links

- Backend web app: `captcha-system/main.py` (serves `index.html`) — default port 5174
- OCR / model microservice: `captcha-system/model_service.py` — default port 8001
- CAPTCHA generator: `captcha-system/generate.py` and `generate_dataset.py`
- Behavioral model training: `captcha-system/captcha_mouse_movement_prediction/train_model.py`
- Attack suite: `attackers/` (attacker_1..4.py, utils)

---

## Table of Contents

1. Project overview
2. Local dev setup (Python, venv, dependencies)
3. Run the services (backend, OCR microservice)
4. Generate datasets and CAPTCHAs
5. Train the mouse movement & recognition models
6. Attack simulations (how to run the attacker scripts)
7. API & Frontend details
8. File structure overview
9. Raspberry Pi deploy hints

---

## Project overview

This project attempts to produce CAPTCHAs that resist AI-based solving by combining multiple defenses:

- Camouflage-style visual CAPTCHAs (text blended into backgrounds; see `generate.py`).
- Behavioral detection using keystroke/mouse dynamics and an ML model (`captcha_mouse_movement_prediction`).
- Browser fingerprinting and navigator signal validation.
- Rate limiting and server-side PoW challenge (proof-of-work) to slow automated solvers.
- Optional OCR-based API (`model_service.py`) that provides a microservice for canvas OCR and movement model evaluation.

This repo includes attacker scripts under `attackers/` to validate and benchmark how easily these CAPTCHAs can be bypassed.

---

## Prerequisites

- Python 3.8 (recommended) — some dependencies like EasyOCR may need newer Python versions or specific wheels.
- pip
- Virtualenv (recommended to isolate dependencies)
- Git
- For Playwright-based attackers: Playwright CLI and browser binaries

Windows users: this repo has been tested on Windows with Python 3.8 and a standard PowerShell environment.

---

### File Structure

```
CS7NS1-Anti-AI-CAPTCHA/
├── captcha-system/                    # Main CAPTCHA service
│   ├── main.py                        # Backend API (FastAPI) - 1900 lines
│   ├── generate.py                    # CAPTCHA image generation - 475 lines
│   ├── model_service.py               # ML microservice - 222 lines
│   ├── index.html                     # Frontend SPA - 987 lines
│   ├── config/
│   │   └── constants.py               # System constants
│   ├── captcha_mouse_movement_prediction/
│   │   ├── utils.py                   # Feature extraction utilities
│   │   ├── train_model.py             # ML model training
│   │   └── models/
│   │       └── mouse_movement_model.pkl  # Trained classifier
│   ├── fonts/                         # TrueType fonts for CAPTCHA text
│   ├── background_images/             # Background textures for camouflage
│   ├── overlay_images/                # Overlay textures for blending
│   ├── captcha_images/                # Generated CAPTCHA storage
│   │   └── drawings/                  # User canvas submissions
│   └── logs/                          # Persistent audit logs
│
└── attackers/                         # Attack validation suite
    ├── attacker_1.py                  # EasyOCR + Multi-channel attack
    ├── attacker_2.py                  # CNN/CTC model attack
    ├── attacker_3.py                  # LLM (Gemini) attack
    ├── attacker_4.py                  # YOLO object detection attack
    ├── utils.py                       # Shared attack utilities
    ├── symbol_mouse_data.json         # Pre-recorded human strokes
    ├── train.py                       # CNN training script
    ├── train_yolo.py                  # YOLO training script
    └── attack_results.csv             # Attack outcome tracking
```

---

## Local Development Setup

1. Clone the repo

```powershell
git clone https://github.com/AjinkyaTaranekar/CS7NS1-Anti-AI-CAPTCHA.git
cd CS7NS1-Anti-AI-CAPTCHA
```

2. Create & activate virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1  # Windows PowerShell
# or for cmd: venv\Scripts\activate
# or linux / macOS: source venv/bin/activate
```

3. Install backend dependencies

```powershell
pip install -r captcha-system/backend-service-requirements.txt
```

Install the OCR/model microservice dependencies (separate environment if preferred):

```powershell
pip install -r captcha-system/model-service-requirements.txt
```

For the attacker suite, install:

```powershell
pip install -r attackers/requirements.txt
# Playwright requires an extra setup step to install browsers
python -m playwright install chromium
```

NOTE: If you are deploying on resource-constrained Pi hardware, use `captcha-system/backend-service-rasp-requirements.txt` for a trimmed set of compatible versions.

---

## Running the Services

1. Start the OCR/Model Microservice (optional but recommended if you want to use the OCR endpoint):

```powershell
cd captcha-system
python model_service.py
# OR explicitly with uvicorn
powershell -Command "uvicorn model_service:app --host 0.0.0.0 --port 8001 --reload"
```

1.1 Optionally, connect the microservice with NGROK for external access:

```powershell
ngrok http 8001
```

1.2 Substitute the public URL in `captcha-system/config/constants.py` under `MODEL_SERVICE_URL`.

2. Start the main backend service (serves `index.html` and REST APIs):

```powershell
# From captcha-system folder
python main.py
# OR explicitly with uvicorn for more controlled configs
powershell -Command "uvicorn main:app --host 0.0.0.0 --port 5174 --reload"
```

3. Visit the frontend at `http://localhost:5174`. The web page (`index.html`) calls the backend endpoints: `/api/captcha/challenge` and `/api/signup`.

---

## Endpoints (Quick Reference)

- GET / — Serves `index.html` (main SPA)
- POST /api/captcha/challenge — Generates a new CAPTCHA + PoW challenge (response model: `CaptchaResponse`)
- POST /api/signup — Signup endpoint that validates PoW, behavioral signals, fingerprint, and the CAPTCHA solution (request model: `SignupRequest`)

Microservice endpoints:
Additional microservice endpoints:

- POST /ocr — OCR the provided base64 PNG payload using EasyOCR (model service)
- POST /human_evaluate — Query the human/bot classifier with kinematic vectors (returns probability list)

The frontend also sends navigator signals for additional server-side scoring. The server uses rate limiting and returns helpful codes (e.g. 428 for refresh/challenge-required).

---

## Generate sample CAPTCHAs & Datasets

1. Generate a handful of sample CAPTCHAs with `generate.py`:

```powershell
cd captcha-system
python generate.py --count 10 --output-dir sample_data --bg-dir background_images --ov-dir overlay_images --fonts-dir fonts
```

2. Generate widespread synthetic dataset using `generate_dataset.py` (multi-process):

```powershell
cd captcha-system
python generate_dataset.py
```

`generate.py` has CLI options for width/height, blur, font-size, difficulty, and more.

---

## Train the Movement & Recognition Models

Training uses the `captcha_mouse_movement_prediction` scripts and dataset files.

Steps:

1. Place the behavioral dataset in `captcha-system/captcha_mouse_movement_prediction/data/` (e.g., download from the provided dataset/Zenodo output).
2. Run training:

```powershell
cd captcha-system/captcha_mouse_movement_prediction
python train_model.py
```

This script trains a human vs bot XGBoost classifier and a character recognition classifier (if training data is present) — outputted model files are saved under `captcha_mouse_movement_prediction/models/`.

IMPORTANT: Generated models are referenced by the microservice and backend, so ensure the produced files are correctly placed. `config/constants.py` contains: `SYMBOLS`, `MOUSE_MOVEMENT_MODEL`, and `CHAR_RECOGNITION_MODEL`.

---

## Attack Suite: How to Run Attackers

The `attackers` folder holds multiple attacker scripts that attempt to automate solving / bypassing CAPTCHAs. They rely on Playwright and EasyOCR and include example workflows.

Install attacker dependencies:

```powershell
pip install -r attackers/requirements.txt
python -m playwright install chromium
```

Then run an attacker script to simulate an attack against the running backend (defaults: `http://localhost:5174`):

```powershell
cd attackers
python attacker_1.py
python attacker_2.py
python attacker_3.py
python attacker_4.py
```

Policy notes:

- `attack-recordings/` will contain optional Playwright video recordings when `record_video=True` in attacker scripts.
- `attack_results.csv` will contain a summary of attack attempts and results.

---

## Notes for Developers (Important details / files to inspect)

- `captcha-system/main.py` — FastAPI backend implementing PoW, scoring, fingerprinting, rate-limiting and serving the SPA.
- `captcha-system/model_service.py` — standalone EasyOCR and human movement model microservice (port 8001 by default).
- `captcha-system/generate.py` — image generator with color/palette and prompt-injection techniques to confuse LLMs.
- `captcha-system/generate_dataset.py` — multiprocess wrapper to generate thousands of CAPTCHAs.
- `captcha-system/config/constants.py` — shared constants for model file names and symbols.
- `captcha-system/captcha_mouse_movement_prediction` — feature extraction, scripts and models for human vs bot detection.
- `attackers` — attacker scripts and helpers (Playwright / EasyOCR-based tactics).

Log files are in `captcha-system/logs/` (e.g., `signup_attempts.log`). Generated CAPTCHAs are placed in `captcha_images` by default and sample outputs from `generate.py` live in `sample_data` when you use that command.

---

## Raspberry Pi / Low-power Deployment

To deploy the CAPTCHA system on a Raspberry Pi, follow these additional steps:

## Port Forwarding from Raspberry Pi to Local Machine

To access the CAPTCHA system running on your Raspberry Pi from your local machine, you can set up port forwarding using SSH. Here’s how to do it:

1. Create an SSH tunnel from your local machine to the Jump Server:

   ```bash
   ssh -L 5174:localhost:5174 <username>@macneill.scss.tcd.ie
   ```

2. After entering your password, the tunnel will be established. Now, create another SSH tunnel from the Jump Server to your Raspberry Pi:

   ```bash
   ssh -L 5174:0.0.0.0:5174 <username>@rasp-015.berry.scss.tcd.ie
   ```

3. After entering your password, the second tunnel will be established. Now run the CAPTCHA server on your Raspberry Pi if it is not already running.
4 Use `captcha-system/backend-service-rasp-requirements.txt` to install matching dependency versions for a limited environment.
5 Run services with `python main.py` and `python model_service.py` for simplicity. If resources are constrained, you can avoid starting the separate OCR microservice and rely on EasyOCR in-process for smaller deployments (but this may increase memory usage).

---

## Acknowledgements & Resources

- Some datasets and ideas used for training and evaluation are external (e.g. mouse-stroke datasets like those on Zenodo). See `captcha_mouse_movement_prediction/` for details and dataset links.

---
