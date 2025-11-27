# CS7NS1 – Anti-AI CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 at Trinity College Dublin. This repository contains a configurable CAPTCHA generator, a FastAPI backend that enforces several anti-automation signals (mouse/keystroke analysis, honeypots, navigator fingerprints, PoW), an OCR microservice, and several attack suites to evaluate the robustness of the system.

---

## Quick summary

- Backend (dev): runs at `http://localhost:5174` (default) — `python main.py` or `uvicorn main:app --reload`
- Model/OCR microservice (dev): runs at `http://localhost:8001` (default) — `python model_service.py` or `uvicorn model_service:app --reload`
- Generator CLI: `python generate.py` and dataset helper `python generate_dataset.py`

---

## Basic setup (local development)

1. Clone repository

```powershell
git clone https://github.com/AjinkyaTaranekar/CS7NS1-Anti-AI-CAPTCHA.git
cd CS7NS1-Anti-AI-CAPTCHA
```

2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1 # (PowerShell)
```

3. Install dependencies (backend)

```powershell
pip install -r captcha-system/backend-service-requirements.txt
```

Install microservice deps:

```powershell
pip install -r captcha-system/model-service-requirements.txt
```

Install attacker dependencies (if you want to run the attack suite locally):

```powershell
pip install -r attackers/requirements.txt
python -m playwright install chromium
```

---

## Run services

Start the microservice (optional):

```powershell
cd captcha-system
python model_service.py
# or: uvicorn model_service:app --host 0.0.0.0 --port 8001 --reload
```

Start the main backend:

```powershell
cd captcha-system
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 5174 --reload
```

Open `http://localhost:5174` in a browser to interact with the SPA and the APIs.

---

## Common endpoints

- GET `/` — serves `index.html`.
- POST `/api/captcha/challenge` — request a new CAPTCHA challenge (includes a PoW challenge token).
- GET `/captcha_images/{captcha_id}.png` — serves the generated CAPTCHA image file.
- POST `/api/signup` — signup endpoint (POST body expected to conform to `SignupRequest` model in `main.py`).

Microservice endpoints (OCR & evaluation):

- POST `/ocr` — returns OCR result for supplied base64 image.
- POST `/human_evaluate` — returns human-probability predictions for kinematic vectors.

---

## Generating CAPTCHAs and Datasets

Generate a handful of CAPTCHAs with the CLI:

```powershell
cd captcha-system
python generate.py --count 10 --output-dir sample_data --bg-dir background_images --ov-dir overlay_images --fonts-dir fonts
```

Create a large dataset (multiprocessing helper):

```powershell
python generate_dataset.py
```

---

## Train the models

Place relevant CSVs in `captcha_mouse_movement_prediction/data/` and run training:

```powershell
cd captcha-system/captcha_mouse_movement_prediction
python train_model.py
```

This trains a human vs bot detector (XGBoost) and a character recognition model if char-training data is available. Trained models are saved under `captcha_mouse_movement_prediction/models/`.

---

## Attacker scripts & evaluation

The `attackers` folder contains multiple test attackers that attempt to solve the CAPTCHA or bypass protections. These scripts often use Playwright to automate the browser and either EasyOCR/LLM/YOLO pipelines to result in text.

Run an attacker to see how the system holds up (ensure backend & microservice are running):

```powershell
cd attackers
python attacker_1.py
```

Recorded playback videos are placed in `attack-recordings/` and results are appended to `attack_results.csv`.

---

## File structure (high-level)

- `captcha-system/`
  - `main.py` — FastAPI backend
  - `index.html` — frontend SPA
  - `generate.py` — CAPTCHA generator
  - `model_service.py` — OCR and human model microservice
  - `captcha_mouse_movement_prediction/` — feature extraction & training
  - `background_images/`, `overlay_images/`, `fonts/` — generator assets
  - `logs/` and `captcha_images/` — runtime outputs
- `attackers/` — attacker scripts, utilities, and recorded run artifacts

---

## Raspberry Pi / Low-resource notes

- Use `captcha-system/backend-service-rasp-requirements.txt` for low-memory / Pi-friendly package versions.
- Consider running only the main backend on the Pi and host the OCR microservice on a more powerful machine.

---

## Contributing & Tests

- Tests are currently empty by default in `captcha-system/tests/` — PRs that add testing and CI are welcome.

---

If you want, I can add a `docker-compose.yml` for easily spinning up the backend + microservice. Tell me if you'd like Compose or Dockerfiles for each service.

# CS7NS1 – Anti-AI CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 at Trinity College Dublin. This repository contains a configurable CAPTCHA generator, a FastAPI backend that enforces multiple anti-automation signals (mouse/keystroke analysis, honeypots, navigator fingerprints, PoW). It also includes a microservice for OCR and several attack suites to evaluate the robustness of the system.

---

## Quick Links

- Backend (FastAPI SPA): `captcha-system/main.py` — default port: `5174`
- Microservice (EasyOCR + Human model): `captcha-system/model_service.py` — default port: `8001`
- CAPTCHA generator & dataset: `captcha-system/generate.py`, `captcha-system/generate_dataset.py`
- Behavioral model training: `captcha-system/captcha_mouse_movement_prediction/train_model.py`
- Attack validation suite: `attackers/` (attacker_1.py … attacker_4.py)

---

## Table of contents

1. Project overview
2. Local development setup
3. Running the services
4. Endpoints and frontend
5. Generating CAPTCHAs & datasets
6. Training models
7. Attack suite
8. Notes for developers & files
9. Raspberry Pi/low-power deployment

---

## 1. Project overview

The repository aims to create a CAPTCHA that is more resilient to machine-based attacks by combining:

- Camouflage-based visual obfuscation.
- Behavioral signals (mouse & keystroke dynamics) scored by ML.
- Browser fingerprinting and navigator snapshot analysis.
- Proof-of-work (PoW) tokens to slow brute-force attempts.
- Rate limiting and honeypots to detect bots and LLM-targeted attacks.

The repo also contains attacker scripts that simulate automated attacks to benchmark defenses.

---

## 2. Local development setup

Prerequisites

- Python (3.8 recommended)
- pip
- Virtualenv (recommended)
- Git
- Playwright for browser automation (used by attackers)

Install and set up

```powershell
git clone https://github.com/AjinkyaTaranekar/CS7NS1-Anti-AI-CAPTCHA.git
cd CS7NS1-Anti-AI-CAPTCHA
python -m venv venv
venv\Scripts\Activate.ps1  # PowerShell activate
```

Install backend dependencies

```powershell
pip install -r captcha-system/backend-service-requirements.txt
```

Install OCR / model microservice dependencies (optional)

```powershell
pip install -r captcha-system/model-service-requirements.txt
```

Install attacker dependencies (optional)

```powershell
pip install -r attackers/requirements.txt
python -m playwright install chromium
```

Use the Raspberry Pi-specific requirements if targeting deployment: `captcha-system/backend-service-rasp-requirements.txt`.

---

## 3. Running the services

Start the microservice (optional)

```powershell
cd captcha-system
python model_service.py
# or: uvicorn model_service:app --host 0.0.0.0 --port 8001 --reload
```

Start the main backend

```powershell
cd captcha-system
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 5174 --reload
```

Then browse to `http://localhost:5174` to use the SPA. The SPA interacts with the API endpoints for captcha issuance and signup.

---

## 4. Endpoints and frontend

Key backend endpoints

- `GET /` — serves `index.html` (frontend)
- `POST /api/captcha/challenge` — request a new CAPTCHA (PoW challenge returned)
- `POST /api/signup` — submit a signup form with PoW and CAPTCHA solution

Key microservice endpoints (OCR / kinematic model)

- `POST /ocr` — performs OCR on a base64-encoded PNG
- `POST /human_evaluate` — evaluate human vs bot for a set of kinematic vectors

Requests and behavior are illustrated in the frontend `captcha-system/index.html` (SPA) which calls `/api/captcha/challenge` and `/api/signup` directly.

---

## 5. Generating CAPTCHAs & datasets

Generate a few example CAPTCHAs

```powershell
cd captcha-system
python generate.py --count 10 --output-dir sample_data --bg-dir background_images --ov-dir overlay_images --fonts-dir fonts
```

Create a large dataset for training (multiprocessing helper)

```powershell
python generate_dataset.py
```

`generate.py` has a CLI for setting width, height, blur radius, font size, colorblind palette, difficulty, and more.

---

## 6. Training models

Training the human vs bot classifier and the character recognition model uses the `captcha_mouse_movement_prediction` scripts.

1. Place the dataset CSV files under `captcha-system/captcha_mouse_movement_prediction/data/`.
2. Train the model(s):

```powershell
cd captcha-system/captcha_mouse_movement_prediction
python train_model.py
```

Models will be saved under `captcha_mouse_movement_prediction/models/`. The backend and microservice expect the movement model to be available with the filename configured in `config/constants.py`.

---

## 7. Attack suite (validation)

Attackers are under `attackers/` and provide various attack strategies:

- `attacker_1.py` — EasyOCR-based multi-channel channel attack + Playwright replay of strokes
- `attacker_2.py` — CNN/CTC-based approach
- `attacker_3.py` — LLM-based attack demonstration (litellm)
- `attacker_4.py` — YOLO-based object detection attack

Run an attacker

```powershell
cd attackers
python attacker_1.py
```

Recorded attack videos are saved to `attack-recordings/` when `record_video=True`, and per-attack outputs are saved in `attack_results.csv`.

---

## 8. Notes for developers & files

- Backend: `captcha-system/main.py` — PoW, rate-limiting (slowapi), scoring and verification.
- OCR microservice: `captcha-system/model_service.py` — easyocr + movement model endpoints.
- Generator: `captcha-system/generate.py` — camouflage generator with text/overlay blending and prompt injection decoys.
- Dataset helper: `captcha-system/generate_dataset.py` — multiprocessing wrapper.
- Config: `captcha-system/config/constants.py` — contains key constants (symbols, model names).
- Behavioral model: `captcha-system/captcha_mouse_movement_prediction/` — datasets and training utilities.
- Attackers: `attackers/` with helper utilities in `attackers/utils.py`.

Log files: `captcha-system/logs/signup_attempts.log`.
Generated CAPTCHAs: `captcha_images` (created at runtime) or `sample_data` for `generate.py` output.

---

## 9. Raspberry Pi / Low-power deployment

- Use `captcha-system/backend-service-rasp-requirements.txt` to install compatible package versions on ARM/low-power devices.
- To conserve memory, consider disabling the OCR microservice and running only the main backend on a Pi (the microservice can be moved to a more powerful server if needed).

---

## Tests & Contributing

- Tests are located (empty by default) in `captcha-system/tests/`.
- Contributions are welcome — PRs that improve test coverage, clarity, and reproducible runs are appreciated.

---

If you'd like, I can generate a `docker-compose.yml` to run the backend and microservice together for local testing — tell me if you prefer Compose or plain processes.

# CS7NS1 – Anti-AI CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 at Trinity College Dublin. This repository contains a configurable CAPTCHA generator, a FastAPI backend that enforces several anti-automation signals (mouse/keystroke analysis, honeypots, navigator fingerprints, PoW), an OCR microservice, and several attack suites to evaluate the robustness of the system.

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

- Use `captcha-system/backend-service-rasp-requirements.txt` to install matching dependency versions for a limited environment.
- Run services with `python main.py` and `python model_service.py` for simplicity. If resources are constrained, you can avoid starting the separate OCR microservice and rely on EasyOCR in-process for smaller deployments (but this may increase memory usage).

---

## Contribution & Tests

- Tests are located (empty by default) in `captcha-system/tests/`. Additional test coverage and CI integration are welcome.

---

## Acknowledgements & Resources

- Some datasets and ideas used for training and evaluation are external (e.g. mouse-stroke datasets like those on Zenodo). See `captcha_mouse_movement_prediction/` for details and dataset links.

---

If you'd like, I can also generate a `docker-compose.yml` for easy local testing (e.g., start backend + model service + optional attacker container) — let me know how you want to use the system (local dev vs production vs tests).

# CS7NS1 – Anti-AI CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 at Trinity College Dublin. This repository contains a configurable CAPTCHA generator, a FastAPI backend that enforces several anti-automation signals (mouse/keystroke analysis, honeypots, navigator fingerprints, PoW), an OCR microservice, and several attack suites to evaluate the robustness of the system.

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
   ssh -L 8000:localhost:5174 taraneka@macneill.scss.tcd.ie
   ```

2. After entering your password, the tunnel will be established. Now, create another SSH tunnel from the Jump Server to your Raspberry Pi:

   ```bash
   ssh -L 8000:0.0.0.0:8000 taraneka@rasp-015.berry.scss.tcd.ie
   ```

3. After entering your password, the second tunnel will be established. Now run the CAPTCHA server on your Raspberry Pi if it is not already running.
4 Use `captcha-system/backend-service-rasp-requirements.txt` to install matching dependency versions for a limited environment.
5 Run services with `python main.py` and `python model_service.py` for simplicity. If resources are constrained, you can avoid starting the separate OCR microservice and rely on EasyOCR in-process for smaller deployments (but this may increase memory usage).

---

## Acknowledgements & Resources

- Some datasets and ideas used for training and evaluation are external (e.g. mouse-stroke datasets like those on Zenodo). See `captcha_mouse_movement_prediction/` for details and dataset links.

---
