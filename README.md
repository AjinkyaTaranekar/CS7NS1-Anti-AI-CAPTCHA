# CS7NS1-Anti-AI-CAPTCHA

An Anti-AI CAPTCHA system for CS7NS1 module at Trinity College Dublin. This project aims to create a CAPTCHA system that is more resistant to AI-based solvers by incorporating various techniques such as image distortion, optical illusions, rate limiting, honey pots and browser fingerprinting.

## Local Development Setup for CAPTCHA System

### Prerequisites

- Python 3.8 (As this is going to be deployed on Raspberry Pi to tackle resource constraints)
- pip
- Virtualenv (optional but recommended)
- Git
- A web browser for testing the CAPTCHA system

### Setup Instructions

1. **Clone the Repository**

   ```bash
       git clone https://github.com/AjinkyaTaranekar/CS7NS1-Anti-AI-CAPTCHA.git
   ```

2. **Create and Activate a Virtual Environment (Optional)**

   ```bash
        python -m venv venv
        # On Windows
        venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
   ```

3. **Install Required Packages**

   ```bash
    pip install -r requirements.txt
   ```

4. **Generate Sample CAPTCHA Data**

   Ensure you have a `symbols.txt` file with the desired symbols and a directory `background_images` with background images, `overlay_images` with overlay images, and a `fonts` directory with font files.

    ```bash
    python generate.py --width 128 --height 64 --min-length 5 --max-length 5 --symbols symbols.txt --count 10 --output-dir sample_data --bg-dir background_images --ov-dir overlay_images --font-dir fonts
    ```

5. **Train the Mouse Movement Prediction Model**

   Navigate to the `captcha_mouse_movement_prediction` directory and run the training script:

   Download the data folder from the provided link (<https://zenodo.org/records/50022>) and extract it into `captcha_mouse_movement_prediction/data/`.

   ```bash
   cd captcha-system/captcha_mouse_movement_prediction
   python train_model.py
   cd ../..
   ```

6. **Run the Server**

    ```bash
    python main.py
    ```

7. **Access the CAPTCHA System**
    Open your web browser and navigate to `http://localhost:8000` to access the CAPTCHA system.

## Deploying and Running on Raspberry Pi

To deploy the CAPTCHA system on a Raspberry Pi, follow these additional steps:

## Port Forwarding from Raspberry Pi to Local Machine

To access the CAPTCHA system running on your Raspberry Pi from your local machine, you can set up port forwarding using SSH. Here’s how to do it:

1. Create an SSH tunnel from your local machine to the Jump Server:

   ```bash
   ssh -L 8000:localhost:8000 taraneka@macneill.scss.tcd.ie
   ```

2. After entering your password, the tunnel will be established. Now, create another SSH tunnel from the Jump Server to your Raspberry Pi:

   ```bash
   ssh -L 8000:0.0.0.0:8000 taraneka@rasp-015.berry.scss.tcd.ie
   ```

3. After entering your password, the second tunnel will be established. Now run the CAPTCHA server on your Raspberry Pi if it is not already running.
4. Now, you can access the CAPTCHA system running on your Raspberry Pi by navigating to `http://localhost:8000` on your local machine’s web browser.   Make sure that the CAPTCHA server is running on the Raspberry Pi before trying to access it from your local machine.
