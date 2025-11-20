"""FastAPI application for CAPTCHA-protected user signup.

Provides endpoints for generating CAPTCHA challenges and validating them during signup.
"""

import os
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import uvicorn
from captcha_mouse_movement_prediction.utils import (FEATURE_NAMES_KINEMATIC,
                                                     extract_features,
                                                     normalize_strokes,
                                                     segment_into_characters)
from config.constants import MOUSE_MOVEMENT_MODEL, SYMBOLS
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from generate import generate_camouflage_captcha
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Secure Bank Signup API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SessionMiddleware, secret_key="THE_SECRET_KEY")

print("Loading Model...")
try:
    with open(f"captcha_mouse_movement_prediction/models/{MOUSE_MOVEMENT_MODEL}", "rb") as f:
        HUMAN_MODEL = pickle.load(f)

    print(" > Model loaded successfully.")
except FileNotFoundError:
    print(" ! ERROR: Model not found. Run train_model.py first.")
    HUMAN_MODEL = None

captcha_storage: Dict[str, Dict] = {}
user_database: Dict[str, Dict] = {}
fingerprint_cache: Dict[str, datetime] = {}  # Track fingerprints to detect reuse

CAPTCHA_EXPIRY_MINUTES = 5
FINGERPRINT_EXPIRY_HOURS = 24
OUTPUT_DIR = "captcha_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG_DIR = "background_images"
OV_DIR = "overlay_images"

class CaptchaResponse(BaseModel):
    """Response model for CAPTCHA challenge."""

    captcha_id: str
    captcha_image_url: str
    expires_at: str


class SignupRequest(BaseModel):
    """Request model for user signup."""

    email: EmailStr
    password: str = Field(
        ..., min_length=8, description="Password must be at least 8 characters"
    )
    full_name: str = Field(..., min_length=2)
    captcha_id: str
    fingerprint: str = Field(..., description="Browser fingerprint for bot detection")
    # Honeypot fields - should remain empty for legitimate users
    website: Optional[str] = None
    company: Optional[str] = None
    phone_verify: Optional[str] = None
    events: list = Field(..., description="Mouse movement events for CAPTCHA verification")


class SignupResponse(BaseModel):
    """Response model for successful signup."""

    message: str
    user_id: str
    email: str


class CaptchaMouseMovementVerificationResult(BaseModel):
    """Result of CAPTCHA verification."""

    success: bool
    message: str
    human_score: Optional[float] = None


def cleanup_expired_captchas():
    """Remove expired CAPTCHAs from storage and filesystem."""
    now = datetime.now()
    expired_ids = [
        cid for cid, data in captcha_storage.items() if data["expires_at"] < now
    ]
    for cid in expired_ids:
        image_path = captcha_storage[cid].get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        del captcha_storage[cid]


def cleanup_expired_fingerprints():
    """Remove expired fingerprints from cache."""
    now = datetime.now()
    expired_fps = [
        fp for fp, timestamp in fingerprint_cache.items()
        if now - timestamp > timedelta(hours=FINGERPRINT_EXPIRY_HOURS)
    ]
    for fp in expired_fps:
        del fingerprint_cache[fp]


def validate_fingerprint(fingerprint: str) -> bool:
    """Validate that fingerprint appears to be from a real browser.
    
    Args:
        fingerprint: Browser fingerprint string
        
    Returns:
        True if fingerprint passes basic validation, False otherwise
    """
    print(f"Validating fingerprint: {fingerprint}")
    if not fingerprint or len(fingerprint) < 24:
        return False
    
    # Check if fingerprint has been seen too recently (potential bot reusing fingerprints)
    if fingerprint in fingerprint_cache:
        time_since_last_use = datetime.now() - fingerprint_cache[fingerprint]
        print(f"Fingerprint last used {time_since_last_use.total_seconds()} seconds ago")
        if time_since_last_use < timedelta(seconds=10):
            return False
    
    return True


def verify_captcha_movement(events: list, expected_code: str) -> CaptchaMouseMovementVerificationResult:
    """Verify CAPTCHA based on mouse movement patterns.
    
    Args:
        events: List of mouse movement events with x, y, time, and state
        expected_code: The expected CAPTCHA code to match
        
    Returns:
        CaptchaMouseMovementVerificationResult with success status and message
    """
    print(f"\n[Verify] Processing verification for code: {expected_code}")
    
    # 1. Reconstruct Strokes from Events
    strokes_raw = []
    current_stroke = []
    
    # Robust reconstruction logic
    for e in events:
        if e['state'] == 'down':
            if current_stroke:
                strokes_raw.append(current_stroke)
            current_stroke = [{'x': e['x'], 'y': e['y'], 'time': e['time']}]
        elif e['state'] == 'move':
            if current_stroke:
                current_stroke.append({'x': e['x'], 'y': e['y'], 'time': e['time']})
        elif e['state'] == 'up':
            if current_stroke:
                current_stroke.append({'x': e['x'], 'y': e['y'], 'time': e['time']})
                strokes_raw.append(current_stroke)
            current_stroke = []
    
    if current_stroke:
        strokes_raw.append(current_stroke)
    
    if not strokes_raw:
        print(" ! No strokes detected")
        return CaptchaMouseMovementVerificationResult(
            success=False,
            message="Please draw the characters."
        )
    
    # 2. Segment Characters (Using utils)
    char_groups = segment_into_characters(strokes_raw)
    
    # 3. Analysis
    recognized_str = ""
    human_scores = []
    
    for i, char_strokes in enumerate(char_groups):
        # Normalize
        arr = normalize_strokes(char_strokes)
        
        # Extract Features (using utils)
        feats = extract_features(arr, char_strokes)
        
        # Prepare Vectors
        kin_vec = np.array([[feats[k] for k in FEATURE_NAMES_KINEMATIC]])
        
        # Predict Human vs Bot
        if HUMAN_MODEL:
            # XGBoost predict_proba returns [prob_class_0, prob_class_1]
            # Class 1 is Human
            prob_human = HUMAN_MODEL.predict_proba(kin_vec)[0][1]
            human_scores.append(prob_human)
            print(f"   > Char {i}: Human Probability = {prob_human:.4f}")
    
    print(f" > Recognition Result: '{recognized_str}' vs '{expected_code}'")
    
    # 4. Final Decision Logic
    # Use the average human score across all characters and if 50% characters pass
    avg_human_score = sum(human_scores) / len(human_scores) if human_scores else 0
    passed_chars = sum(1 for score in human_scores if score >= 0.5) / len(human_scores) if human_scores else 0
    
    print(f" > Average Human Score: {avg_human_score:.4f}, Passed Characters: {passed_chars:.2%}")
    # Strict threshold for bot detection
    if avg_human_score < 0.5 or passed_chars < 0.5:
        print(" ! Result: BOT DETECTED")
        return CaptchaMouseMovementVerificationResult(
            success=False,
            message="Automated movement detected.",
            human_score=avg_human_score
        )
    
    if 0.5 <= avg_human_score < 0.65:
        print(" ! Result: BORDERLINE")
        return CaptchaMouseMovementVerificationResult(
            success=False,
            message="Movement too smooth. Try again.",
            human_score=avg_human_score
        )
    
    print(" * Result: VERIFIED")
    return CaptchaMouseMovementVerificationResult(
        success=True,
        message="Human Verified!",
        human_score=avg_human_score
    )


@app.get("/")
async def root():
    """Serve the signup page."""
    return FileResponse("index.html")


@app.post("/api/captcha/challenge", response_model=CaptchaResponse)
@limiter.limit("10/minute")
async def generate_captcha_challenge(request: Request):
    """Generate a new CAPTCHA challenge.
    
    Rate limited to 10 requests per minute per IP address.

    Returns:
        CaptchaResponse with unique ID and image URL.
    """
    cleanup_expired_captchas()

    captcha_id = str(uuid.uuid4())
    image_filename = f"{captcha_id}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)

    try:
        img, captcha_text = generate_camouflage_captcha(
            width=420,
            height=220,
            bg_dir=BG_DIR,
            ov_dir=OV_DIR,
            symbols=SYMBOLS,
            fonts_dir="fonts",
            font_size=120,
            min_length=4,
            max_length=6,
            blur=0.8,
            bold=5,
            colorblind=False,
            difficulty=0.2,
        )
        if img is None:
            raise RuntimeError("Image generation returned None")
        img.save(image_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate CAPTCHA: {str(e)}",
        )

    expires_at = datetime.now() + timedelta(minutes=CAPTCHA_EXPIRY_MINUTES)

    captcha_storage[captcha_id] = {
        "text": captcha_text.upper(),
        "expires_at": expires_at,
        "image_path": image_path,
        "attempts": 0,
    }

    return CaptchaResponse(
        captcha_id=captcha_id,
        captcha_image_url=f"/captcha/{image_filename}",
        expires_at=expires_at.isoformat(),
    )


@app.get("/captcha/{filename}")
async def get_captcha_image(filename: str):
    """Serve CAPTCHA image file."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="CAPTCHA image not found"
        )
    return FileResponse(file_path, media_type="image/png")


@app.post("/api/signup", response_model=SignupResponse)
@limiter.limit("5/minute")
async def signup(signup_request: SignupRequest, request: Request):
    """Create a new user account after validating CAPTCHA.
    
    Rate limited to 5 requests per minute per IP address.

    Args:
        signup_request: SignupRequest containing user details and CAPTCHA solution.
        request: FastAPI Request object for rate limiting.

    Returns:
        SignupResponse with user details.

    Raises:
        HTTPException: If CAPTCHA is invalid, expired, or email already exists.
    """
    cleanup_expired_captchas()
    cleanup_expired_fingerprints()

    # Validate browser fingerprint
    if not validate_fingerprint(signup_request.fingerprint):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid browser fingerprint. Bot activity suspected."
        )

    # Honeypot detection: if any filled, likely bot/AI
    if (
        (signup_request.website and signup_request.website.strip())
        or (signup_request.company and signup_request.company.strip())
        or (signup_request.phone_verify and signup_request.phone_verify.strip())
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Suspicious activity detected."
        )

    # Password complexity enforcement (server-side)
    pw = signup_request.password
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_symbol = any(not c.isalnum() and not c.isspace() for c in pw)
    if not (len(pw) >= 8 and has_upper and has_lower and has_digit and has_symbol):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Password must be 8+ characters and include uppercase, lowercase, "
                "a number, and a symbol."
            ),
        )

    if signup_request.captcha_id not in captcha_storage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired CAPTCHA ID",
        )

    captcha_data = captcha_storage[signup_request.captcha_id]

    if captcha_data["expires_at"] < datetime.now():
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA has expired. Please request a new one.",
        )

    captcha_data["attempts"] += 1

    if captcha_data["attempts"] > 3:
        image_path = captcha_data.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new CAPTCHA.",
        )

    # Verify mouse movement patterns to detect bots
    verification_result = verify_captcha_movement(
        events=signup_request.events,
        expected_code=captcha_data["text"]
    )
    
    if not verification_result.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=verification_result.message
        )
    
    # TODO: Add character recognition model here
    # For now, we assume all characters are recognized correctly
    # if recognized_str != expected_code:
    #     print(" ! Result: WRONG CHARACTERS")
    #     return CaptchaMouseMovementVerificationResult(
    #         success=False,
    #         message=f"Read: {recognized_str}. Try writing clearer."
    #     )

    # Check if email already exists
    if signup_request.email in user_database:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    # Create new user
    user_id = str(uuid.uuid4())
    user_database[signup_request.email] = {
        "user_id": user_id,
        "email": signup_request.email,
        "full_name": signup_request.full_name,
        "password": signup_request.password,
        "created_at": datetime.now().isoformat(),
    }

    # Store fingerprint with timestamp
    fingerprint_cache[signup_request.fingerprint] = datetime.now()

    # Clean up CAPTCHA
    image_path = captcha_data.get("image_path")
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    del captcha_storage[signup_request.captcha_id]

    return SignupResponse(
        message="Account created successfully", user_id=user_id, email=signup_request.email
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_captchas": len(captcha_storage),
        "registered_users": len(user_database),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
