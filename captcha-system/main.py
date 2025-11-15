"""FastAPI application for CAPTCHA-protected user signup.

Provides endpoints for generating CAPTCHA challenges and validating them during signup.
"""

import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from generate import generate_camouflage_captcha
from pydantic import BaseModel, EmailStr, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Secure Bank Signup API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

captcha_storage: Dict[str, Dict] = {}
user_database: Dict[str, Dict] = {}
fingerprint_cache: Dict[str, datetime] = {}  # Track fingerprints to detect reuse

CAPTCHA_EXPIRY_MINUTES = 5
FINGERPRINT_EXPIRY_HOURS = 24
OUTPUT_DIR = "captcha_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
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
    captcha_answer: str
    fingerprint: str = Field(..., description="Browser fingerprint for bot detection")
    # Honeypot fields - should remain empty for legitimate users
    website: Optional[str] = None
    company: Optional[str] = None
    phone_verify: Optional[str] = None


class SignupResponse(BaseModel):
    """Response model for successful signup."""

    message: str
    user_id: str
    email: str


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


def get_random_image(directory: str) -> str:
    """Get a random image file from the specified directory."""
    images = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".avif"))
    ]
    if not images:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No images found in {directory}",
        )
    return random.choice(images)


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
            symbols_file="symbols.txt",
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

    if signup_request.captcha_answer.upper() != captcha_data["text"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect CAPTCHA answer"
        )

    if signup_request.email in user_database:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

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
