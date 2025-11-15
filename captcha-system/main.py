"""FastAPI application for CAPTCHA-protected user signup.

Provides endpoints for generating CAPTCHA challenges and validating them during signup.
"""

import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from generate import create_camouflage_text

app = FastAPI(title="Secure Bank Signup API", version="1.0.0")

captcha_storage: Dict[str, Dict] = {}
user_database: Dict[str, Dict] = {}

CAPTCHA_EXPIRY_MINUTES = 5
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
async def generate_captcha_challenge():
    """Generate a new CAPTCHA challenge.

    Returns:
        CaptchaResponse with unique ID and image URL.
    """
    cleanup_expired_captchas()

    captcha_id = str(uuid.uuid4())
    text_length = random.randint(4, 6)
    captcha_text = "".join(random.choice(SYMBOLS) for _ in range(text_length))

    image_filename = f"{captcha_id}.png"
    image_path = os.path.join(OUTPUT_DIR, image_filename)

    try:
        bg_image = get_random_image(BG_DIR)
        ov_image = get_random_image(OV_DIR)

        create_camouflage_text(
            bg_path=bg_image,
            overlay_path=ov_image,
            text=captcha_text,
            width=420,
            height=220,
            blur_radius=0.8,
            bold_amount=5,
            colorblind=False,
            difficulty=0.2,
            output_path=image_path,
        )
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
async def signup(request: SignupRequest):
    """Create a new user account after validating CAPTCHA.

    Args:
        request: SignupRequest containing user details and CAPTCHA solution.

    Returns:
        SignupResponse with user details.

    Raises:
        HTTPException: If CAPTCHA is invalid, expired, or email already exists.
    """
    cleanup_expired_captchas()

    if request.captcha_id not in captcha_storage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired CAPTCHA ID",
        )

    captcha_data = captcha_storage[request.captcha_id]

    if captcha_data["expires_at"] < datetime.now():
        del captcha_storage[request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA has expired. Please request a new one.",
        )

    captcha_data["attempts"] += 1

    if captcha_data["attempts"] > 3:
        image_path = captcha_data.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        del captcha_storage[request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new CAPTCHA.",
        )

    if request.captcha_answer.upper() != captcha_data["text"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect CAPTCHA answer"
        )

    if request.email in user_database:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    user_id = str(uuid.uuid4())
    user_database[request.email] = {
        "user_id": user_id,
        "email": request.email,
        "full_name": request.full_name,
        "password": request.password,
        "created_at": datetime.now().isoformat(),
    }

    image_path = captcha_data.get("image_path")
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    del captcha_storage[request.captcha_id]

    return SignupResponse(
        message="Account created successfully", user_id=user_id, email=request.email
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
