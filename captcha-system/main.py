"""FastAPI application for CAPTCHA-protected user signup.

Provides endpoints for generating CAPTCHA challenges and validating them during signup.
"""

import base64
import hashlib
import os
import pickle
import secrets
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
    """Response model for CAPTCHA challenge.

    Includes the proof-of-work challenge string and difficulty so the client
    can solve the puzzle before attempting signup.
    """

    captcha_id: str
    captcha_image_url: str
    expires_at: str
    # Server-generated PoW challenge (store in session + captcha_storage)
    challenge: str
    difficulty: int


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
    mouseData: Optional[list] = Field(default=None, description="Global mouse/touch tracking data")
    keystrokeData: Optional[list] = Field(default=None, description="Keyboard tracking data")
    canvasImage: Optional[str] = Field(default=None, description="Base64 encoded canvas drawing image")
    # nonce produced by client by solving the proof-of-work challenge
    nonce: Optional[int] = Field(default=None, description="Proof-of-work nonce")
    # mining time in milliseconds (how long it took the client to solve the PoW)
    mining_time_ms: Optional[int] = Field(default=None, description="Proof-of-work time in ms")


class SignupResponse(BaseModel):
    """Response model for successful signup."""

    message: str
    user_id: str
    email: str


class BotDetectionScore(BaseModel):
    """Comprehensive bot detection scoring result."""
    
    final_score: float = Field(..., description="Final weighted score (0-1, 1=human, 0=bot)")
    verdict: str = Field(..., description="HUMAN, SUSPICIOUS, or BOT")
    confidence: str = Field(..., description="high, medium, or low")
    individual_scores: Dict[str, float] = Field(..., description="Individual component scores")
    weighted_scores: Dict[str, float] = Field(..., description="Weighted component contributions")
    signals: dict = Field(..., description="Detailed signal information")
    recommendation: str = Field(..., description="Action recommendation")


# Scoring Configuration - Similar to Cloudflare's approach
SCORING_WEIGHTS = {
    # adjusted weights to include PoW timing as a separate signal
    'captcha_mouse_movement': 0.28,
    'behavioral_mouse': 0.145,
    'behavioral_keystroke': 0.145,
    'fingerprint_validity': 0.09,
    'honeypot': 0.09,
    'timing_analysis': 0.10,
    'canvas_analysis': 0.05,
    'rate_limit_history': 0.05,
    'pow_timing': 0.05
}

# Thresholds for final verdict
THRESHOLD_HUMAN = 0.65      # >= 0.65 = Likely human
THRESHOLD_BOT = 0.35        # <= 0.35 = Likely bot
# Between 0.35 and 0.65 = Suspicious, needs additional verification


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


def validate_fingerprint(fingerprint: str):
    """Validate that fingerprint appears to be from a real browser.
    
    Args:
        fingerprint: Browser fingerprint string
        
    Returns:
        Tuple of (is_valid, score) where score is 0-1 (1=human-like, 0=bot-like)
    """
    print(f"Validating fingerprint: {fingerprint}")
    score = 0.5  # Neutral starting point
    
    if not fingerprint or len(fingerprint) < 24:
        return False, 0.0
    
    # Length check (real fingerprints are typically longer)
    if len(fingerprint) >= 32:
        score += 0.2
    elif len(fingerprint) >= 28:
        score += 0.1
    
    # Check for entropy (diverse characters suggest real fingerprint)
    unique_chars = len(set(fingerprint))
    if unique_chars > 20:
        score += 0.2
    elif unique_chars > 15:
        score += 0.1
    
    # Check if fingerprint has been seen too recently (potential bot reusing fingerprints)
    if fingerprint in fingerprint_cache:
        time_since_last_use = datetime.now() - fingerprint_cache[fingerprint]
        print(f"Fingerprint last used {time_since_last_use.total_seconds()} seconds ago")
        if time_since_last_use < timedelta(seconds=10):
            score -= 0.4  # Heavy penalty for rapid reuse
        elif time_since_last_use < timedelta(minutes=1):
            score -= 0.2  # Moderate penalty
    else:
        score += 0.1  # First-time fingerprint is slightly positive
    
    # Normalize to 0-1 range
    score = max(0.0, min(1.0, score))
    is_valid = score >= 0.3
    
    return is_valid, score


def analyze_behavioral_data(mouse_data: Optional[list], 
                           keystroke_data: Optional[list]) -> dict:
    """Analyze behavioral data and return individual scores.
    
    Args:
        mouse_data: Global mouse/touch tracking data
        keystroke_data: Keyboard tracking data
        
    Returns:
        Dictionary with mouse_score, keystroke_score (0-1), and detailed metrics
    """
    print("\n[Behavioral Analysis] Starting server-side analysis...")
    
    bot_indicators = []
    human_indicators = []
    final_score = 50.0  # Neutral starting point
    
    # Server-side mouse data analysis
    if mouse_data and len(mouse_data) > 20:
        # Calculate entropy in mouse movements
        positions = [(m['x'], m['y']) for m in mouse_data if 'x' in m and 'y' in m]
        
        if len(positions) > 10:
            # Check for unique positions (humans have more variation)
            unique_positions = len(set(positions))
            uniqueness_ratio = unique_positions / len(positions)
            
            print(f" > Mouse uniqueness ratio: {uniqueness_ratio:.2f}")
            
            if uniqueness_ratio < 0.3:
                bot_indicators.append("Low mouse movement variation")
                final_score -= 15
            elif uniqueness_ratio > 0.7:
                human_indicators.append("High mouse movement variation")
                final_score += 10
            
            # Check time distribution
            timestamps = [m['timestamp'] for m in mouse_data if 'timestamp' in m]
            if len(timestamps) > 10:
                time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_time_diff = sum(time_diffs) / len(time_diffs)
                
                # Bots have very consistent timing
                if avg_time_diff < 5:
                    bot_indicators.append("Extremely fast mouse updates")
                    final_score -= 10
    else:
        bot_indicators.append("Insufficient mouse tracking data")
        final_score -= 10
    
    # Server-side keystroke data analysis
    if keystroke_data and len(keystroke_data) > 10:
        # Calculate typing rhythm
        keydown_events = [k for k in keystroke_data if k.get('type') == 'down']
        
        if len(keydown_events) > 5:
            timestamps = [k['timestamp'] for k in keydown_events]
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                std_dev = (sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)) ** 0.5
                
                print(f" > Keystroke avg interval: {avg_interval:.1f}ms, std dev: {std_dev:.1f}ms")
                
                # Bots have very low standard deviation
                if std_dev < 10 and len(intervals) > 10:
                    bot_indicators.append("Suspiciously consistent typing rhythm")
                    final_score -= 20
                elif std_dev > 30:
                    human_indicators.append("Natural typing rhythm variation")
                    final_score += 10
                
                # Extremely fast typing
                if avg_interval < 50:
                    bot_indicators.append(f"Impossibly fast typing: {avg_interval:.1f}ms")
                    final_score -= 15
    else:
        bot_indicators.append("Insufficient keystroke data")
        final_score -= 5
    
    # Normalize score to 0-1 range (0=bot, 1=human)
    final_score = max(0, min(100, final_score)) / 100.0
    
    print(f"\n[Behavioral Analysis] Behavioral Score: {final_score:.3f}")
    print(f" > Bot indicators: {bot_indicators}")
    print(f" > Human indicators: {human_indicators}")
    
    # Split into mouse and keystroke scores for weighted system
    mouse_score = 0.5
    if mouse_data and len(mouse_data) > 20:
        mouse_score = final_score  # Use combined score for mouse if available
    
    keystroke_score = 0.5
    if keystroke_data and len(keystroke_data) > 10:
        keystroke_score = final_score  # Use combined score for keystroke if available
    
    return {
        'mouse_score': mouse_score,
        'keystroke_score': keystroke_score,
        'bot_indicators': bot_indicators,
        'human_indicators': human_indicators,
        'data_quality': {
            'mouse_events': len(mouse_data) if mouse_data else 0,
            'keystroke_events': len(keystroke_data) if keystroke_data else 0
        }
    }


def analyze_canvas_complexity(canvas_image: Optional[str]) -> float:
    """Analyze canvas drawing complexity to detect bot-like simplicity.
    
    Args:
        canvas_image: Base64 encoded canvas image
        
    Returns:
        Score from 0-1 (1=complex/human-like, 0=simple/bot-like)
    """
    if not canvas_image or not canvas_image.startswith('data:image'):
        return 0.3  # Neutral-low score if no canvas data
    
    try:
        # Extract base64 data
        image_data = canvas_image.split(',')[1]
        
        # Simple heuristic: longer data = more complex drawing
        data_length = len(image_data)
        
        # Typical range: 5000-50000 characters for human drawings
        if data_length < 2000:
            score = 0.1  # Too simple, likely bot
        elif data_length < 5000:
            score = 0.4
        elif data_length < 15000:
            score = 0.7
        elif data_length < 50000:
            score = 0.9
        else:
            score = 1.0  # Very complex
        
        print(f" > Canvas complexity: {data_length} chars -> score: {score:.3f}")
        return score
    except Exception as e:
        print(f" ! Canvas analysis error: {e}")
        return 0.3


def analyze_timing_patterns(signup_request: SignupRequest, captcha_created_at: datetime) -> float:
    """Analyze timing patterns to detect bot-like behavior.
    
    Args:
        signup_request: The signup request data
        captcha_created_at: When CAPTCHA was created
        
    Returns:
        Score from 0-1 (1=human-like timing, 0=bot-like timing)
    """
    score = 0.5
    
    # Check form completion time
    time_to_complete = (datetime.now() - captcha_created_at).total_seconds()
    
    print(f" > Form completion time: {time_to_complete:.1f}s")
    
    # Humans typically take 10-120 seconds to complete CAPTCHA + form
    if time_to_complete < 3:
        score = 0.0  # Impossibly fast = bot
    elif time_to_complete < 5:
        score = 0.2  # Very fast = likely bot
    elif time_to_complete < 10:
        score = 0.5  # Fast = suspicious
    elif time_to_complete < 120:
        score = 1.0  # Normal human range
    else:
        score = 0.7  # Very slow = possibly legitimate but unusual
    
    return score


def analyze_pow_timing(mining_time_ms: Optional[int]) -> float:
    """Analyze the time it took the client to solve the PoW puzzle.

    Returns a score 0-1 where higher = human-like (longer solve times), lower = suspiciously fast.
    Very fast completion times indicate a powerful machine or pre-computed answer.
    """
    if mining_time_ms is None:
        # No data — give neutral-low score
        return 0.5

    t = int(mining_time_ms)
    print(f" > PoW solve time: {t}ms")

    # Scoring heuristic:
    # - <= 200ms: extremely fast -> likely powerful machine (score low)
    # - 200-1000ms: fast -> suspicious
    # - 1000-5000ms: typical quick human-ish
    # - 5s-30s: normal human solve durations for this difficulty -> best score
    # - >30s: slower than normal but still human-like
    if t <= 200:
        return 0.1
    if t <= 1000:
        return 0.3
    if t <= 5000:
        return 0.6
    if t <= 30000:
        return 1.0
    return 0.8


def calculate_comprehensive_bot_score(
    captcha_result: CaptchaMouseMovementVerificationResult,
    behavioral_analysis: dict,
    fingerprint_score: float,
    honeypot_triggered: bool,
    timing_score: float,
    canvas_score: float,
    captcha_attempts: int,
    pow_time_score: float = 0.5,
    pow_time_ms: Optional[int] = None
) -> BotDetectionScore:
    """Calculate final bot detection score using weighted approach similar to Cloudflare.
    
    Args:
        captcha_result: CAPTCHA verification result with human score
        behavioral_analysis: Behavioral data analysis results
        fingerprint_score: Browser fingerprint validation score
        honeypot_triggered: Whether honeypot fields were filled
        timing_score: Timing pattern analysis score
        canvas_score: Canvas drawing complexity score
        captcha_attempts: Number of CAPTCHA attempts
        
    Returns:
        BotDetectionScore with comprehensive analysis
    """
    print("\n" + "="*80)
    print("[COMPREHENSIVE BOT DETECTION] Calculating Final Score")
    print("="*80)
    
    # Individual component scores (0-1, where 1=human, 0=bot)
    individual_scores = {
        'captcha_mouse_movement': captcha_result.human_score if captcha_result.human_score else 0.0,
        'behavioral_mouse': behavioral_analysis['mouse_score'],
        'behavioral_keystroke': behavioral_analysis['keystroke_score'],
        'fingerprint_validity': fingerprint_score,
        'honeypot': 0.0 if honeypot_triggered else 1.0,
        'timing_analysis': timing_score,
        'canvas_analysis': canvas_score,
        'pow_timing': pow_time_score,
        'rate_limit_history': 1.0 - (captcha_attempts * 0.2)  # Penalize multiple attempts
    }
    
    # Calculate weighted contributions
    weighted_scores = {}
    total_weighted_score = 0.0
    
    print("\nIndividual Component Scores:")
    print("-" * 80)
    
    for component, score in individual_scores.items():
        weight = SCORING_WEIGHTS.get(component, 0.0)
        weighted_contribution = score * weight
        weighted_scores[component] = weighted_contribution
        total_weighted_score += weighted_contribution
        
        print(f"{component:30s}: {score:.3f} × {weight:.2f} = {weighted_contribution:.4f}")
    
    print("-" * 80)
    print(f"{'FINAL WEIGHTED SCORE':30s}: {total_weighted_score:.4f}")
    
    # Determine verdict based on thresholds
    if total_weighted_score >= THRESHOLD_HUMAN:
        verdict = "HUMAN"
        recommendation = "ALLOW"
        confidence = "high" if total_weighted_score >= 0.80 else "medium"
    elif total_weighted_score <= THRESHOLD_BOT:
        verdict = "BOT"
        recommendation = "BLOCK"
        confidence = "high" if total_weighted_score <= 0.25 else "medium"
    else:
        verdict = "SUSPICIOUS"
        recommendation = "CHALLENGE" if total_weighted_score > 0.50 else "BLOCK_SOFT"
        confidence = "low"
    
    # Collect detailed signals
    signals = {
        'captcha_verification': captcha_result.success,
        'captcha_human_score': captcha_result.human_score,
        'behavioral_indicators': {
            'bot_flags': behavioral_analysis['bot_indicators'],
            'human_flags': behavioral_analysis['human_indicators'],
            'data_quality': behavioral_analysis['data_quality']
        },
        'fingerprint_valid': fingerprint_score > 0.3,
        'honeypot_triggered': honeypot_triggered,
        'timing_seconds': timing_score,
        'canvas_provided': canvas_score > 0.3,
        'pow_time_ms': pow_time_ms,
        'pow_time_score': pow_time_score,
        'attempt_count': captcha_attempts
    }
    
    print(f"\n{'VERDICT':30s}: {verdict} (confidence: {confidence})")
    print(f"{'RECOMMENDATION':30s}: {recommendation}")
    print("="*80 + "\n")
    
    return BotDetectionScore(
        final_score=total_weighted_score,
        verdict=verdict,
        confidence=confidence,
        individual_scores=individual_scores,
        weighted_scores=weighted_scores,
        signals=signals,
        recommendation=recommendation
    )


def save_canvas_image(canvas_image: str, captcha_id: str) -> Optional[str]:
    """Save the canvas drawing image to disk.
    
    Args:
        canvas_image: Base64 encoded image data
        captcha_id: CAPTCHA ID for filename
        
    Returns:
        Path to saved image or None if failed
    """
    try:
        if not canvas_image or not canvas_image.startswith('data:image'):
            return None
        
        # Extract base64 data
        image_data = canvas_image.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Save to file
        drawings_dir = os.path.join(OUTPUT_DIR, 'drawings')
        os.makedirs(drawings_dir, exist_ok=True)
        
        image_path = os.path.join(drawings_dir, f"{captcha_id}_drawing.png")
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f" > Canvas drawing saved: {image_path}")
        return image_path
    except Exception as e:
        print(f" ! Failed to save canvas image: {e}")
        return None


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

    # Proof-of-Work challenge token — store server-side and return to client
    challenge = secrets.token_hex(8)
    # Difficulty = number of leading hex '0' characters required in SHA256 hex
    # Increased to 5 per user's request (still relatively low but stronger than 3)
    difficulty = 5

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

    created_at = datetime.now()
    expires_at = created_at + timedelta(minutes=CAPTCHA_EXPIRY_MINUTES)

    captcha_storage[captcha_id] = {
        "text": captcha_text.upper(),
        "created_at": created_at,
        "expires_at": expires_at,
        "image_path": image_path,
        "attempts": 0,
        # used to track proof-of-work usage and prevent replay
        "pow_challenge": challenge,
        "difficulty": difficulty,
        "pow_attempts": 0,
        "used_nonces": [],
        "pow_solved": False,
    }

    # Save the challenge also in user's session so the client-server pair
    # has an additional reference point (developer requested behavior)
    try:
        # request.session is provided by SessionMiddleware
        request.session["challenge"] = challenge
        request.session["captcha_id"] = captcha_id
        request.session["difficulty"] = difficulty
    except Exception:
        # Not fatal, but keep generation robust
        pass

    return CaptchaResponse(
        captcha_id=captcha_id,
        captcha_image_url=f"/captcha/{image_filename}",
        expires_at=expires_at.isoformat(),
        challenge=challenge,
        difficulty=difficulty,
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

    # 1. Validate browser fingerprint
    fingerprint_valid, fingerprint_score = validate_fingerprint(signup_request.fingerprint)
    print(f"[1/8] Fingerprint validation: valid={fingerprint_valid}, score={fingerprint_score:.3f}")


    # 1b. Verify proof-of-work: ensure client solved server challenge
    if signup_request.nonce is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing proof-of-work nonce. Please solve the challenge before signing up."
        )

    # Prefer per-captcha pow_challenge stored in captcha_storage, fallback to session
    pow_challenge = None
    pow_difficulty = None
    if signup_request.captcha_id in captcha_storage:
        pow_challenge = captcha_storage[signup_request.captcha_id].get("pow_challenge")
        pow_difficulty = captcha_storage[signup_request.captcha_id].get("difficulty", 3)
    else:
        pow_challenge = request.session.get("challenge")
        pow_difficulty = request.session.get("difficulty", 3)

    if not pow_challenge:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Server-side challenge not found. Please request a new CAPTCHA and try again."
        )

    # Verify hash
    combined = f"{pow_challenge}{signup_request.nonce}"
    result_hash = hashlib.sha256(combined.encode()).hexdigest()
    prefix = "0" * int(pow_difficulty)
    print(f"[PoW] verifying: combined='{combined[:20]}...' hash={result_hash} need_prefix='{prefix}'")
    if not result_hash.startswith(prefix):
        # Fail early if PoW invalid — if we have a captcha record increment pow_attempts and
        # potentially invalidate the captcha to prevent brute forcing.
        if signup_request.captcha_id in captcha_storage:
            captcha_storage[signup_request.captcha_id]["pow_attempts"] += 1
            attempts = captcha_storage[signup_request.captcha_id]["pow_attempts"]
            print(f"[PoW] invalid nonce — pow_attempts={attempts}")
            # Invalidate after N bad PoW attempts
            if attempts > 10:
                image_path = captcha_storage[signup_request.captcha_id].get("image_path")
                if image_path and os.path.exists(image_path):
                    os.remove(image_path)
                del captcha_storage[signup_request.captcha_id]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Too many invalid proof attempts — CAPTCHA invalidated. Please request a new one."
                )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid proof-of-work nonce. Please solve the challenge correctly."
        )
    else:
        print(" * Proof-of-Work verified")
        
    # 2. Honeypot detection: if any filled, likely bot/AI
    honeypot_triggered = bool(
        (signup_request.website and signup_request.website.strip())
        or (signup_request.company and signup_request.company.strip())
        or (signup_request.phone_verify and signup_request.phone_verify.strip())
    )
    print(f"[2/8] Honeypot check: triggered={honeypot_triggered}")

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

    # 3. Check CAPTCHA validity
    if signup_request.captcha_id not in captcha_storage:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired CAPTCHA ID",
        )

    captcha_data = captcha_storage[signup_request.captcha_id]

    # Prevent nonce replay and ensure challenge hasn't already been used for another signup
    if captcha_data.get("pow_solved"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This CAPTCHA's proof-of-work has already been used. Request a new CAPTCHA."
        )

    used_nonces = captcha_data.setdefault("used_nonces", [])
    if signup_request.nonce in used_nonces:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nonce already used for this CAPTCHA. Please request a new one."
        )

    # mark nonce used immediately (single-use protection)
    used_nonces.append(signup_request.nonce)
    captcha_data["pow_solved"] = True

    if captcha_data["expires_at"] < datetime.now():
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CAPTCHA has expired. Please request a new one.",
        )

    captcha_data["attempts"] += 1
    print(f"[3/8] CAPTCHA attempt #{captcha_data['attempts']}")

    if captcha_data["attempts"] > 3:
        image_path = captcha_data.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new CAPTCHA.",
        )

    # 4. Verify CAPTCHA mouse movement patterns
    print(f"[4/8] Verifying CAPTCHA mouse movements...")
    verification_result = verify_captcha_movement(
        events=signup_request.events,
        expected_code=captcha_data["text"]
    )
    
    # 5. Analyze behavioral data (mouse & keystroke)
    print(f"[5/8] Analyzing behavioral patterns...")
    behavioral_analysis = analyze_behavioral_data(
        mouse_data=signup_request.mouseData,
        keystroke_data=signup_request.keystrokeData
    )
    
    # 6. Analyze canvas drawing complexity
    print(f"[6/8] Analyzing canvas complexity...")
    canvas_score = analyze_canvas_complexity(signup_request.canvasImage)
    
    # 7. Analyze timing patterns
    print(f"[7/8] Analyzing timing patterns...")
    captcha_created_at = captcha_data.get("created_at", datetime.now() - timedelta(minutes=1))
    timing_score = analyze_timing_patterns(signup_request, captcha_created_at)
    
    # 8. Calculate comprehensive bot detection score
    print(f"[8/8] Calculating comprehensive bot score...")
    # Calculate PoW timing score (how long the client took to solve the puzzle)
    pow_time_score = analyze_pow_timing(signup_request.mining_time_ms)
    print(f"[PoW] pow_time_score={pow_time_score:.3f}")

    bot_detection = calculate_comprehensive_bot_score(
        captcha_result=verification_result,
        behavioral_analysis=behavioral_analysis,
        fingerprint_score=fingerprint_score,
        honeypot_triggered=honeypot_triggered,
        timing_score=timing_score,
        canvas_score=canvas_score,
        captcha_attempts=captcha_data["attempts"],
        pow_time_score=pow_time_score,
        pow_time_ms=signup_request.mining_time_ms,
    )
    
    # Make decision based on comprehensive score
    if bot_detection.recommendation == "BLOCK":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bot detected (score: {bot_detection.final_score:.3f}). Access denied."
        )
    elif bot_detection.recommendation == "BLOCK_SOFT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Suspicious activity detected. Please try again with natural interactions."
        )
    elif bot_detection.recommendation == "CHALLENGE":
        # Issue additional challenge - frontend will generate new CAPTCHA
        print(f" ⚠ CHALLENGE: Suspicious score {bot_detection.final_score:.3f} - requiring additional verification")
        # Clean up current CAPTCHA before issuing new one
        image_path = captcha_data.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail="Additional verification required. Please complete a new CAPTCHA challenge."
        )
    
    # Save canvas drawing image
    canvas_image_path = None
    if signup_request.canvasImage:
        canvas_image_path = save_canvas_image(
            signup_request.canvasImage, 
            signup_request.captcha_id
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

    # Create new user with comprehensive bot detection metadata
    user_id = str(uuid.uuid4())
    user_database[signup_request.email] = {
        "user_id": user_id,
        "email": signup_request.email,
        "full_name": signup_request.full_name,
        "password": signup_request.password,
        "created_at": datetime.now().isoformat(),
        "bot_detection": {
            "final_score": bot_detection.final_score,
            "verdict": bot_detection.verdict,
            "confidence": bot_detection.confidence,
            "individual_scores": bot_detection.individual_scores,
            "recommendation": bot_detection.recommendation
        },
        "canvas_image_path": canvas_image_path
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
