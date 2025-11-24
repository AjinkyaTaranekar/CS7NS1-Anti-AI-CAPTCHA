"""FastAPI application for CAPTCHA-protected user signup.

Provides endpoints for generating CAPTCHA challenges and validating them during signup.
"""

import base64
import hashlib
import logging
import os
import pickle
import secrets
import uuid
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
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

# --- Logging setup (persistent logs for signup attempts & captcha events) ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "signup_attempts.log")

# Configure root logger for this module / app
logger = logging.getLogger("captcha_system")
logger.setLevel(logging.INFO)

# Rotating file handler to avoid uncontrolled log growth
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")

# Add a small filter to ensure every log record gets a trace_id attribute so
# our formatter can print a trace/correlation id even when a LoggerAdapter
# does not supply one.
class TraceFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'trace_id'):
            record.trace_id = '-'
        return True

formatter = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] [trace=%(trace_id)s] %(message)s")
file_handler.setFormatter(formatter)
logger.addFilter(TraceFilter())
logger.addHandler(file_handler)

# Also keep a console-friendly handler so devs still see messages while running
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.info("Starting CAPTCHA service — configuring logging and model load")


def get_trace_logger(trace_id: Optional[str]):
    """Return a LoggerAdapter that attaches a trace_id to each LogRecord.

    Use this where you know a captcha_id (or other correlation id) and want
    to include it in subsequent log messages so they can be correlated.
    """
    return logging.LoggerAdapter(logger, {"trace_id": trace_id if trace_id else '-'})

# Print still there for backwards compatibility (keeps tests/examples unchanged)
print("Loading Model...")
try:
    with open(f"captcha_mouse_movement_prediction/models/{MOUSE_MOVEMENT_MODEL}", "rb") as f:
        HUMAN_MODEL = pickle.load(f)

    logger.info("Model loaded successfully: %s", MOUSE_MOVEMENT_MODEL)
    print(" > Model loaded successfully.")
except FileNotFoundError:
    logger.warning("Model file not found: %s — run train_model.py to create model", MOUSE_MOVEMENT_MODEL)
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
    logger.debug("Validating fingerprint: %s", hashlib.sha1(fingerprint.encode()).hexdigest() if fingerprint else None)
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
    
    logger.info("Fingerprint validation result: is_valid=%s score=%.3f for fp_len=%d", is_valid, score, len(fingerprint) if fingerprint else 0)
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
    logger.debug("[Behavioral Analysis] Starting server-side analysis for mouse events=%d keystroke_events=%d", len(mouse_data) if mouse_data else 0, len(keystroke_data) if keystroke_data else 0)
    
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
    
    logger.info("[Behavioral Analysis] Behavioral Score: %.3f", final_score)
    logger.debug("[Behavioral Analysis] Bot indicators=%s Human indicators=%s data_quality=%s", bot_indicators, human_indicators, {'mouse_events': len(mouse_data) if mouse_data else 0, 'keystroke_events': len(keystroke_data) if keystroke_data else 0})
    
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
        logger.debug("Canvas not provided or invalid format -> returning neutral-low score")
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
        
        logger.info("Canvas complexity: %d chars -> score: %.3f", data_length, score)
        return score
    except Exception as e:
        logger.exception("Canvas analysis error")
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
    
    logger.debug("Form completion time: %.1fs since captcha creation", time_to_complete)
    
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
    logger.debug("PoW solve time: %dms", t)

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
    logger.info("[COMPREHENSIVE BOT DETECTION] Calculating final score for captcha attempts=%d", captcha_attempts)
    
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
    
    logger.debug("Individual component scores: %s", individual_scores)
    
    for component, score in individual_scores.items():
        weight = SCORING_WEIGHTS.get(component, 0.0)
        weighted_contribution = score * weight
        weighted_scores[component] = weighted_contribution
        total_weighted_score += weighted_contribution
        
        print(f"{component:30s}: {score:.3f} × {weight:.2f} = {weighted_contribution:.4f}")
    
    logger.info("FINAL WEIGHTED SCORE: %.4f", total_weighted_score)
    
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
    
    logger.info("VERDICT: %s (confidence: %s) RECOMMENDATION: %s", verdict, confidence, recommendation)
    
    logger.debug("Signals: %s", signals)
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


def verify_captcha_movement(events: list, expected_code: str, trace_id: Optional[str] = None) -> CaptchaMouseMovementVerificationResult:
    """Verify CAPTCHA based on mouse movement patterns.
    
    Args:
        events: List of mouse movement events with x, y, time, and state
        expected_code: The expected CAPTCHA code to match
        
    Returns:
        CaptchaMouseMovementVerificationResult with success status and message
    """
    tlog = get_trace_logger(trace_id)
    tlog.info("[Verify] Processing verification for expected_code=%s", expected_code)
    
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
        tlog.warning("No strokes detected in submitted events")
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
            tlog.debug("Char %d human probability=%.4f", i, prob_human)
    
    tlog.info("Recognition: expected=%s reconstructed_chars=%d human_scores_count=%d", expected_code, len(char_groups), len(human_scores))
    
    # 4. Final Decision Logic
    # Use the average human score across all characters and if 50% characters pass
    avg_human_score = sum(human_scores) / len(human_scores) if human_scores else 0
    passed_chars = sum(1 for score in human_scores if score >= 0.5) / len(human_scores) if human_scores else 0
    
    tlog.info("Average human score=%.4f passed_chars_ratio=%.2f", avg_human_score, passed_chars)
    # Strict threshold for bot detection
    if avg_human_score < 0.5 or passed_chars < 0.5:
        tlog.warning("Result: BOT DETECTED avg_human_score=%.4f passed_chars=%.2f", avg_human_score, passed_chars)
        return CaptchaMouseMovementVerificationResult(
            success=False,
            message="Automated movement detected.",
            human_score=avg_human_score
        )
    
    if 0.5 <= avg_human_score < 0.65:
        tlog.warning("Result: BORDERLINE avg_human_score=%.4f passed_chars=%.2f", avg_human_score, passed_chars)
        return CaptchaMouseMovementVerificationResult(
            success=False,
            message="Movement too smooth. Try again.",
            human_score=avg_human_score
        )
    
    tlog.info("Result: VERIFIED avg_human_score=%.4f passed_chars=%.2f", avg_human_score, passed_chars)
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
    # Increased to 4 per user's request (still relatively low but stronger than 3)
    difficulty = 4

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
        "shown_to_user": False,
        "pow_attempts": 0,
        "used_nonces": [],
        "pow_solved": False,
    }

    tlog = get_trace_logger(captcha_id)
    tlog.info("CAPTCHA generated: id=%s text_len=%d expires_at=%s image=%s", captcha_id, len(captcha_text), expires_at.isoformat(), image_path)

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

    captcha_id = filename.rsplit('.', 1)[0]
    if captcha_id not in captcha_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="CAPTCHA not found"
        )

    if captcha_storage[captcha_id]["expires_at"] < datetime.now():
        raise HTTPException(
            status_code=status.HTTP_410_GONE, detail="CAPTCHA has expired"
        )

    if captcha_storage[captcha_id]["shown_to_user"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="CAPTCHA image already served"
        )
    
    # Mark the CAPTCHA as shown to the user
    captcha_storage[captcha_id]["shown_to_user"] = True
    tlog = get_trace_logger(captcha_id)
    tlog.info("CAPTCHA image served: id=%s filename=%s", captcha_id, filename)

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

    # Establish trace logger (use captcha_id as trace/correlation id) — signup_request.captcha_id required in schema
    trace_id = getattr(signup_request, 'captcha_id', None)
    tlog = get_trace_logger(trace_id)

    # 1. Validate browser fingerprint
    fingerprint_valid, fingerprint_score = validate_fingerprint(signup_request.fingerprint)
    tlog.info("[1/8] Fingerprint validation: valid=%s score=%.3f for fp_hash=%s", fingerprint_valid, fingerprint_score, hashlib.sha1(signup_request.fingerprint.encode()).hexdigest() if signup_request.fingerprint else None)


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
    tlog.info("[PoW] verifying: combined_prefix=%s hash=%s need_prefix=%s", combined[:20], result_hash, prefix)
    if not result_hash.startswith(prefix):
        # Fail early if PoW invalid — if we have a captcha record increment pow_attempts and
        # potentially invalidate the captcha to prevent brute forcing.
        if signup_request.captcha_id in captcha_storage:
            captcha_storage[signup_request.captcha_id]["pow_attempts"] += 1
            attempts = captcha_storage[signup_request.captcha_id]["pow_attempts"]
            tlog.warning("[PoW] invalid nonce — pow_attempts=%d", attempts)
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

        tlog.warning("[PoW] invalid nonce nonce=%s", signup_request.nonce)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid proof-of-work nonce. Please solve the challenge correctly."
        )
    else:
        tlog.info("* Proof-of-Work verified nonce=%s", signup_request.nonce)
        
    # 2. Honeypot detection: if any filled, likely bot/AI
    honeypot_triggered = bool(
        (signup_request.website and signup_request.website.strip())
        or (signup_request.company and signup_request.company.strip())
        or (signup_request.phone_verify and signup_request.phone_verify.strip())
    )
    tlog.info("[2/8] Honeypot check: triggered=%s email=%s", honeypot_triggered, signup_request.email)

    # Password complexity enforcement (server-side)
    pw = signup_request.password
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_symbol = any(not c.isalnum() and not c.isspace() for c in pw)
    if not (len(pw) >= 8 and has_upper and has_lower and has_digit and has_symbol):
        tlog.warning("Password complexity violation for email=%s has_upper=%s has_lower=%s has_digit=%s has_symbol=%s len=%d", signup_request.email, has_upper, has_lower, has_digit, has_symbol, len(pw))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Password must be 8+ characters and include uppercase, lowercase, "
                "a number, and a symbol."
            ),
        )

    # 3. Check CAPTCHA validity
    if signup_request.captcha_id not in captcha_storage:
        tlog.warning("Invalid or expired CAPTCHA id provided for email=%s", signup_request.email)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired CAPTCHA ID",
        )

    captcha_data = captcha_storage[signup_request.captcha_id]

    # Prevent nonce replay and ensure challenge hasn't already been used for another signup
    if captcha_data.get("pow_solved"):
        tlog.warning("Replay attack: proof-of-work already used")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This CAPTCHA's proof-of-work has already been used. Request a new CAPTCHA."
        )

    used_nonces = captcha_data.setdefault("used_nonces", [])
    if signup_request.nonce in used_nonces:
        tlog.warning("Nonce already used nonce=%s", signup_request.nonce)
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
    tlog.info("[3/8] CAPTCHA attempt #%d email=%s", captcha_data['attempts'], signup_request.email)

    if captcha_data["attempts"] > 3:
        image_path = captcha_data.get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            tlog.info("Removed CAPTCHA image due to too many attempts: %s", image_path)
        del captcha_storage[signup_request.captcha_id]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new CAPTCHA.",
        )

    # 4. Verify CAPTCHA mouse movement patterns
    tlog.info("[4/8] Verifying CAPTCHA mouse movements")
    verification_result = verify_captcha_movement(
        events=signup_request.events,
        expected_code=captcha_data["text"],
        trace_id=trace_id
    )
    
    # 5. Analyze behavioral data (mouse & keystroke)
    tlog.info("[5/8] Analyzing behavioral patterns for email=%s", signup_request.email)
    behavioral_analysis = analyze_behavioral_data(
        mouse_data=signup_request.mouseData,
        keystroke_data=signup_request.keystrokeData
    )
    
    # 6. Analyze canvas drawing complexity
    tlog.info("[6/8] Analyzing canvas complexity",)
    canvas_score = analyze_canvas_complexity(signup_request.canvasImage)
    
    # 7. Analyze timing patterns
    tlog.info("[7/8] Analyzing timing patterns")
    captcha_created_at = captcha_data.get("created_at", datetime.now() - timedelta(minutes=1))
    timing_score = analyze_timing_patterns(signup_request, captcha_created_at)
    
    # 8. Calculate comprehensive bot detection score
    tlog.info("[8/8] Calculating comprehensive bot score for email=%s", signup_request.email)
    # Calculate PoW timing score (how long the client took to solve the puzzle)
    pow_time_score = analyze_pow_timing(signup_request.mining_time_ms)
    tlog.info("[PoW] pow_time_score=%0.3f ms=%s", pow_time_score, signup_request.mining_time_ms)

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
    tlog.info("Bot detection result: score=%0.3f verdict=%s recommendation=%s", bot_detection.final_score, bot_detection.verdict, bot_detection.recommendation)

    if bot_detection.recommendation == "BLOCK":
        tlog.warning("Blocking signup: bot detected score=%0.3f", bot_detection.final_score)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Bot detected (score: {bot_detection.final_score:.3f}). Access denied."
        )
    elif bot_detection.recommendation == "BLOCK_SOFT":
        tlog.warning("Soft-block: suspicious score=%0.3f", bot_detection.final_score)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Suspicious activity detected. Please try again with natural interactions."
        )
    elif bot_detection.recommendation == "CHALLENGE":
        tlog.warning("Additional challenge required: score=%0.3f", bot_detection.final_score)
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
        if canvas_image_path:
            tlog.info("Canvas drawing saved path=%s", canvas_image_path)
    
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
        tlog.warning("Duplicate registration attempt for email=%s", signup_request.email)
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

    tlog.info("User registered: email=%s user_id=%s final_score=%0.3f verdict=%s", signup_request.email, user_id, bot_detection.final_score, bot_detection.verdict)

    # Store fingerprint with timestamp
    fingerprint_cache[signup_request.fingerprint] = datetime.now()

    # Clean up CAPTCHA
    image_path = captcha_data.get("image_path")
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
        tlog.info("Removed CAPTCHA image after successful registration: %s", image_path)
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


@app.get("/api/logs/recent")
async def recent_logs(request: Request, lines: int = 200):
    """Return the last N lines of the persistent signup_attempts log.

    Security: requires X-Admin-Token header to match environment variable LOG_ACCESS_TOKEN.
    If LOG_ACCESS_TOKEN is not set, access is denied to avoid unintended exposure.
    """
    token = os.environ.get("LOG_ACCESS_TOKEN")
    header = request.headers.get("x-admin-token")
    if not token:
        logger.warning("Attempt to access logs but LOG_ACCESS_TOKEN not set — denying")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Log access not configured")

    if token != header:
        logger.warning("Unauthorized log access attempt from %s", request.client.host if request.client else None)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")

    # Efficiently read last lines
    try:
        with open(LOG_FILE, 'rb') as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            blocksize = 1024
            data = bytearray()
            blocks = -1
            while len(data.splitlines()) <= lines and abs(blocks * blocksize) < filesize:
                try:
                    f.seek(blocks * blocksize, os.SEEK_END)
                except OSError:
                    f.seek(0)
                    data = f.read()
                    break
                data[0:0] = f.read(blocksize)
                blocks -= 1

        content = data.decode('utf-8', errors='replace').splitlines()[-lines:]
        logger.info("Admin fetched recent %d log lines", lines)
        return {"lines": content}
    except FileNotFoundError:
        logger.warning("Log file not found when admin tried to fetch recent logs")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found")
    except Exception as e:
        logger.exception("Error reading recent logs: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error reading logs")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
