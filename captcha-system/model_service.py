"""
Standalone OCR microservice using EasyOCR.

Run this service separately, e.g.:

    uvicorn ocr_service:app --host 0.0.0.0 --port 8001 --reload

Then your main CAPTCHA backend can call:
    POST http://localhost:8001/ocr

with a base64 canvas image.
"""

import base64
import os
import uuid
from typing import List, Optional

import easyocr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pickle
import numpy as np
import uvicorn


# ---------- Pydantic models ----------

class OCRRequest(BaseModel):
    """
    Request for OCR service.
    Send the canvas image as a base64-encoded PNG string.
    Can be with or without 'data:image/png;base64,' prefix.
    """
    image_base64: str = Field(..., description="Base64-encoded PNG from canvas")


class OCRResultItem(BaseModel):
    """
    One OCR detection result (for debugging / inspection).
    """
    text: str
    confidence: Optional[float] = None
    box: Optional[List[List[float]]] = None  # bounding box


class OCRResponse(BaseModel):
    """
    Response from OCR service.
    - recognized: normalized string (joined, uppercased, no spaces)
    - raw: raw concatenated text as read by EasyOCR
    - items: individual OCR results (optional, for debugging)
    """
    recognized: Optional[str]
    raw: Optional[str]
    items: List[OCRResultItem]
    
class HumanEvalRequest(BaseModel):
    kin_vectors: list   # list of lists; each element = feature vector for one character


class HumanEvalResponse(BaseModel):
    probabilities: list  # list of floats, prob of human for each vector


# ---------- FastAPI app & EasyOCR init ----------

app = FastAPI(title="CAPTCHA OCR Service", version="1.0.0")

print("[OCR] Initializing EasyOCR reader (CPU, lang='en')...")
try:
    OCR_READER = easyocr.Reader(["en"], gpu=False)
    print("[OCR] EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"[OCR] ERROR: Failed to initialize EasyOCR: {e}")
    OCR_READER = None


HUMAN_MODEL = None

print("[HUMAN] Loading human movement model...")
try:
    with open("captcha_mouse_movement_prediction/models/mouse_movement_model.pkl", "rb") as f:
        HUMAN_MODEL = pickle.load(f)
    print("[HUMAN] Model loaded successfully.")
except Exception as e:
    HUMAN_MODEL = None
    print(f"[HUMAN] ERROR loading human movement model: {e}")


# ---------- Utility: decode base64 → temp PNG ----------

def decode_base64_to_temp_png(b64_string: str) -> str:
    """
    Convert base64 PNG (with or without data URL prefix) into
    a temporary PNG file on disk. Returns the file path.
    """
    try:
        # If it's like 'data:image/png;base64,AAAA...', split header
        if "," in b64_string:
            _, encoded = b64_string.split(",", 1)
        else:
            encoded = b64_string

        img_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image data: {e}",
        )

    tmp_name = f"tmp_ocr_{uuid.uuid4().hex}.png"
    with open(tmp_name, "wb") as f:
        f.write(img_bytes)

    return tmp_name


# ---------- OCR endpoint ----------

@app.post("/human_evaluate", response_model=HumanEvalResponse)
async def human_evaluate(req: HumanEvalRequest):
    if HUMAN_MODEL is None:
        raise HTTPException(
            status_code=500,
            detail="Human movement model not loaded."
        )

    kin_vecs = np.array(req.kin_vectors)

    try:
        probs = HUMAN_MODEL.predict_proba(kin_vecs)
        # Each prediction gives [prob_bot, prob_human]
        human_probs = [float(p[1]) for p in probs]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Human model prediction failed: {e}"
        )

    return HumanEvalResponse(probabilities=human_probs)


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(req: OCRRequest):
    """
    Perform OCR on a base64-encoded canvas image.
    Returns normalized recognized text + raw text + per-item results.
    """
    if OCR_READER is None:
        raise HTTPException(
            status_code=500,
            detail="EasyOCR not initialized on server.",
        )

    img_path = decode_base64_to_temp_png(req.image_base64)

    try:
        # EasyOCR returns a list of (bbox, text, confidence)
        results = OCR_READER.readtext(img_path, detail=True, paragraph=True)
    except Exception as e:
        # Clean up temp file before raising
        try:
            os.remove(img_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {e}",
        )

    # Clean up temp file
    try:
        os.remove(img_path)
    except OSError:
        pass

    items: List[OCRResultItem] = []
    raw_texts: List[str] = []

    for res in results:
        # res: [bbox, text, confidence]
        if len(res) < 2:
            continue
        box = res[0]
        text = res[1]
        conf = res[2] if len(res) >= 3 else None

        raw_texts.append(text)
        items.append(
            OCRResultItem(
                text=text,
                confidence=conf,
                box=box,
            )
        )

    if raw_texts:
        raw = "".join(raw_texts)
        recognized = "".join(raw.split()).upper()  # strip spaces + uppercase
    else:
        raw = None
        recognized = None

    return OCRResponse(
        recognized=recognized,
        raw=raw,
        items=items,
    )


# ---------- Health check ----------

@app.get("/health")
async def health():
    return {"status": "ok", "easyocr_initialized": OCR_READER is not None}

# ---------- Run app with Uvicorn ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
