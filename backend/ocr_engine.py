"""OCR Engine — PaddleOCR-primary with EasyOCR and Tesseract fallbacks.

Cascade:
  1. PaddleOCR  (deep-learning detection + recognition, handles rotation)
  2. EasyOCR    (second DL engine; activates only if PaddleOCR unavailable)
  3. Tesseract  (legacy LSTM engine; last resort)

Public API
----------
  ocr_image(image, file_hash=None, page_num=0) -> (str, float)
      Run OCR on a PIL.Image.  Results are disk-cached by (sha256, page_num).
  compute_file_hash(content: bytes) -> str
      SHA-256 hex digest of raw file bytes; use as cache key.
  clean_ocr_text(text: str) -> str
      Standalone cleaner; normalise / filter OCR output.
"""

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_VECTOR_DATA_DIR = Path(os.getenv("VECTOR_DATA_DIR", "vector_data"))
OCR_CACHE_PATH   = _VECTOR_DATA_DIR / "ocr_cache.json"

_cache: dict          = {}
_cache_loaded: bool   = False
_cache_lock           = threading.Lock()


def _load_cache() -> dict:
    if OCR_CACHE_PATH.exists():
        try:
            return json.loads(OCR_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("OCR cache unreadable, starting fresh: %s", exc)
    return {}


def _save_cache(cache: dict) -> None:
    try:
        OCR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        OCR_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("OCR cache write failed: %s", exc)


def _get_cache() -> dict:
    global _cache, _cache_loaded
    if not _cache_loaded:
        with _cache_lock:
            if not _cache_loaded:
                _cache = _load_cache()
                _cache_loaded = True
    return _cache


# ---------------------------------------------------------------------------
# Engine lazy initialisation (thread-safe)
# ---------------------------------------------------------------------------

_INIT_LOCK   = threading.Lock()
_paddle_ocr  = None   # None = not yet tried; False = unavailable
_easy_ocr    = None   # None = not yet tried; False = unavailable


def _get_paddle_ocr():
    """Return a PaddleOCR instance, or None if unavailable."""
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr if _paddle_ocr is not False else None

    with _INIT_LOCK:
        if _paddle_ocr is not None:
            return _paddle_ocr if _paddle_ocr is not False else None
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415

            instance = PaddleOCR(
                use_angle_cls=True,   # handles rotated / skewed pages
                lang="en",
                use_gpu=False,
                show_log=False,
                enable_mkldnn=True,   # Intel MKL-DNN CPU optimisation
            )
            _paddle_ocr = instance
            logger.info("PaddleOCR initialised (primary engine)")
        except ImportError:
            logger.warning("paddleocr not installed — falling back to EasyOCR / Tesseract")
            _paddle_ocr = False
        except Exception as exc:
            logger.error("PaddleOCR init failed: %s", exc)
            _paddle_ocr = False

    return _paddle_ocr if _paddle_ocr is not False else None


def _get_easy_ocr():
    """Return an EasyOCR Reader, or None if unavailable."""
    global _easy_ocr
    if _easy_ocr is not None:
        return _easy_ocr if _easy_ocr is not False else None

    with _INIT_LOCK:
        if _easy_ocr is not None:
            return _easy_ocr if _easy_ocr is not False else None
        try:
            import easyocr  # noqa: PLC0415

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _easy_ocr = reader
            logger.info("EasyOCR initialised (secondary engine)")
        except ImportError:
            logger.warning("easyocr not installed — Tesseract is the last fallback")
            _easy_ocr = False
        except Exception as exc:
            logger.error("EasyOCR init failed: %s", exc)
            _easy_ocr = False

    return _easy_ocr if _easy_ocr is not False else None


# ---------------------------------------------------------------------------
# Per-engine OCR helpers
# ---------------------------------------------------------------------------

def _ocr_paddle(image) -> tuple[str, float]:
    """Run PaddleOCR on a PIL Image. Returns (text, avg_confidence)."""
    ocr = _get_paddle_ocr()
    if ocr is None:
        return "", 0.0

    try:
        import numpy as np  # noqa: PLC0415

        img_array = np.array(image.convert("RGB"))
        result    = ocr.ocr(img_array, cls=True)

        if not result or result[0] is None:
            return "", 0.0

        texts, confs = [], []
        for line in result[0]:
            if not line or len(line) != 2:
                continue
            _, (text, conf) = line
            conf = float(conf)
            if conf >= 0.5 and text.strip():
                texts.append(text.strip())
                confs.append(conf)

        if not texts:
            return "", 0.0

        avg_conf = sum(confs) / len(confs)
        return "\n".join(texts), avg_conf

    except Exception as exc:
        logger.warning("PaddleOCR page error: %s", exc)
        return "", 0.0


def _ocr_easyocr(image) -> tuple[str, float]:
    """Run EasyOCR on a PIL Image. Returns (text, avg_confidence)."""
    reader = _get_easy_ocr()
    if reader is None:
        return "", 0.0

    try:
        import numpy as np  # noqa: PLC0415

        img_array = np.array(image.convert("RGB"))
        result    = reader.readtext(img_array, detail=1)

        if not result:
            return "", 0.0

        texts, confs = [], []
        for (_, text, conf) in result:
            conf = float(conf)
            if conf >= 0.5 and text.strip():
                texts.append(text.strip())
                confs.append(conf)

        if not texts:
            return "", 0.0

        avg_conf = sum(confs) / len(confs)
        return "\n".join(texts), avg_conf

    except Exception as exc:
        logger.warning("EasyOCR page error: %s", exc)
        return "", 0.0


def _ocr_tesseract(image) -> tuple[str, float]:
    """Run Tesseract OCR on a PIL Image. Returns (text, -1.0) — no per-line confidence."""
    try:
        import pytesseract  # noqa: PLC0415

        text = pytesseract.image_to_string(image, config="--psm 3 --oem 3") or ""
        return text, -1.0
    except Exception as exc:
        logger.warning("Tesseract page error: %s", exc)
        return "", 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_file_hash(content: bytes) -> str:
    """SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(content).hexdigest()


def ocr_image(
    image,
    file_hash: Optional[str] = None,
    page_num: int = 0,
) -> tuple[str, float]:
    """OCR a PIL Image using PaddleOCR → EasyOCR → Tesseract cascade.

    Parameters
    ----------
    image     : PIL.Image.Image
    file_hash : SHA-256 hex digest of the source file (for caching).
                If None, result is not cached.
    page_num  : 0-based page index within the source file.

    Returns
    -------
    (cleaned_text, confidence)
        confidence is the average engine confidence (0–1), or -1 for Tesseract.
    """
    cache_key = f"{file_hash}:{page_num}" if file_hash else None

    # --- cache read ---
    if cache_key:
        cache = _get_cache()
        if cache_key in cache:
            entry = cache[cache_key]
            logger.debug("OCR cache hit key=%s engine=%s", cache_key, entry.get("engine"))
            return entry["text"], entry["confidence"]

    # --- cascade ---
    text, conf, engine_used = "", 0.0, "none"

    text, conf = _ocr_paddle(image)
    if text.strip():
        engine_used = "paddle"
    else:
        text, conf = _ocr_easyocr(image)
        if text.strip():
            engine_used = "easyocr"
        else:
            text, conf = _ocr_tesseract(image)
            if text.strip():
                engine_used = "tesseract"

    text = clean_ocr_text(text)

    # --- cache write ---
    if cache_key:
        with _cache_lock:
            cache = _get_cache()
            cache[cache_key] = {
                "text": text,
                "confidence": conf,
                "engine": engine_used,
            }
            _save_cache(cache)

    logger.debug(
        "OCR complete engine=%s conf=%.2f chars=%d", engine_used, conf, len(text)
    )
    return text, conf


def clean_ocr_text(text: str) -> str:
    """Normalise and filter OCR output into clean English text.

    Steps
    -----
    1. NFC Unicode normalisation (removes combining chars / control codes).
    2. Fix hyphenated line-breaks (word-\\ncontinuation → wordcontinuation).
    3. Collapse runs of spaces/tabs to a single space per line.
    4. Drop lines with < 30 % alphanumeric content (OCR noise).
    5. Remove consecutive duplicate lines.
    6. Collapse runs of > 2 blank lines to one.
    """
    if not text:
        return ""

    # 1. NFC normalisation; strip C-category control chars (keep \\n \\t space)
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch in "\n\t "
    )

    # 2. Fix hyphenated line-breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 3. Collapse spaces/tabs within lines
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Filter low-quality lines
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if len(line) < 3:
            continue
        alnum = sum(1 for c in line if c.isalnum())
        if alnum / len(line) < 0.30:
            continue
        cleaned_lines.append(line)

    # 5. Remove consecutive duplicate lines
    deduped: list[str] = []
    prev = None
    for line in cleaned_lines:
        if line and line == prev:
            continue
        deduped.append(line)
        prev = line if line else prev

    text = "\n".join(deduped)

    # 6. Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
