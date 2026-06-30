"""Shared utilities for ingestion, preprocessing, logging, and UI history."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from werkzeug.utils import secure_filename

from config import (
    ALLOWED_EXTENSIONS,
    HISTORY_PATH,
    IMAGE_SIZE,
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_FOLDER,
    UPLOAD_FOLDER,
)


def ensure_directories() -> None:
    """Create runtime directories used by training, inference, and Flask."""
    for directory in (MODEL_DIR, UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "wildfire") -> logging.Logger:
    """Configure a file and console logger once per process."""
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_DIR / "wildfire_app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_filename(filename: str) -> str:
    """Return a collision-resistant, browser-safe filename."""
    safe_name = secure_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    stem = Path(safe_name).stem[:80] or "image"
    return f"{stem}_{uuid4().hex[:12]}{suffix}"


def validate_image(path: Path) -> bool:
    """Return True if Pillow can identify and verify the image."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def collect_image_files(directory: Path) -> list[Path]:
    """Collect valid-looking image paths recursively by extension."""
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS
    )


def preprocess_image(image_path: str | Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load an RGB image, resize to model input, normalize, and add batch dimension."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_prediction_history(limit: int = 8) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(history, list):
        return []
    return history[:limit]


def append_prediction_history(entry: dict, limit: int = 20) -> None:
    history = load_prediction_history(limit=limit)
    stamped_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **entry,
    }
    save_json(HISTORY_PATH, [stamped_entry, *history][:limit])


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def summarize_skipped(paths: Iterable[Path]) -> str:
    skipped = list(paths)
    if not skipped:
        return "No corrupted images skipped."
    sample = ", ".join(path.name for path in skipped[:5])
    remainder = "" if len(skipped) <= 5 else f" and {len(skipped) - 5} more"
    return f"Skipped {len(skipped)} corrupted image(s): {sample}{remainder}"


def ensure_dashboard_placeholder_image(path: Path, title: str, subtitle: str) -> Path:
    """Generate a static placeholder chart image when training artifacts are unavailable."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), color=(17, 22, 30))
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = 80, 90, width - 80, height - 110
    draw.rectangle((left, top, right, bottom), outline=(74, 88, 104), width=3)

    for step in range(1, 6):
        y = top + int(((bottom - top) / 6) * step)
        draw.line((left, y, right, y), fill=(42, 52, 64), width=1)

    points = []
    span = right - left
    for idx in range(9):
        x = left + int((idx / 8) * span)
        y = bottom - int((bottom - top) * (0.15 + (0.08 * idx)))
        points.append((x, y))
    draw.line(points, fill=(255, 114, 70), width=6)

    draw.text((left + 14, top + 14), title, fill=(244, 247, 251))
    draw.text((left + 14, bottom + 24), subtitle, fill=(169, 179, 190))
    image.save(path, format="PNG")
    return path
