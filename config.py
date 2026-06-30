"""Central configuration for the wildfire detection system."""

from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = BASE_DIR / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

DATASET_DIR = BASE_DIR / "dataset"
FIRE_DIR = DATASET_DIR / "fire"
NO_FIRE_DIR = DATASET_DIR / "no_fire"

MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "wildfire_model.h5"

STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
OUTPUT_FOLDER = STATIC_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
HISTORY_PATH = OUTPUT_FOLDER / "prediction_history.json"

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
SEED = int(os.getenv("SEED", "42"))

TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10

CLASS_NAMES = {
    0: "No Fire",
    1: "Fire",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
