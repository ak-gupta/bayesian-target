"""Import path."""

from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent
METADATA_DIR = CURR_DIR / "metadata"
DATA_DIR = CURR_DIR / "data"
