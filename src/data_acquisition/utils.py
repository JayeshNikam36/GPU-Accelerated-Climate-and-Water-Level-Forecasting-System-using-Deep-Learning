# src/data_acquisition/utils.py
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_data_path(subdir: str = "") -> str:
    """Get safe path from .env or default."""
    base_dir = os.getenv("DATA_RAW_DIR", "data/raw")
    full_path = os.path.join(base_dir, subdir)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime, handle common formats."""
    formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")

def log_error(msg: str):
    logger.error(msg)