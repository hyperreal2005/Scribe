import logging
import os
from logging.handlers import RotatingFileHandler


def _setup_logger(name: str, log_file: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_ai_logger() -> logging.Logger:
    log_path = os.getenv("AI_LOG_FILE", "logs/ai_frontend.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    return _setup_logger("ai_frontend", log_path)
