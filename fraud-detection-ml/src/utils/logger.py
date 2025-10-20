# src/utils/logger.py
import logging
import os
from datetime import datetime

def get_logger(name: str = "ml_pipeline", log_dir: str = "logs", level=logging.INFO):
    """
    Returns a logger instance with both console + file handlers.

    Parameters
    ----------
    name : str
        Logger name (e.g. 'feature_engineering', 'imputer').
    log_dir : str
        Directory where log files are stored.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d}.log")

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # prevent duplicate handlers

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger
