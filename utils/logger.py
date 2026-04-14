"""
Logger ringkas dengan timestamp untuk Anney GPT.
"""

import logging
import sys
from datetime import datetime


def get_logger(name: str = "anney", log_file: str = None) -> logging.Logger:
    """
    Buat logger yang menulis ke konsol dan (opsyenal) ke fail.

    Args:
        name:     Nama logger
        log_file: Laluan fail log (None = konsol sahaja)

    Returns:
        Logger yang dikonfigurasi
    """
    logger = logging.getLogger(name)

    # Elak duplikasi handler jika dipanggil berkali-kali
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler konsol
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler fail (opsyenal)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
