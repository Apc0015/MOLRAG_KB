"""Logging configuration for MolRAG"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "100 MB",
    retention: str = "10 days",
    format_string: Optional[str] = None
) -> None:
    """
    Configure loguru logger for MolRAG

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        rotation: When to rotate log files
        retention: How long to keep log files
        format_string: Custom format string (optional)
    """
    # Remove default logger
    logger.remove()

    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

    # Add file handler if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=format_string,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True
        )

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: Optional[str] = None):
    """
    Get logger instance

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize default logger
setup_logger()
