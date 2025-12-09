"""
Logging setup module.

Configures application-wide logging with file and console handlers.
Supports both standard text format and structured JSON logging.
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    log_format: str | None = None,
    max_bytes: int = 10485760,  # 10 MB
    backup_count: int = 5,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure application logging.

    Sets up both console and file logging with rotation.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Path to log file. Defaults to logs/app.log.
        log_format: Log message format string.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of backup files to keep.
        json_format: If True, use JSON structured logging format.

    Returns:
        logging.Logger: Configured root logger.
    """
    # Check for JSON format via environment variable
    if os.environ.get("LOG_FORMAT", "").lower() == "json":
        json_format = True

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Default log file
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter based on format type
    formatter = JSONFormatter() if json_format else logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        root_logger.warning(f"Could not create file handler: {e}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger: Named logger instance.
    """
    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Change the logging level for all handlers.

    Args:
        level: New logging level.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)
