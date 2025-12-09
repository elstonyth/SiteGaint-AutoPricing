"""
Structured logging configuration.

Supports both text and JSON log formats based on config.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

try:
    from pythonjsonlogger import jsonlogger

    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        # Add extra fields if present
        record.extra_fields = getattr(record, "extra_fields", {})
        return super().format(record)


class JSONFormatter(jsonlogger.JsonFormatter if HAS_JSON_LOGGER else logging.Formatter):
    """JSON log formatter with standard fields."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add any extra fields passed via extra={}
        if hasattr(record, "extra_fields"):
            log_record.update(record.extra_fields)


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_format: "text" for human-readable, "json" for structured.
        log_file: Optional file path for log output.
        max_bytes: Max log file size before rotation.
        backup_count: Number of backup files to keep.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter based on format type
    if log_format.lower() == "json" and HAS_JSON_LOGGER:
        formatter = JSONFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    root_logger.info(f"Logging configured: level={level}, format={log_format}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured fields to log messages."""

    def __init__(self, logger: logging.Logger, **fields):
        self.logger = logger
        self.fields = fields
        self.old_factory = None

    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        fields = self.fields

        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.extra_fields = fields
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)
        return False
