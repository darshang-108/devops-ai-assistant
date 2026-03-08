"""
Structured JSON logging – writes one JSON object per line to both
``logs/app.log`` and stdout so logs are queryable in ELK / CloudWatch
and visible in Docker / terminal output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")


class _JSONFormatter(logging.Formatter):
    """Formats each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached by the caller
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        return json.dumps(log_obj, default=str)


def setup_logger(name: str = "api") -> logging.Logger:
    """
    Create (or return the existing) logger with JSON file + console handlers.

    The ``logs/`` directory is created automatically if it does not exist.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on hot-reload
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = _JSONFormatter()

    # ── File handler ─────────────────────────────────────────────────────
    log_path = Path(LOG_FILE)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as exc:
        print(f"⚠️  Could not create log file {log_path}: {exc}", file=sys.stderr)

    # ── Console handler ──────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def log_request(
    logger: logging.Logger,
    *,
    endpoint: str,
    question: str,
    answer_length: int,
    sources: list[str],
    duration_ms: float,
    status: str,
    ip: str,
    error: str | None = None,
) -> None:
    """
    Emit a structured JSON log entry for a single ``/ask`` request.

    Parameters are passed as keyword-only arguments so call-sites are
    self-documenting.
    """
    extra: dict = {
        "endpoint": endpoint,
        "question": question,
        "answer_length": answer_length,
        "sources": sources,
        "duration_ms": round(duration_ms, 1),
        "status": status,
        "ip": ip,
    }
    if error:
        extra["error"] = error

    record = logger.makeRecord(
        name=logger.name,
        level=logging.ERROR if status == "error" else logging.INFO,
        fn="",
        lno=0,
        msg=f"{status.upper()} {endpoint}",
        args=(),
        exc_info=None,
    )
    record.extra_fields = extra  # type: ignore[attr-defined]
    logger.handle(record)
