"""Structured JSON logging setup."""
import logging
import sys


def setup_logging(level: str = 'INFO'):
    """
    Configure root logger to emit structured JSON-style logs.
    Falls back to plain text if python-json-logger is not installed.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    try:
        from pythonjsonlogger import jsonlogger

        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(levelname)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S',
        )
        handler.setFormatter(formatter)
    except ImportError:
        # Graceful fallback
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ('transformers', 'sentence_transformers', 'urllib3', 'httpx'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root
