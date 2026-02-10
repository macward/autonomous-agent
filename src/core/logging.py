"""Logging configuration for audit trail."""

import logging
import sys
from datetime import datetime


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure structured logging for the agent.

    All agent actions are logged for audit purposes.
    """
    level = logging.DEBUG if debug else logging.INFO

    # Create formatter with timestamp and structured fields
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Root logger
    root_logger = logging.getLogger("agent")
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"agent.{name}")
