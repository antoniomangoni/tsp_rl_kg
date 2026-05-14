"""Logging configuration using Loguru.

Call ``configure_logging()`` once at programme entry to set up sinks.
Every module simply does ``from loguru import logger`` to emit logs.
An ``InterceptHandler`` bridges stdlib ``logging`` so that SB3, PyTorch,
and other libraries are routed through Loguru automatically.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """stdlib logging handler that forwards records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Find the Loguru level that matches the stdlib level.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    *,
    json_output: bool = False,
) -> None:
    """Set up Loguru sinks and intercept stdlib logging.

    Parameters
    ----------
    log_dir:
        Directory for log files. A ``{time}`` placeholder is included in
        the file name so each run gets its own file.
    level:
        Minimum log level for all sinks.
    json_output:
        If ``True`` an additional JSON-formatted file sink is created.
    """
    import os

    os.makedirs(log_dir, exist_ok=True)

    # Remove default Loguru sink so we control everything.
    logger.remove()

    # Console - coloured, human-readable
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File - rotated at 10 MB, kept for 7 days
    logger.add(
        os.path.join(log_dir, "tsp_rl_kg_{time}.log"),
        level=level,
        rotation="10 MB",
        retention="7 days",
        format=("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"),
    )

    if json_output:
        logger.add(
            os.path.join(log_dir, "tsp_rl_kg_{time}.json"),
            level=level,
            rotation="10 MB",
            retention="7 days",
            serialize=True,
        )

    # Intercept stdlib logging (SB3, PyTorch, Gymnasium, etc.)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
