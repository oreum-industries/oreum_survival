# src.engine.logger.py
"""Organise logging"""

import logging
from pathlib import Path

from pyprojroot.here import here
from rich.logging import RichHandler

__all__ = ["get_logger"]

_handler_kws = dict(
    log_format_file=(
        "%(asctime)s - [%(levelname)s] - %(name)s.%(funcName)s(%(lineno)d) - %(message)s"
    ),
    log_format_rich="%(name)s.%(funcName)s(%(lineno)d) - %(message)s",
)


def _get_file_handler(fqn=Path("logs", "log.log"), **kwargs) -> logging.FileHandler:
    """Conventional filehandler
    TODO Consider that file-based handling often not appropriate
    e.g. for a container. Also consider using e.g. TimedRotatingFileHandler
    """
    fh = logging.FileHandler(str(fqn), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(kwargs.pop("log_format_file")))
    return fh


def _get_rich_handler(**kwargs) -> RichHandler:
    """Stream handler using Rich (for formatting)"""
    rh = RichHandler(rich_tracebacks=True, tracebacks_suppress=["pymc"])
    rh.setLevel(logging.INFO)
    rh.setFormatter(logging.Formatter(kwargs.pop("log_format_rich")))
    return rh


def get_logger(
    name: str, log_to_file: bool = True, fqn: Path = None, notebook: bool = False
) -> logging.Logger:
    """Create a logger, force the file handler to use the same file"""
    level = logging.DEBUG
    if notebook:
        level = logging.WARNING

    fqn = Path(here("logs").resolve(strict=True), f"{name}.log") if fqn is None else fqn
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_to_file:
        logger.addHandler(_get_file_handler(fqn, **_handler_kws))
    else:  # go to stream
        logger.addHandler(_get_rich_handler(**_handler_kws))

    return logger
