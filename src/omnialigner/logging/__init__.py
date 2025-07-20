from .logging import setup_logger
import logging
logger = setup_logger(level=logging.INFO)

__all__ = ["logger"]