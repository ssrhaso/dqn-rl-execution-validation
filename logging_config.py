"""
Centralised logging configuration for production reliability and debugging.

Provides clean, professional logging setup for all application components.
"""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> None:
    """
    Configure application-wide logging with professional formatting.
    
    Logs to both console (real-time) and file (archive).
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Professional formatter: clean and readable
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler (real-time output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (persistent log)
    file_handler = logging.FileHandler(log_dir / "execution.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)