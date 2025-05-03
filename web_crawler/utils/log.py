"""
Logging utility functions for the web crawler.
"""

import os
import logging
import json
from logging.handlers import RotatingFileHandler

# Dictionary to keep track of loggers
_loggers = {}


def get_logger(name, level=logging.INFO):
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger.
        level (int): Log level.
        
    Returns:
        logging.Logger: Logger instance.
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
        
    # Create a new logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Store logger in dictionary
    _loggers[name] = logger
    return logger


def init_file_logging(name, log_dir="logs", level=logging.INFO, max_bytes=10485760, backup_count=5):
    """
    Initialize logging to a file with rotation.
    
    Args:
        name (str): Name of the logger.
        log_dir (str): Directory to store log files.
        level (int): Log level.
        max_bytes (int): Maximum size of log file before rotation.
        backup_count (int): Number of backup files to keep.
        
    Returns:
        logging.Logger: Logger instance.
    """
    logger = get_logger(name, level)
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file handler
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    return logger


class JsonFormatter(logging.Formatter):
    """
    Format log records as JSON.
    """
    
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        
        if hasattr(record, "exc_info") and record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)