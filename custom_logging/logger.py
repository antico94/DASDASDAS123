# custom_logging/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config


class Logger:
    """Custom logger for the trading bot."""

    @staticmethod
    def setup_logger(name, log_file=None, log_level=None):
        """Set up a logger instance with the specified name and file.

        Args:
            name (str): The logger name
            log_file (str, optional): The log file path. Defaults to the config value.
            log_level (str, optional): Log level override. Defaults to the config value.

        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(name)

        # Clear existing handlers to avoid duplicates if logger already exists
        if logger.handlers:
            logger.handlers = []

        # Set log level from config or override
        log_level = log_level or Config.LOG_LEVEL
        level = getattr(logging, log_level)
        logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(Config.LOG_FORMAT)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Default to INFO for console unless in DEBUG mode
        if log_level == "DEBUG":
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)

        logger.addHandler(console_handler)

        # Add file handler if a log file is specified
        if log_file is None:
            log_file = Config.LOG_FILE

        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Create rotating file handler - always use full log level for file
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)  # Full logging to file
            logger.addHandler(file_handler)

        # Silence sqlalchemy engine logs
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

        return logger


# Create a default application logger
app_logger = Logger.setup_logger('trading_bot')


# This function can be used to set the verbosity of console output
def set_console_log_level(verbose=False):
    """Set the console log level based on verbosity flag.

    Args:
        verbose (bool): Whether to show verbose (DEBUG) output
    """
    logger = logging.getLogger('trading_bot')

    # Find the console handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            if verbose:
                handler.setLevel(logging.DEBUG)
                app_logger.info("Console logging set to DEBUG (verbose)")
            else:
                handler.setLevel(logging.INFO)
                app_logger.info("Console logging set to INFO (normal)")
            break