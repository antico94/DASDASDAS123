# logging/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config


class Logger:
    """Custom logger for the trading bot."""

    @staticmethod
    def setup_logger(name, log_file=None):
        """Set up a logger instance with the specified name and file.

        Args:
            name (str): The logger name
            log_file (str, optional): The log file path. Defaults to the config value.

        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(name)

        # Set log level from config
        log_level = getattr(logging, Config.LOG_LEVEL)
        logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(Config.LOG_FORMAT)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Add file handler if a log file is specified
        if log_file is None:
            log_file = Config.LOG_FILE

        if log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


# Create a default application logger
app_logger = Logger.setup_logger('trading_bot')