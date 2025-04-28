# db_logger/logging_setup.py
import logging
from db_logger.db_logger import DBLogger


class DBLogHandler(logging.Handler):
    """Handler to route log messages to the database logger."""

    def emit(self, record):
        """Process a log record for database logging."""
        try:
            # Extract log level and component
            level_name = record.levelname
            component = record.name
            message = self.format(record)

            # Map logging levels to our log types
            if record.levelno >= logging.ERROR:
                DBLogger.log_error(component, message, error_type="ERROR")
            elif record.levelno >= logging.WARNING:
                DBLogger.log_error(component, message, error_type="WARNING")
            else:
                # Map INFO and DEBUG to event types
                event_type = "INFO" if record.levelno >= logging.INFO else "DEBUG"
                DBLogger.log_event(event_type, message, component=component)

        except Exception:
            self.handleError(record)


def setup_logging(console_level=logging.INFO, file_level=logging.DEBUG,
                  db_level=logging.INFO, log_file=None):
    """Configure the logging system with console, file, and database outputs.

    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        db_level: Logging level for database output
        log_file: Path to log file (None to disable file logging)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all messages at root

    # Clear existing handlers
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Database handler
    db_handler = DBLogHandler()
    db_handler.setLevel(db_level)
    db_handler.setFormatter(file_formatter)  # Use detailed formatter
    root_logger.addHandler(db_handler)

    # Return the root logger
    return root_logger