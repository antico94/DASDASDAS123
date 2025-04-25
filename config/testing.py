# config/testing.py
from config.base_config import BaseConfig


class TestingConfig(BaseConfig):
    """Testing environment configuration."""

    # Override base settings for testing
    LOG_LEVEL = "DEBUG"

    # Use an in-memory or test database
    DB_NAME = "XAUUSDTradingBot_Test"