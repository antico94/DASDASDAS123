from config.base_config import BaseConfig


class ProductionConfig(BaseConfig):
    """Development environment configuration."""

    # Database settings
    DB_SERVER = "localhost"
    DB_NAME = "TestDB"

    # MT5 settings (you'll need to customize these)
    MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5\\terminal64.exe"
    MT5_LOGIN = 166774
    MT5_PASSWORD = "O11e7nqlX."
    MT5_SERVER = "FusionMarkets-Demo"

    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_FILE = "logs/trading_bot_dev.log"