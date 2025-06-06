# config/base_config.py
class BaseConfig:
    """Base configuration class with common settings for all environments."""

    # Application settings
    APP_NAME = "XAU/USD Trading Bot"
    VERSION = "1.0.0"

    # MT5 settings
    MT5_TERMINAL_PATH = "C:\Program Files\MetaTrader 5\\terminal64.exe"
    MT5_LOGIN = 166774
    MT5_PASSWORD = "O11e7nqlX."
    MT5_SERVER = "FusionMarkets-Demo"

    # Trading settings
    SYMBOL = "XAUUSD"
    DEFAULT_VOLUME = 0.01  # Minimum lot size
    MAX_POSITIONS = 200  # Maximum number of open positions

    # Risk management
    MAX_RISK_PER_TRADE_PERCENT = 1.0  # Maximum risk per trade as percentage of account
    MAX_DAILY_RISK_PERCENT = 5.0  # Maximum daily risk as percentage of account
    MAX_DRAWDOWN_PERCENT = 15.0  # Maximum allowed drawdown as percentage

    # Database settings
    DB_DRIVER = "ODBC Driver 17 for SQL Server"
    DB_SERVER = "localhost"
    DB_NAME = "TestDB"
    DB_USERNAME = "app_user"
    DB_PASSWORD = "password01!"

    # Strategy settings
    STRATEGIES_ENABLED = ["triple_ma", "moving_average", "breakout", "range_bound", "momentum_scalping", "ichimoku"]  # Updated with triple_ma

    # Original Moving Average Trend Strategy
    MA_FAST_PERIOD = 20
    MA_SLOW_PERIOD = 50
    MA_TIMEFRAME = "H1"

    # Triple Moving Average Strategy (new)
    TRIPLE_MA_FAST_PERIOD = 10      # Fast EMA period
    TRIPLE_MA_MEDIUM_PERIOD = 50    # Medium SMA period
    TRIPLE_MA_SLOW_PERIOD = 200     # Slow SMA period
    TRIPLE_MA_TIMEFRAME = "H4"      # Default timeframe

    # Breakout Strategy
    BO_TIMEFRAME = "M15"
    BO_LOOKBACK_PERIODS = 48
    BO_MIN_RANGE_BARS = 10
    BO_VOLUME_THRESHOLD = 1.5
    BO_ATR_PERIOD = 14

    # Range-Bound Strategy
    RB_TIMEFRAME = "M15"
    RB_LOOKBACK_PERIODS = 48
    RB_MIN_RANGE_BARS = 10
    RB_RSI_PERIOD = 14
    RB_RSI_OVERBOUGHT = 70
    RB_RSI_OVERSOLD = 30
    RB_ADX_PERIOD = 14
    RB_ADX_THRESHOLD = 20

    # Momentum Scalping Strategy
    MS_TIMEFRAME = "M5"
    MS_EMA_PERIOD = 20
    MS_MACD_FAST = 12
    MS_MACD_SLOW = 26
    MS_MACD_SIGNAL = 9
    MS_RSI_PERIOD = 14
    MS_RSI_THRESHOLD_HIGH = 60
    MS_RSI_THRESHOLD_LOW = 40
    MS_STOCH_K_PERIOD = 14
    MS_STOCH_D_PERIOD = 3
    MS_STOCH_SLOWING = 3
    MS_MOMENTUM_PERIOD = 10
    MS_VOLUME_THRESHOLD = 1.5
    MS_MAX_SPREAD = 3.0
    MS_CONSIDER_SESSION = True

    # Ichimoku Cloud Strategy
    IC_TIMEFRAME = "H1"
    IC_TENKAN_PERIOD = 9
    IC_KIJUN_PERIOD = 26
    IC_SENKOU_B_PERIOD = 52

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "trading_bot.log"


    @property
    def DATABASE_URI(self):
        """Return the database connection URI."""
        return (f"mssql+pyodbc://{self.DB_USERNAME}:{self.DB_PASSWORD}@"
                f"{self.DB_SERVER}/{self.DB_NAME}"
                f"?driver={self.DB_DRIVER}")