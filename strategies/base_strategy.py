# strategies/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from db_logger.db_logger import DBLogger
from data.models import StrategySignal
from mt5_connector.data_fetcher import MT5DataFetcher


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, symbol, timeframe, name=None, data_fetcher=None):
        """Initialize the strategy.

        Args:
            symbol (str): Symbol to trade (e.g., 'XAUUSD')
            timeframe (str): Chart timeframe (e.g., 'M5', 'H1')
            name (str, optional): Strategy name. Defaults to class name.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None (creates new).
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.name = name or self.__class__.__name__
        self.data_fetcher = data_fetcher or MT5DataFetcher()

        # Log initialization
        DBLogger.log_event("INFO", f"Initialized strategy: {self.name} for {self.symbol} {self.timeframe}", "Strategy")

    @abstractmethod
    def analyze(self, data):
        """Analyze market data and generate trading signals.

        This method must be implemented by subclasses.

        Args:
            data (pandas.DataFrame): OHLC data for analysis

        Returns:
            list: Generated trading signals
        """
        pass

    def get_ohlc_data(self, count=200):
        """Get OHLC data for analysis.

        Args:
            count (int, optional): Number of candles. Defaults to 100.

        Returns:
            pandas.DataFrame: OHLC data
        """
        return self.data_fetcher.get_latest_data_to_dataframe(
            symbol=self.symbol,
            timeframe=self.timeframe,
            count=count
        )

    def generate_signals(self):
        """Generate trading signals based on current market data.

        Returns:
            list: Generated trading signals
        """
        try:
            # Get data for analysis
            data = self.get_ohlc_data()

            if data.empty:
                DBLogger.log_event("WARNING", f"No data available for {self.symbol} {self.timeframe}", "Strategy")
                return []

            # Log data summary
            DBLogger.log_event("DEBUG", f"Analyzing {len(data)} candles for {self.symbol} {self.timeframe}", "Strategy")
            DBLogger.log_event("DEBUG", f"Data range: {data.index.min()} to {data.index.max()}", "Strategy")

            # Call the strategy's analysis method
            signals = self.analyze(data)

            # Log generated signals
            if signals:
                DBLogger.log_event("INFO", f"Generated {len(signals)} signals for {self.symbol} {self.timeframe}", "Strategy")
                for signal in signals:
                    DBLogger.log_event("INFO", f"Signal: {signal.signal_type} at {signal.price}", "Strategy")
            else:
                DBLogger.log_event("DEBUG", f"No signals generated for {self.symbol} {self.timeframe}", "Strategy")

            return signals

        except Exception as e:
            DBLogger.log_error("Strategy", f"Error generating signals for {self.name}", exception=e)
            return []

    def create_signal(self, signal_type, price, strength=0.5, metadata=None):
        """Create a strategy signal.

        Args:
            signal_type (str): Signal type ('BUY', 'SELL', 'CLOSE', etc.)
            price (float): Signal price
            strength (float, optional): Signal strength (0-1). Defaults to 0.5.
            metadata (dict, optional): Additional signal data. Defaults to None.

        Returns:
            StrategySignal: Created signal
        """
        import json

        # Validate inputs
        if signal_type not in ['BUY', 'SELL', 'CLOSE', 'CANCEL']:
            error_msg = f"Invalid signal type: {signal_type}"
            DBLogger.log_error("Strategy", error_msg)
            raise ValueError(error_msg)

        if not isinstance(price, (int, float)) or price <= 0:
            error_msg = f"Invalid price: {price}"
            DBLogger.log_error("Strategy", error_msg)
            raise ValueError(f"Invalid price: {price}")

        if not (0 <= strength <= 1):
            error_msg = f"Invalid strength: {strength}. Must be between 0 and 1."
            DBLogger.log_error("Strategy", error_msg)
            raise ValueError(error_msg)

        # Convert NumPy values to native Python types for JSON serialization
        processed_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if hasattr(value, 'item'):  # For NumPy scalars
                    processed_metadata[key] = value.item()
                elif isinstance(value, bool):  # Handle boolean values
                    processed_metadata[key] = bool(value)
                else:
                    processed_metadata[key] = value

        # Create signal
        signal = StrategySignal(
            strategy_name=self.name,
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=datetime.utcnow(),
            signal_type=signal_type,
            price=float(price) if hasattr(price, 'item') else price,
            strength=float(strength) if hasattr(strength, 'item') else strength,
            signal_data=json.dumps(processed_metadata or {})  # Use signal_data to match model field
        )

        # Log signal creation
        DBLogger.log_event("DEBUG",
            f"Created {signal_type} signal for {self.symbol} at {price} with strength {strength}",
            "Strategy")

        return signal