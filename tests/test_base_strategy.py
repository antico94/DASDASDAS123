# tests/test_base_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


# Create a concrete subclass for testing the abstract base class
class ConcreteStrategy(BaseStrategy):
    """Concrete implementation of BaseStrategy for testing."""

    def __init__(self, symbol="XAUUSD", timeframe="H1", name="TestStrategy", data_fetcher=None):
        super().__init__(symbol, timeframe, name, data_fetcher)
        self.min_required_candles = 20

    def analyze(self, data):
        """Implement the required abstract method."""
        signal = self.create_signal(
            signal_type="BUY",
            price=1800.0,
            strength=0.7,
            metadata={"test": "value"}
        )
        return [signal]


class TestBaseStrategy(unittest.TestCase):
    """Unit tests for the BaseStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_data_fetcher = MagicMock()
        self.strategy = ConcreteStrategy(
            symbol="XAUUSD",
            timeframe="H1",
            name="TestStrategy",
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test that initialization works correctly."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "H1")
        self.assertEqual(self.strategy.name, "TestStrategy")
        self.assertIsNotNone(self.strategy.logger)
        self.assertIsNotNone(self.strategy.data_fetcher)

    def test_get_ohlc_data(self):
        """Test the get_ohlc_data method."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'open': [1800, 1805],
            'high': [1810, 1815],
            'low': [1795, 1800],
            'close': [1805, 1810]
        })

        # Configure the mock to return the DataFrame
        self.mock_data_fetcher.get_latest_data_to_dataframe.return_value = mock_df

        # Call the method
        result = self.strategy.get_ohlc_data(count=10)

        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result['close'].iloc[-1], 1810)

        # Verify the mock was called with the right parameters
        self.mock_data_fetcher.get_latest_data_to_dataframe.assert_called_with(
            symbol="XAUUSD",
            timeframe="H1",
            count=10
        )

    def test_generate_signals_success(self):
        """Test generate_signals method when analysis succeeds."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'open': [1800, 1805],
            'high': [1810, 1815],
            'low': [1795, 1800],
            'close': [1805, 1810]
        })

        # Mock data fetcher
        self.mock_data_fetcher.get_latest_data_to_dataframe.return_value = mock_df

        # Create mock signal
        mock_signal = MagicMock()
        mock_signal.signal_type = "BUY"
        mock_signal.price = 1810.0

        # Mock analyze method
        with patch.object(self.strategy, 'analyze', return_value=[mock_signal]):
            # Call generate_signals
            signals = self.strategy.generate_signals()

            # Verify results
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].signal_type, "BUY")
            self.assertEqual(signals[0].price, 1810.0)

    def test_generate_signals_empty_data(self):
        """Test generate_signals method with empty data."""
        # Mock empty DataFrame
        self.mock_data_fetcher.get_latest_data_to_dataframe.return_value = pd.DataFrame()

        # Call generate_signals
        signals = self.strategy.generate_signals()

        # Verify empty result
        self.assertEqual(len(signals), 0)

    def test_generate_signals_exception(self):
        """Test generate_signals method when an exception occurs."""
        # Create mock DataFrame
        mock_df = pd.DataFrame({
            'open': [1800, 1805],
            'high': [1810, 1815],
            'low': [1795, 1800],
            'close': [1805, 1810]
        })

        # Mock data fetcher
        self.mock_data_fetcher.get_latest_data_to_dataframe.return_value = mock_df

        # Mock analyze to raise an exception
        with patch.object(self.strategy, 'analyze', side_effect=Exception("Test exception")):
            # Call generate_signals
            signals = self.strategy.generate_signals()

            # Verify empty result due to exception
            self.assertEqual(len(signals), 0)

    def test_create_signal_valid(self):
        """Test create_signal method with valid inputs."""
        # Call create_signal
        signal = self.strategy.create_signal(
            signal_type="BUY",
            price=1820.5,
            strength=0.75,
            metadata={"test": "data", "numeric": 123.45}
        )

        # Verify signal properties
        self.assertEqual(signal.signal_type, "BUY")
        self.assertEqual(signal.price, 1820.5)
        self.assertEqual(signal.strength, 0.75)
        self.assertEqual(signal.symbol, "XAUUSD")
        self.assertEqual(signal.timeframe, "H1")
        self.assertEqual(signal.strategy_name, "TestStrategy")

        # Verify metadata was serialized correctly
        import json
        metadata = json.loads(signal.signal_data)
        self.assertEqual(metadata["test"], "data")
        self.assertEqual(metadata["numeric"], 123.45)

    def test_create_signal_invalid_type(self):
        """Test create_signal method with invalid signal type."""
        # Test with invalid signal type
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="INVALID",
                price=1800.0
            )

    def test_create_signal_invalid_price(self):
        """Test create_signal method with invalid price."""
        # Test with negative price
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="BUY",
                price=-1800.0
            )

        # Test with zero price
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="BUY",
                price=0
            )

        # Test with non-numeric price
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="BUY",
                price="invalid"
            )

    def test_create_signal_invalid_strength(self):
        """Test create_signal method with invalid strength."""
        # Test with negative strength
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="BUY",
                price=1800.0,
                strength=-0.5
            )

        # Test with strength > 1
        with self.assertRaises(ValueError):
            self.strategy.create_signal(
                signal_type="BUY",
                price=1800.0,
                strength=1.5
            )

    def test_create_signal_numpy_values(self):
        """Test create_signal with NumPy scalar values."""
        # Create a signal with numpy scalar values
        signal = self.strategy.create_signal(
            signal_type="BUY",
            price=np.float64(1850.75),
            strength=np.float32(0.8),
            metadata={
                "numpy_int": np.int32(42),
                "numpy_float": np.float64(123.456),
                "numpy_bool": np.bool_(True)
            }
        )

        # Verify conversion to Python types
        self.assertIsInstance(signal.price, float)
        self.assertIsInstance(signal.strength, float)

        # Verify metadata conversion
        import json
        metadata = json.loads(signal.signal_data)
        self.assertIsInstance(metadata["numpy_int"], int)
        self.assertIsInstance(metadata["numpy_float"], float)
        self.assertIsInstance(metadata["numpy_bool"], bool)


if __name__ == '__main__':
    unittest.main()