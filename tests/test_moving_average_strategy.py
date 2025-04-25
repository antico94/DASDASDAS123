# tests/test_moving_average_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.moving_average import MovingAverageStrategy


class TestMovingAverageStrategy(unittest.TestCase):
    """Unit tests for the Moving Average Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = MovingAverageStrategy(
            symbol="XAUUSD",
            timeframe="H1",
            fast_period=20,
            slow_period=50,
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "H1")
        self.assertEqual(self.strategy.fast_period, 20)
        self.assertEqual(self.strategy.slow_period, 50)
        # tests/test_moving_average_strategy.py (continued)
        self.assertEqual(self.strategy.name, "MA_Trend")
        self.assertEqual(self.strategy.min_required_candles, 60)  # slow_period + 10

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # Fast period >= slow period should raise ValueError
            MovingAverageStrategy(fast_period=50, slow_period=20)

    def test_calculate_indicators(self):
        """Test indicator calculation."""
        # Create sample data (100 candles)
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators
        result = self.strategy.calculate_indicators(data)

        # Check that expected columns were added
        self.assertIn('fast_ema', result.columns)
        self.assertIn('slow_ema', result.columns)
        self.assertIn('ema_diff', result.columns)
        self.assertIn('crossover', result.columns)
        self.assertIn('swing_high', result.columns)
        self.assertIn('swing_low', result.columns)

        # Verify EMA calculations
        self.assertEqual(len(result['fast_ema']), 100)
        self.assertEqual(len(result['slow_ema']), 100)

        # Verify crossover signal values are -1, 0, or 1
        self.assertTrue(all(result['crossover'].isin([-1, 0, 1])))

    def test_analyze_bullish_crossover(self):
        """Test generation of bullish crossover signal."""
        # Create sample data with a bullish crossover on the last candle
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators manually
        data['fast_ema'] = data['close'].ewm(span=20, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=50, adjust=False).mean()

        # Set up crossover condition
        data['ema_diff'] = data['fast_ema'] - data['slow_ema']
        data['crossover'] = 0

        # Create a bullish crossover on the last candle
        data.loc[data.index[-2], 'ema_diff'] = -1  # Previous candle: fast EMA below slow EMA
        data.loc[data.index[-1], 'ema_diff'] = 1  # Last candle: fast EMA above slow EMA
        data.loc[data.index[-1], 'crossover'] = 1  # Set crossover signal

        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Patch the calculate_indicators method to return our prepared data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.signal_type, "BUY")
        self.assertAlmostEqual(signal.price, data['close'].iloc[-1])
        self.assertEqual(signal.strength, 0.7)

        # Verify metadata
        import json
        metadata = json.loads(signal.metadata)
        self.assertIn('fast_ema', metadata)
        self.assertIn('slow_ema', metadata)
        self.assertIn('stop_loss', metadata)
        self.assertIn('reason', metadata)
        self.assertEqual(metadata['reason'], 'Bullish EMA crossover')

    def test_analyze_bearish_crossover(self):
        """Test generation of bearish crossover signal."""
        # Create sample data with a bearish crossover on the last candle
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators manually
        data['fast_ema'] = data['close'].ewm(span=20, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=50, adjust=False).mean()

        # Set up crossover condition
        data['ema_diff'] = data['fast_ema'] - data['slow_ema']
        data['crossover'] = 0

        # Create a bearish crossover on the last candle
        data.loc[data.index[-2], 'ema_diff'] = 1  # Previous candle: fast EMA above slow EMA
        data.loc[data.index[-1], 'ema_diff'] = -1  # Last candle: fast EMA below slow EMA
        data.loc[data.index[-1], 'crossover'] = -1  # Set crossover signal

        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Patch the calculate_indicators method to return our prepared data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.signal_type, "SELL")
        self.assertAlmostEqual(signal.price, data['close'].iloc[-1])
        self.assertEqual(signal.strength, 0.7)

        # Verify metadata
        import json
        metadata = json.loads(signal.metadata)
        self.assertIn('fast_ema', metadata)
        self.assertIn('slow_ema', metadata)
        self.assertIn('stop_loss', metadata)
        self.assertIn('reason', metadata)
        self.assertEqual(metadata['reason'], 'Bearish EMA crossover')

    def test_analyze_pullback_entry_bullish(self):
        """Test generation of bullish pullback entry signal."""
        # Create sample data with a pullback to fast EMA in uptrend
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators manually
        data['fast_ema'] = data['close'].ewm(span=20, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=50, adjust=False).mean()
        data['ema_diff'] = data['fast_ema'] - data['slow_ema']
        data['crossover'] = 0

        # Set up uptrend pullback conditions
        # 1. fast_ema > slow_ema (uptrend)
        data.loc[data.index[-1], 'fast_ema'] = 1810
        data.loc[data.index[-1], 'slow_ema'] = 1800

        # 2. last candle closed above fast_ema
        data.loc[data.index[-1], 'close'] = 1815

        # 3. previous candle touched/crossed below fast_ema
        data.loc[data.index[-2], 'low'] = 1805
        data.loc[data.index[-2], 'fast_ema'] = 1808

        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Patch the calculate_indicators method to return our prepared data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.signal_type, "BUY")
        self.assertAlmostEqual(signal.price, data['close'].iloc[-1])
        self.assertEqual(signal.strength, 0.6)  # Lower strength for pullback entries

        # Verify metadata
        import json
        metadata = json.loads(signal.metadata)
        self.assertIn('reason', metadata)
        self.assertEqual(metadata['reason'], 'Pullback to fast EMA in uptrend')

    def test_no_signal_generation(self):
        """Test that no signals are generated when conditions aren't met."""
        # Create sample data with no signal conditions
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators manually
        data['fast_ema'] = data['close'].ewm(span=20, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=50, adjust=False).mean()
        data['ema_diff'] = data['fast_ema'] - data['slow_ema']
        data['crossover'] = 0  # No crossover

        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Ensure no pullback conditions are met either
        data.loc[data.index[-1], 'fast_ema'] = 1810
        data.loc[data.index[-1], 'slow_ema'] = 1800
        data.loc[data.index[-1], 'close'] = 1805  # Between fast and slow EMA
        data.loc[data.index[-2], 'low'] = 1815  # Did not touch fast EMA
        data.loc[data.index[-2], 'fast_ema'] = 1810

        # Patch the calculate_indicators method to return our prepared data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify no signals were generated
        self.assertEqual(len(signals), 0)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create small dataset with too few candles
        dates = [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 10),
            'high': np.random.normal(1810, 10, 10),
            'low': np.random.normal(1790, 10, 10),
            'close': np.random.normal(1800, 10, 10),
            'volume': np.random.normal(1000, 100, 10)
        }, index=dates)

        # Call analyze and verify no signals are returned
        signals = self.strategy.analyze(data)
        self.assertEqual(len(signals), 0)

    if __name__ == '__main__':
        unittest.main()