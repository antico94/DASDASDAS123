# tests/test_enhanced_moving_average_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.moving_average import EnhancedMovingAverageStrategy


class TestEnhancedMovingAverageStrategy(unittest.TestCase):
    """Unit tests for the Enhanced Moving Average Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = EnhancedMovingAverageStrategy(
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
        self.assertEqual(self.strategy.name, "EnhancedMA_Trend")
        self.assertEqual(self.strategy.min_required_candles, 80)  # slow_period + 30

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # Fast period >= slow period should raise ValueError
            EnhancedMovingAverageStrategy(fast_period=50, slow_period=20)

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
        self.assertIn('trend_bias', result.columns)
        self.assertIn('atr', result.columns)
        self.assertIn('swing_high', result.columns)
        self.assertIn('swing_low', result.columns)
        self.assertIn('pullback_to_fast_ema', result.columns)

        # Verify EMA calculations
        self.assertEqual(len(result['fast_ema']), 100)
        self.assertEqual(len(result['slow_ema']), 100)

        # Verify crossover signal values are -1, 0, or 1
        self.assertTrue(all(result['crossover'].isin([-1, 0, 1])))

        # Verify trend_bias values are -1, 0, or 1
        self.assertTrue(all(result['trend_bias'].isin([-1, 0, 1])))

        # Verify pullback_to_fast_ema values are -1, 0, or 1
        self.assertTrue(all(result['pullback_to_fast_ema'].isin([-1, 0, 1])))

    def test_bullish_crossover_signal(self):
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

        # Set up a bullish crossover
        data = self.strategy.calculate_indicators(data)

        # Manually set up a bullish crossover on the last candle
        data.loc[data.index[-2], 'fast_ema'] = 1795  # Previous candle: fast EMA below slow EMA
        data.loc[data.index[-2], 'slow_ema'] = 1800
        data.loc[data.index[-2], 'ema_diff'] = -5

        data.loc[data.index[-1], 'fast_ema'] = 1805  # Last candle: fast EMA crosses above slow EMA
        data.loc[data.index[-1], 'slow_ema'] = 1800
        data.loc[data.index[-1], 'ema_diff'] = 5
        data.loc[data.index[-1], 'crossover'] = 1  # Bullish crossover
        data.loc[data.index[-1], 'trend_bias'] = 1  # Bullish trend
        data.loc[data.index[-1], 'atr'] = 10  # Set ATR for stop calculation
        data.loc[data.index[-5:-1], 'swing_low'] = 1785  # Set recent swing lows for stop placement

        # Mock the _generate_bullish_crossover_signal method to use our data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "BUY")
        self.assertEqual(signals[0].strength, 0.8)  # Strength for crossover signals

        # Check metadata
        import json
        metadata = json.loads(signals[0].metadata)
        self.assertIn('fast_ema', metadata)
        self.assertIn('slow_ema', metadata)
        self.assertIn('stop_loss', metadata)
        self.assertIn('take_profit_1r', metadata)
        self.assertIn('take_profit_2r', metadata)
        self.assertIn('signal_type', metadata)
        self.assertEqual(metadata['signal_type'], 'crossover')
        self.assertEqual(metadata['reason'], 'Bullish EMA crossover')

    def test_bearish_crossover_signal(self):
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

        # Set up a bearish crossover
        data = self.strategy.calculate_indicators(data)

        # Manually set up a bearish crossover on the last candle
        data.loc[data.index[-2], 'fast_ema'] = 1805  # Previous candle: fast EMA above slow EMA
        data.loc[data.index[-2], 'slow_ema'] = 1800
        data.loc[data.index[-2], 'ema_diff'] = 5

        data.loc[data.index[-1], 'fast_ema'] = 1795  # Last candle: fast EMA crosses below slow EMA
        data.loc[data.index[-1], 'slow_ema'] = 1800
        data.loc[data.index[-1], 'ema_diff'] = -5
        data.loc[data.index[-1], 'crossover'] = -1  # Bearish crossover
        data.loc[data.index[-1], 'trend_bias'] = -1  # Bearish trend
        data.loc[data.index[-1], 'atr'] = 10  # Set ATR for stop calculation
        data.loc[data.index[-5:-1], 'swing_high'] = 1815  # Set recent swing highs for stop placement

        # Mock the calculate_indicators method to use our data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "SELL")
        self.assertEqual(signals[0].strength, 0.8)  # Strength for crossover signals

        # Check metadata
        import json
        metadata = json.loads(signals[0].metadata)
        self.assertIn('fast_ema', metadata)
        self.assertIn('slow_ema', metadata)
        self.assertIn('stop_loss', metadata)
        self.assertIn('take_profit_1r', metadata)
        self.assertIn('take_profit_2r', metadata)
        self.assertIn('signal_type', metadata)
        self.assertEqual(metadata['signal_type'], 'crossover')
        self.assertEqual(metadata['reason'], 'Bearish EMA crossover')

    def test_bullish_pullback_signal(self):
        """Test generation of bullish pullback signal."""
        # Create sample data with a bullish pullback setup
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Set up a bullish pullback
        data = self.strategy.calculate_indicators(data)

        # Create a bullish trend environment
        data.loc[data.index[-5:], 'trend_bias'] = 1  # Bullish trend for several bars

        # Last candle shows a pullback to EMA and bounce
        data.loc[data.index[-1], 'fast_ema'] = 1800
        data.loc[data.index[-1], 'slow_ema'] = 1790
        data.loc[data.index[-1], 'low'] = 1798  # Dipped to/below fast EMA
        data.loc[data.index[-1], 'close'] = 1805  # But closed above
        data.loc[data.index[-1], 'crossover'] = 0  # No crossover
        data.loc[data.index[-1], 'pullback_to_fast_ema'] = 1  # Bullish pullback to fast EMA
        data.loc[data.index[-1], 'atr'] = 10  # Set ATR for stop calculation
        data.loc[data.index[-5:-1], 'swing_low'] = 1785  # Set recent swing lows for stop placement

        # Mock the calculate_indicators method to use our data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "BUY")
        self.assertEqual(signals[0].strength, 0.7)  # Strength for pullback entry

        # Check metadata
        import json
        metadata = json.loads(signals[0].metadata)
        self.assertIn('signal_type', metadata)
        self.assertEqual(metadata['signal_type'], 'pullback')
        self.assertEqual(metadata['reason'], 'Pullback to fast EMA in uptrend')

    def test_bearish_pullback_signal(self):
        """Test generation of bearish pullback signal."""
        # Create sample data with a bearish pullback setup
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Set up a bearish pullback
        data = self.strategy.calculate_indicators(data)

        # Create a bearish trend environment
        data.loc[data.index[-5:], 'trend_bias'] = -1  # Bearish trend for several bars

        # Last candle shows a pullback to EMA and rejection
        data.loc[data.index[-1], 'fast_ema'] = 1800
        data.loc[data.index[-1], 'slow_ema'] = 1810
        data.loc[data.index[-1], 'high'] = 1802  # Rose to/above fast EMA
        data.loc[data.index[-1], 'close'] = 1795  # But closed below
        data.loc[data.index[-1], 'crossover'] = 0  # No crossover
        data.loc[data.index[-1], 'pullback_to_fast_ema'] = -1  # Bearish pullback to fast EMA
        data.loc[data.index[-1], 'atr'] = 10  # Set ATR for stop calculation
        data.loc[data.index[-5:-1], 'swing_high'] = 1815  # Set recent swing highs for stop placement

        # Mock the calculate_indicators method to use our data
        with patch.object(self.strategy, 'calculate_indicators', return_value=data):
            signals = self.strategy.analyze(data)

        # Verify signal generation
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, "SELL")
        self.assertEqual(signals[0].strength, 0.7)  # Strength for pullback entry

        # Check metadata
        import json
        metadata = json.loads(signals[0].metadata)
        self.assertIn('signal_type', metadata)
        self.assertEqual(metadata['signal_type'], 'pullback')
        self.assertEqual(metadata['reason'], 'Pullback to fast EMA in downtrend')

    def test_no_signal_generation(self):
        """Test that no signals are generated when conditions aren't met."""
        # Create sample data with no signal conditions
        dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 1, 100),  # Very tight range, no clear trend
            'high': np.random.normal(1802, 1, 100),
            'low': np.random.normal(1798, 1, 100),
            'close': np.random.normal(1800, 1, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators
        result = self.strategy.calculate_indicators(data)

        # Make sure there's no signal
        result.loc[result.index[-1], 'crossover'] = 0
        result.loc[result.index[-1], 'pullback_to_fast_ema'] = 0

        # Mock the calculate_indicators method to use our data
        with patch.object(self.strategy, 'calculate_indicators', return_value=result):
            signals = self.strategy.analyze(data)

        # Verify no signals were generated
        self.assertEqual(len(signals), 0)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create small dataset with too few candles
        dates = [datetime.now() - timedelta(hours=i) for i in range(20, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 20),
            'high': np.random.normal(1810, 10, 20),
            'low': np.random.normal(1790, 10, 20),
            'close': np.random.normal(1800, 10, 20),
            'volume': np.random.normal(1000, 100, 20)
        }, index=dates)

        # Call analyze and verify no signals are returned
        signals = self.strategy.analyze(data)
        self.assertEqual(len(signals), 0)


if __name__ == '__main__':
    unittest.main()