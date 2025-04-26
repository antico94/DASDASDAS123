# tests/test_breakout_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.breakout import BreakoutStrategy


class TestBreakoutStrategy(unittest.TestCase):
    """Unit tests for the Breakout Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = BreakoutStrategy(
            symbol="XAUUSD",
            timeframe="M15",
            lookback_periods=48,
            min_range_bars=10,
            volume_threshold=1.5,
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "M15")
        self.assertEqual(self.strategy.lookback_periods, 48)
        self.assertEqual(self.strategy.min_range_bars, 10)
        self.assertEqual(self.strategy.volume_threshold, 1.5)
        self.assertEqual(self.strategy.name, "Breakout")
        self.assertEqual(self.strategy.min_required_candles, 68)  # lookback_periods + 20

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # lookback_periods < min_range_bars should raise ValueError
            BreakoutStrategy(lookback_periods=5, min_range_bars=10)

    def test_calculate_indicators(self):
        """Test indicator calculation."""
        # Create sample data (100 candles)
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(100, 0, -1)]
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
        self.assertIn('atr', result.columns)
        self.assertIn('volume_ma', result.columns)
        self.assertIn('middle_band', result.columns)
        self.assertIn('upper_band', result.columns)
        self.assertIn('lower_band', result.columns)
        self.assertIn('bb_width', result.columns)
        self.assertIn('in_range', result.columns)
        self.assertIn('range_top', result.columns)
        self.assertIn('range_bottom', result.columns)
        self.assertIn('range_bars', result.columns)
        self.assertIn('breakout_signal', result.columns)
        self.assertIn('breakout_strength', result.columns)
        self.assertIn('breakout_stop_loss', result.columns)

    def create_range_and_breakout_data(self):
        """Create simulated data with a range and a breakout."""
        # Create 100 candles
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(100, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Create a range in the middle (candles 30-80)
        range_low = 1790
        range_high = 1810

        for i in range(30, 80):
            # Prices stay within the range
            data.loc[data.index[i], 'low'] = np.random.uniform(range_low, range_low + 5)
            data.loc[data.index[i], 'high'] = np.random.uniform(range_high - 5, range_high)
            data.loc[data.index[i], 'open'] = np.random.uniform(range_low + 3, range_high - 3)
            data.loc[data.index[i], 'close'] = np.random.uniform(range_low + 3, range_high - 3)
            data.loc[data.index[i], 'volume'] = np.random.normal(1000, 100)

        # Create a breakout on candles 81-85 (bullish)
        for i in range(80, 85):
            # Increasing breakout pattern
            breakout_percent = (i - 79) * 0.2  # 0.2%, 0.4%, 0.6%, 0.8%, 1.0%
            data.loc[data.index[i], 'low'] = range_high * (1 + breakout_percent * 0.5 / 100)
            data.loc[data.index[i], 'high'] = range_high * (1 + breakout_percent * 1.5 / 100)
            data.loc[data.index[i], 'open'] = range_high * (1 + breakout_percent * 0.7 / 100)
            data.loc[data.index[i], 'close'] = range_high * (1 + breakout_percent * 1.2 / 100)
            # Increase volume on breakout
            data.loc[data.index[i], 'volume'] = 1000 * (2 + breakout_percent)

        # Create a bearish breakout on candles 90-95
        for i in range(90, 95):
            # Decreasing breakout pattern
            breakout_percent = (i - 89) * 0.2  # 0.2%, 0.4%, 0.6%, 0.8%, 1.0%
            data.loc[data.index[i], 'high'] = range_low * (1 - breakout_percent * 0.5 / 100)
            data.loc[data.index[i], 'low'] = range_low * (1 - breakout_percent * 1.5 / 100)
            data.loc[data.index[i], 'open'] = range_low * (1 - breakout_percent * 0.7 / 100)
            data.loc[data.index[i], 'close'] = range_low * (1 - breakout_percent * 1.2 / 100)
            # Increase volume on breakout
            data.loc[data.index[i], 'volume'] = 1000 * (2 + breakout_percent)

        return data

    def test_bullish_breakout_detection(self):
        """Test the detection of a bullish breakout."""
        # Create data with a bullish breakout
        data = self.create_range_and_breakout_data()

        # Process the data
        result = self.strategy.calculate_indicators(data)

        # Check that a range was identified
        self.assertTrue(result['in_range'].iloc[79])  # Last candle in the range

        # Check that a breakout was identified
        self.assertEqual(result['breakout_signal'].iloc[84], 1)  # Bullish breakout
        self.assertTrue(result['breakout_strength'].iloc[84] > 0.5)  # Strong breakout
        self.assertTrue(~np.isnan(result['breakout_stop_loss'].iloc[84]))  # Stop loss was calculated

        # Now test the analyze method with a mock
        with patch.object(self.strategy, 'calculate_indicators', return_value=result):
            # Set the last candle to be the breakout candle
            last_candle_index = result.index[84]
            mocked_result = result.loc[:last_candle_index]

            signals = self.strategy.analyze(mocked_result)

            # Verify we get a BUY signal
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].signal_type, "BUY")
            self.assertAlmostEqual(signals[0].price, result['close'].iloc[84])

            # Check metadata
            import json
            metadata = json.loads(signals[0].metadata)
            self.assertIn('stop_loss', metadata)
            self.assertIn('take_profit_1r', metadata)
            self.assertIn('take_profit_extension', metadata)
            self.assertIn('reason', metadata)
            self.assertEqual(metadata['reason'], 'Bullish breakout from consolidation range')

    def test_bearish_breakout_detection(self):
        """Test the detection of a bearish breakout."""
        # Create data with a bearish breakout
        data = self.create_range_and_breakout_data()

        # Process the data
        result = self.strategy.calculate_indicators(data)

        # Check that a bearish breakout was identified
        self.assertEqual(result['breakout_signal'].iloc[94], -1)  # Bearish breakout
        self.assertTrue(result['breakout_strength'].iloc[94] > 0.5)  # Strong breakout
        self.assertTrue(~np.isnan(result['breakout_stop_loss'].iloc[94]))  # Stop loss was calculated

        # Now test the analyze method with a mock
        with patch.object(self.strategy, 'calculate_indicators', return_value=result):
            # Set the last candle to be the breakout candle
            last_candle_index = result.index[94]
            mocked_result = result.loc[:last_candle_index]

            signals = self.strategy.analyze(mocked_result)

            # Verify we get a SELL signal
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].signal_type, "SELL")
            self.assertAlmostEqual(signals[0].price, result['close'].iloc[94])

            # Check metadata
            import json
            metadata = json.loads(signals[0].metadata)
            self.assertIn('stop_loss', metadata)
            self.assertIn('take_profit_1r', metadata)
            self.assertIn('take_profit_extension', metadata)
            self.assertIn('reason', metadata)
            self.assertEqual(metadata['reason'], 'Bearish breakout from consolidation range')

    def test_no_breakout_signals(self):
        """Test that no signals are generated when there's no breakout."""
        # Create data without breakouts (all range)
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(100, 0, -1)]

        data = pd.DataFrame({
            'open': np.random.uniform(1795, 1805, 100),
            'high': np.random.uniform(1800, 1810, 100),
            'low': np.random.uniform(1790, 1800, 100),
            'close': np.random.uniform(1795, 1805, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Process the data
        result = self.strategy.calculate_indicators(data)

        # Check for any breakout signals
        self.assertTrue((result['breakout_signal'] == 0).all())  # No breakout signals

        # Test the analyze method
        with patch.object(self.strategy, 'calculate_indicators', return_value=result):
            signals = self.strategy.analyze(result)

            # Verify we get no signals
            self.assertEqual(len(signals), 0)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create small dataset with too few candles
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(10, 0, -1)]
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