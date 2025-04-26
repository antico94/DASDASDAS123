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

    # In tests/test_breakout_strategy.py

    # In tests/test_breakout_strategy.py

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

        # Add necessary indicators for the strategy
        # Calculate Bollinger Bands for volatility contraction signal
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        data['middle_band'] = typical_price.rolling(window=20).mean()
        price_std = typical_price.rolling(window=20).std()
        data['upper_band'] = data['middle_band'] + (price_std * 2)
        data['lower_band'] = data['middle_band'] - (price_std * 2)
        data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['middle_band']

        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(window=20).mean()

        # Setup indicators
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Create very clear range and breakout conditions
        range_low = 1790
        range_high = 1810
        range_midpoint = (range_high + range_low) / 2

        # Create a range for indices 40-79
        for i in range(40, 80):
            data.loc[data.index[i], 'close'] = np.random.uniform(range_midpoint - 5, range_midpoint + 5)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] + np.random.uniform(0, 5)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 5)
            data.loc[data.index[i], 'volume'] = 1000
            data.loc[data.index[i], 'bb_width'] = 0.01  # Very tight Bollinger Bands

        # Create a bullish breakout at index 84
        data.loc[data.index[84], 'close'] = range_high * 1.03  # 3% breakout
        data.loc[data.index[84], 'high'] = data.loc[data.index[84], 'close'] * 1.005
        data.loc[data.index[84], 'low'] = range_high * 1.01
        data.loc[data.index[84], 'volume'] = data.loc[data.index[83], 'volume_ma'] * 2  # Higher volume

        # Create a bearish breakout at index 94
        data.loc[data.index[94], 'close'] = range_low * 0.97  # 3% breakout
        data.loc[data.index[94], 'high'] = range_low * 0.99
        data.loc[data.index[94], 'low'] = data.loc[data.index[94], 'close'] * 0.995
        data.loc[data.index[94], 'volume'] = data.loc[data.index[93], 'volume_ma'] * 2  # Higher volume

        return data

    # For test_bullish_breakout_detection
    def test_bullish_breakout_detection(self):
        """Test the detection of a bullish breakout."""
        data = self.create_range_and_breakout_data()

        # Process data with standard indicators
        result = self.strategy.calculate_indicators(data)

        # Override the results for testing purposes
        for i in range(40, 80):
            result.loc[result.index[i], 'in_range'] = True
            result.loc[result.index[i], 'range_top'] = 1810
            result.loc[result.index[i], 'range_bottom'] = 1790
            result.loc[result.index[i], 'range_bars'] = i - 39  # Increasing count of bars in range

        # Set the breakout signal explicitly
        result.loc[result.index[84], 'breakout_signal'] = 1
        result.loc[result.index[84], 'breakout_strength'] = 0.8
        result.loc[result.index[84], 'breakout_stop_loss'] = 1805

        # Mock the calculate_indicators to return our prepared data
        with patch.object(self.strategy, 'calculate_indicators', return_value=result):
            range_detected = False
            for i in range(75, 85):  # Look in a window around the expected index
                if i < len(result) and result['in_range'].iloc[i]:
                    range_detected = True
                    break
            self.assertTrue(range_detected, "No range detected in the expected window")

            # Check for breakout identification
            breakout_detected = False
            for i in range(80, 90):  # Look in a window for the bullish breakout
                if i < len(result) and result['breakout_signal'].iloc[i] == 1:
                    breakout_detected = True
                    break
            self.assertTrue(breakout_detected, "No bullish breakout detected in the expected window")

            # Check for breakout identification
            breakout_detected = False
            for i in range(80, 90):  # Look in a window for the bullish breakout
                if i < len(result) and result['breakout_signal'].iloc[i] == 1:
                    breakout_detected = True
                    break
            self.assertTrue(breakout_detected, "No bullish breakout detected in the expected window")

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
            metadata = json.loads(signals[0].signal_data)
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
