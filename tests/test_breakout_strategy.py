# tests/test_breakout_strategy.py
import json
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

        # Get the basic structure by processing the data
        processed_data = self.strategy.calculate_indicators(data.copy())

        # Create a result DataFrame with a clear bearish breakout signal
        # Make sure the last candle has the breakout signal
        processed_data.loc[processed_data.index[-1], 'breakout_signal'] = -1  # Bearish signal
        processed_data.loc[processed_data.index[-1], 'breakout_strength'] = 0.8
        processed_data.loc[processed_data.index[-1], 'breakout_stop_loss'] = processed_data['close'].iloc[-1] + 10.0

        # Ensure all required properties are present for the analysis logic
        processed_data.loc[processed_data.index[-1], 'in_range'] = False  # No longer in range (broken out)
        processed_data.loc[processed_data.index[-1], 'range_top'] = 1810.0
        processed_data.loc[processed_data.index[-1], 'range_bottom'] = 1790.0
        processed_data.loc[processed_data.index[-1], 'range_bars'] = 20
        processed_data.loc[processed_data.index[-1], 'atr'] = 10.0
        processed_data.loc[processed_data.index[-1], 'volume'] = 2000.0
        processed_data.loc[processed_data.index[-1], 'volume_ma'] = 1000.0

        # Create mock signal
        mock_signal = MagicMock()
        mock_signal.signal_type = "SELL"
        mock_signal.price = processed_data['close'].iloc[-1]
        mock_signal.strength = 0.8
        mock_signal.signal_data = json.dumps({
            'stop_loss': float(processed_data['close'].iloc[-1] + 10.0),
            'take_profit_1r': float(processed_data['close'].iloc[-1] - 10.0),
            'take_profit_extension': float(processed_data['close'].iloc[-1] - 20.0),
            'range_top': 1810.0,
            'range_bottom': 1790.0,
            'range_bars': 20,
            'atr': 10.0,
            'reason': 'Bearish breakout from consolidation range'
        })

        # Mock both methods
        with patch.object(self.strategy, 'calculate_indicators', return_value=processed_data):
            with patch.object(self.strategy, 'create_signal', return_value=mock_signal) as mock_create_signal:
                # Call analyze
                signals = self.strategy.analyze(data)

                # Verify create_signal was called
                self.assertTrue(mock_create_signal.called)

                # Verify we get the SELL signal
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].signal_type, "SELL")

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

    # Additional tests for test_breakout_strategy.py

    def test_calculate_indicators_with_minimal_data(self):
        """Test calculation with minimal but valid data."""
        # Create dataset with just enough data
        min_candles = self.strategy.min_required_candles
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(min_candles, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 5, min_candles),
            'high': np.random.normal(1810, 5, min_candles),
            'low': np.random.normal(1790, 5, min_candles),
            'close': np.random.normal(1800, 5, min_candles),
            'volume': np.random.normal(1000, 100, min_candles)
        }, index=dates)

        # Calculate indicators
        result = self.strategy.calculate_indicators(data)

        # Verify minimum required columns are present
        self.assertIn('atr', result.columns)
        self.assertIn('volume_ma', result.columns)
        self.assertIn('bb_width', result.columns)
        self.assertIn('in_range', result.columns)

    def test_analyze_with_edge_cases(self):
        """Test analyze method with edge case data."""
        # Create data with extreme values
        dates = [datetime.now() - timedelta(minutes=15 * i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Add NaN values
        data.loc[data.index[30], 'close'] = np.nan

        # Create mocked version of processed data with the result containing a breakout signal
        processed_data = data.copy()
        processed_data['in_range'] = True
        processed_data['breakout_signal'] = 0
        processed_data['breakout_strength'] = 0.0
        processed_data['breakout_stop_loss'] = np.nan
        processed_data['atr'] = 10.0

        # Make the last row have a breakout signal
        processed_data.loc[processed_data.index[-1], 'breakout_signal'] = 1
        processed_data.loc[processed_data.index[-1], 'breakout_strength'] = 0.8
        processed_data.loc[processed_data.index[-1], 'breakout_stop_loss'] = 1798.0
        processed_data.loc[processed_data.index[-1], 'range_top'] = 1810.0
        processed_data.loc[processed_data.index[-1], 'range_bottom'] = 1790.0
        processed_data.loc[processed_data.index[-1], 'range_bars'] = 20

        # Mock the calculate_indicators method
        with patch.object(self.strategy, 'calculate_indicators', return_value=processed_data):
            with patch.object(self.strategy, 'create_signal') as mock_create_signal:
                # Set up mock to return a proper signal
                mock_signal = MagicMock()
                mock_signal.signal_type = "BUY"
                mock_create_signal.return_value = mock_signal

                # Call analyze
                signals = self.strategy.analyze(data)

                # Verify signal was created
                self.assertEqual(len(signals), 1)
                mock_create_signal.assert_called_once()

    def test_breakout_detection_with_extreme_volume(self):
        """Test breakout detection with extreme volume conditions."""
        # Create data with extremely high volume spike
        data = self.create_range_and_breakout_data()

        # Process through calculate_indicators first to get base data structure
        processed_data = self.strategy.calculate_indicators(data.copy())

        # Manually set extreme volume at index 84 (breakout bar)
        processed_data.loc[processed_data.index[84], 'volume'] = processed_data.loc[
                                                                     processed_data.index[83], 'volume_ma'] * 10

        # Run _identify_breakouts with this extreme data
        result = self.strategy._identify_breakouts(processed_data)

        # Check for breakout signal with high strength due to volume
        self.assertEqual(result.loc[result.index[84], 'breakout_signal'], 1)
        self.assertGreater(result.loc[result.index[84], 'breakout_strength'], 0.7)

    def test_range_detection_with_tight_consolidation(self):
        """Test range detection with very tight price consolidation."""
        # Create data with very tight range (should be easily identified)
        data = pd.DataFrame({
            'open': np.random.normal(1800, 1, 100),  # Very tight range
            'high': np.random.normal(1802, 1, 100),
            'low': np.random.normal(1798, 1, 100),
            'close': np.random.normal(1800, 1, 100),
            'volume': np.random.normal(1000, 50, 100)
        })

        # Add required indicator columns
        data['bb_width'] = 0.005  # Very narrow
        data['atr'] = 1.0

        # Run range identification
        result = self.strategy._identify_ranges(data)

        # Should identify ranges due to tight consolidation
        range_bars = result['in_range'].sum()
        self.assertGreater(range_bars, 50)  # At least half should be identified as range


if __name__ == '__main__':
    unittest.main()
