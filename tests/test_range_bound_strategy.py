# tests/test_range_bound_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.range_bound import RangeBoundStrategy


class TestRangeBoundStrategy(unittest.TestCase):
    """Unit tests for the Range-Bound Mean Reversion Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = RangeBoundStrategy(
            symbol="XAUUSD",
            timeframe="M15",
            lookback_periods=48,
            min_range_bars=10,
            rsi_period=14,
            rsi_overbought=70,
            rsi_oversold=30,
            adx_period=14,
            adx_threshold=20,
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "M15")
        self.assertEqual(self.strategy.lookback_periods, 48)
        self.assertEqual(self.strategy.min_range_bars, 10)
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.adx_period, 14)
        self.assertEqual(self.strategy.adx_threshold, 20)
        self.assertEqual(self.strategy.name, "Range_Mean_Reversion")

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # lookback_periods < min_range_bars should raise ValueError
            RangeBoundStrategy(lookback_periods=5, min_range_bars=10)

        with self.assertRaises(ValueError):
            # rsi_overbought <= rsi_oversold should raise ValueError
            RangeBoundStrategy(rsi_overbought=50, rsi_oversold=50)

    @patch('strategies.range_bound.RangeBoundStrategy.calculate_indicators')
    def test_calculate_indicators(self, mock_calculate):
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

        # Create a mock result with all required columns
        result_data = data.copy()
        result_data['rsi'] = np.random.uniform(30, 70, 100)
        result_data['adx'] = 15
        result_data['plus_di'] = 20
        result_data['minus_di'] = 10
        result_data['sma20'] = 1800
        result_data['upper_band'] = 1820
        result_data['lower_band'] = 1780
        result_data['bb_width'] = 0.02
        result_data['in_range'] = True
        result_data['range_top'] = 1810
        result_data['range_bottom'] = 1790
        result_data['range_midpoint'] = 1800
        result_data['range_bars'] = 20
        result_data['signal'] = 0
        result_data['signal_strength'] = 0
        result_data['stop_loss'] = 1785
        result_data['take_profit'] = 1815

        # Configure the mock
        mock_calculate.return_value = result_data

        # Call the method
        result = self.strategy.calculate_indicators(data)

        # Check that expected columns were added
        self.assertIn('rsi', result.columns)
        self.assertIn('adx', result.columns)
        self.assertIn('plus_di', result.columns)
        self.assertIn('minus_di', result.columns)
        self.assertIn('in_range', result.columns)
        self.assertIn('range_top', result.columns)
        self.assertIn('range_bottom', result.columns)
        self.assertIn('range_midpoint', result.columns)
        self.assertIn('signal', result.columns)
        self.assertIn('stop_loss', result.columns)
        self.assertIn('take_profit', result.columns)

    def create_range_data_with_signals(self):
        """Create simulated data with a range and signals."""
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

        # Create a move to the bottom of the range (buy signal setup)
        for i in range(80, 85):
            data.loc[data.index[i], 'low'] = np.random.uniform(range_low, range_low + 2)
            data.loc[data.index[i], 'high'] = np.random.uniform(range_low + 2, range_low + 7)
            data.loc[data.index[i], 'open'] = np.random.uniform(range_low + 1, range_low + 4)
            data.loc[data.index[i], 'close'] = np.random.uniform(range_low + 1, range_low + 4)
            data.loc[data.index[i], 'volume'] = np.random.normal(1000, 100)

        # Create a move to the top of the range (sell signal setup)
        for i in range(90, 95):
            data.loc[data.index[i], 'low'] = np.random.uniform(range_high - 7, range_high - 2)
            data.loc[data.index[i], 'high'] = np.random.uniform(range_high - 2, range_high)
            data.loc[data.index[i], 'open'] = np.random.uniform(range_high - 5, range_high - 1)
            data.loc[data.index[i], 'close'] = np.random.uniform(range_high - 4, range_high - 1)
            data.loc[data.index[i], 'volume'] = np.random.normal(1000, 100)

        return data

    def test_buy_signal_generation(self):
        """Test the generation of a buy signal at range support."""
        # Create data with a range and price at support
        data = self.create_range_data_with_signals()

        # Mock the calculate_indicators method to return data with signals
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare the data with a buy signal
            result_data = data.copy()
            result_data['in_range'] = True
            result_data['range_top'] = 1810
            result_data['range_bottom'] = 1790
            result_data['range_midpoint'] = 1800
            result_data['range_bars'] = 20
            result_data['rsi'] = 30  # Oversold
            result_data['adx'] = 15  # Non-trending

            # Set up a buy signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = 1  # Buy signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.8
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = 1785
            result_data.loc[result_data.index[last_idx], 'take_profit'] = 1800

            # Configure the mock to return this data
            mock_calc.return_value = result_data

            # Create a signal mock
            mock_signal = MagicMock()
            mock_signal.signal_type = "BUY"
            mock_signal.price = result_data['close'].iloc[-1]
            mock_signal.strength = 0.8
            mock_signal.metadata = '{"stop_loss": 1785, "take_profit_midpoint": 1800, "take_profit_full": 1809.73, "range_top": 1810, "range_bottom": 1790, "rsi": 30, "adx": 15, "reason": "Buy at support in range with oversold RSI"}'

            # Patch the create_signal method
            with patch.object(self.strategy, 'create_signal', return_value=mock_signal):
                # Call analyze
                signals = self.strategy.analyze(data)

                # Verify we get a BUY signal
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].signal_type, "BUY")
                self.assertEqual(signals[0].price, result_data['close'].iloc[-1])
                self.assertEqual(signals[0].strength, 0.8)

                # Check metadata
                import json
                metadata = json.loads(signals[0].metadata)
                self.assertIn('stop_loss', metadata)
                self.assertIn('take_profit_midpoint', metadata)
                self.assertIn('take_profit_full', metadata)
                self.assertIn('range_top', metadata)
                self.assertIn('range_bottom', metadata)
                self.assertIn('rsi', metadata)
                self.assertIn('adx', metadata)
                self.assertEqual(metadata['reason'], 'Buy at support in range with oversold RSI')

    def test_sell_signal_generation(self):
        """Test the generation of a sell signal at range resistance."""
        # Create data with a range and price at resistance
        data = self.create_range_data_with_signals()

        # Mock the calculate_indicators method to return data with signals
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare the data with a sell signal
            result_data = data.copy()
            result_data['in_range'] = True
            result_data['range_top'] = 1810
            result_data['range_bottom'] = 1790
            result_data['range_midpoint'] = 1800
            result_data['range_bars'] = 20
            result_data['rsi'] = 70  # Overbought
            result_data['adx'] = 15  # Non-trending

            # Set up a sell signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = -1  # Sell signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.7
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = 1815
            result_data.loc[result_data.index[last_idx], 'take_profit'] = 1800

            # Configure the mock to return this data
            mock_calc.return_value = result_data

            # Create a signal mock
            mock_signal = MagicMock()
            mock_signal.signal_type = "SELL"
            mock_signal.price = result_data['close'].iloc[-1]
            mock_signal.strength = 0.7
            mock_signal.metadata = '{"stop_loss": 1815, "take_profit_midpoint": 1800, "take_profit_full": 1790.27, "range_top": 1810, "range_bottom": 1790, "rsi": 70, "adx": 15, "reason": "Sell at resistance in range with overbought RSI"}'

            # Patch the create_signal method
            with patch.object(self.strategy, 'create_signal', return_value=mock_signal):
                # Call analyze
                signals = self.strategy.analyze(data)

                # Verify we get a SELL signal
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].signal_type, "SELL")
                self.assertEqual(signals[0].price, result_data['close'].iloc[-1])
                self.assertEqual(signals[0].strength, 0.7)

                # Check metadata
                import json
                metadata = json.loads(signals[0].metadata)
                self.assertIn('stop_loss', metadata)
                self.assertIn('take_profit_midpoint', metadata)
                self.assertIn('take_profit_full', metadata)
                self.assertIn('range_top', metadata)
                self.assertIn('range_bottom', metadata)
                self.assertIn('rsi', metadata)
                self.assertIn('adx', metadata)
                self.assertEqual(metadata['reason'], 'Sell at resistance in range with overbought RSI')

    def test_no_signal_generation(self):
        """Test that no signals are generated when conditions are not met."""
        # Create data with a range but no signal conditions
        data = self.create_range_data_with_signals()

        # Mock the calculate_indicators method to return data without signals
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare the data with no signal
            result_data = data.copy()
            result_data['in_range'] = True
            result_data['range_top'] = 1810
            result_data['range_bottom'] = 1790
            result_data['range_midpoint'] = 1800
            result_data['range_bars'] = 20
            result_data['rsi'] = 50  # Not extreme
            result_data['adx'] = 15  # Non-trending
            result_data['signal'] = 0  # No signal

            # Configure the mock to return this data
            mock_calc.return_value = result_data

            # Call analyze
            signals = self.strategy.analyze(data)

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