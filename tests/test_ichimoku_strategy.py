# tests/test_ichimoku_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.ichimoku import IchimokuStrategy


class TestIchimokuStrategy(unittest.TestCase):
    """Unit tests for the Ichimoku Cloud Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = IchimokuStrategy(
            symbol="XAUUSD",
            timeframe="H1",
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52,
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "H1")
        self.assertEqual(self.strategy.tenkan_period, 9)
        self.assertEqual(self.strategy.kijun_period, 26)
        self.assertEqual(self.strategy.senkou_b_period, 52)
        self.assertEqual(self.strategy.name, "Ichimoku_Cloud")
        self.assertEqual(self.strategy.min_required_candles, 134)  # kijun_period*2 + senkou_b_period + 30

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # tenkan_period >= kijun_period should raise ValueError
            IchimokuStrategy(tenkan_period=26, kijun_period=26)

        with self.assertRaises(ValueError):
            # kijun_period >= senkou_b_period should raise ValueError
            IchimokuStrategy(kijun_period=52, senkou_b_period=52)

    def test_calculate_indicators(self):
        """Test indicator calculation."""
        # Create sample data (200 candles)
        dates = [datetime.now() - timedelta(hours=i) for i in range(200, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 200),
            'high': np.random.normal(1810, 10, 200),
            'low': np.random.normal(1790, 10, 200),
            'close': np.random.normal(1800, 10, 200),
            'volume': np.random.normal(1000, 100, 200)
        }, index=dates)

        # Calculate indicators
        result = self.strategy.calculate_indicators(data)

        # Check that expected columns were added
        self.assertIn('tenkan_sen', result.columns)
        self.assertIn('kijun_sen', result.columns)
        self.assertIn('senkou_span_a', result.columns)
        self.assertIn('senkou_span_b', result.columns)
        self.assertIn('chikou_span', result.columns)
        self.assertIn('atr', result.columns)
        self.assertIn('cloud_bullish', result.columns)
        self.assertIn('signal', result.columns)
        self.assertIn('signal_strength', result.columns)
        self.assertIn('stop_loss', result.columns)
        self.assertIn('take_profit', result.columns)

    def create_bullish_ichimoku_data(self):
        """Create simulated data with a bullish Ichimoku setup."""
        # Create 200 candles
        dates = [datetime.now() - timedelta(hours=i) for i in range(200, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 200),
            'high': np.random.normal(1810, 10, 200),
            'low': np.random.normal(1790, 10, 200),
            'close': np.random.normal(1800, 10, 200),
            'volume': np.random.normal(1000, 100, 200)
        }, index=dates)

        # Create a bullish scenario for Ichimoku
        for i in range(150, 200):
            # Build a bullish trend
            trending_up = (i - 150) / 50.0  # 0.0 to 1.0 scale

            # Price increasing and moving above cloud
            base_price = 1780 + trending_up * 50  # 1780 to 1830

            # Add some noise to price
            data.loc[data.index[i], 'close'] = base_price + np.random.normal(0, 5)
            data.loc[data.index[i], 'open'] = base_price + np.random.normal(0, 5)
            data.loc[data.index[i], 'high'] = base_price + np.random.normal(5, 3)
            data.loc[data.index[i], 'low'] = base_price + np.random.normal(-5, 3)

        return data

    def create_bearish_ichimoku_data(self):
        """Create simulated data with a bearish Ichimoku setup."""
        # Create 200 candles
        dates = [datetime.now() - timedelta(hours=i) for i in range(200, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 200),
            'high': np.random.normal(1810, 10, 200),
            'low': np.random.normal(1790, 10, 200),
            'close': np.random.normal(1800, 10, 200),
            'volume': np.random.normal(1000, 100, 200)
        }, index=dates)

        # Create a bearish scenario for Ichimoku
        for i in range(150, 200):
            # Build a bearish trend
            trending_down = (i - 150) / 50.0  # 0.0 to 1.0 scale

            # Price decreasing and moving below cloud
            base_price = 1820 - trending_down * 50  # 1820 to 1770

            # Add some noise to price
            data.loc[data.index[i], 'close'] = base_price + np.random.normal(0, 5)
            data.loc[data.index[i], 'open'] = base_price + np.random.normal(0, 5)
            data.loc[data.index[i], 'high'] = base_price + np.random.normal(5, 3)
            data.loc[data.index[i], 'low'] = base_price + np.random.normal(-5, 3)

        return data

    # In tests/test_ichimoku_strategy.py
    def test_bullish_signal_generation(self):
        """Test the generation of a bullish signal."""
        # Create data with a bullish Ichimoku setup
        data = self.create_bullish_ichimoku_data()

        # Process the data with real indicator calculation to set up signals properly
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:  # Add this line with the patch
            # Prepare mock data with a buy signal
            result_data = self.strategy.calculate_indicators(data)

            # Make sure there is a buy signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = 1  # Buy signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.8
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = result_data['close'].iloc[-1] - 10
            result_data.loc[result_data.index[last_idx], 'take_profit'] = result_data['close'].iloc[-1] + 15
            result_data.loc[result_data.index[last_idx], 'tenkan_sen'] = result_data['close'].iloc[-1] - 5
            result_data.loc[result_data.index[last_idx], 'kijun_sen'] = result_data['close'].iloc[-1] - 10
            result_data.loc[result_data.index[last_idx], 'senkou_span_a'] = result_data['close'].iloc[-1] - 20
            result_data.loc[result_data.index[last_idx], 'senkou_span_b'] = result_data['close'].iloc[-1] - 25
            result_data.loc[result_data.index[last_idx], 'cloud_bullish'] = True

            # Configure the mock to return this data
            mock_calc.return_value = result_data

            # Call analyze
            signals = self.strategy.analyze(data)

            # Verify we get a BUY signal
            self.assertEqual(len(signals), 1)

    def test_bearish_signal_generation(self):
        """Test the generation of a bearish Ichimoku signal."""
        # Create data with a bearish Ichimoku setup
        data = self.create_bearish_ichimoku_data()

        # Process the data with real indicator calculation to set up signals properly
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare mock data with a sell signal
            result_data = self.strategy.calculate_indicators(data)

            # Make sure there is a sell signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = -1  # Sell signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.7
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = result_data['close'].iloc[-1] + 10
            result_data.loc[result_data.index[last_idx], 'take_profit'] = result_data['close'].iloc[-1] - 15
            result_data.loc[result_data.index[last_idx], 'tenkan_sen'] = result_data['close'].iloc[-1] + 5
            result_data.loc[result_data.index[last_idx], 'kijun_sen'] = result_data['close'].iloc[-1] + 10
            result_data.loc[result_data.index[last_idx], 'senkou_span_a'] = result_data['close'].iloc[-1] + 20
            result_data.loc[result_data.index[last_idx], 'senkou_span_b'] = result_data['close'].iloc[-1] + 25
            result_data.loc[result_data.index[last_idx], 'cloud_bullish'] = False

            # Configure the mock to return this data
            mock_calc.return_value = result_data

            # Call analyze
            signals = self.strategy.analyze(data)

            # Verify we get a SELL signal
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].signal_type, "SELL")
            self.assertEqual(signals[0].price, result_data['close'].iloc[-1])
            self.assertEqual(signals[0].strength, 0.7)

            # Check metadata
            import json
            metadata = json.loads(signals[0].signal_data)
            self.assertIn('stop_loss', metadata)
            self.assertIn('take_profit_1', metadata)
            self.assertIn('take_profit_2', metadata)
            self.assertIn('tenkan_sen', metadata)
            self.assertIn('kijun_sen', metadata)
            self.assertIn('cloud_bullish', metadata)
            self.assertEqual(metadata['reason'], 'Bearish TK cross below cloud with Chikou confirmation')

    def test_signal_identification(self):
        """Test that the strategy correctly identifies Ichimoku patterns."""
        # Create a basic dataset
        dates = [datetime.now() - timedelta(hours=i) for i in range(200, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 200),
            'high': np.random.normal(1810, 10, 200),
            'low': np.random.normal(1790, 10, 200),
            'close': np.random.normal(1800, 10, 200),
            'volume': np.random.normal(1000, 100, 200)
        }, index=dates)

        # Create the initial indicators
        base_data = self.strategy.calculate_indicators(data)

        # Manually create a TK cross bullish setup
        test_idx = 180

        # Setup the previous candle (Tenkan below Kijun)
        base_data.loc[base_data.index[test_idx - 1], 'tenkan_sen'] = 1795
        base_data.loc[base_data.index[test_idx - 1], 'kijun_sen'] = 1800

        # Setup the current candle (Tenkan crosses above Kijun)
        base_data.loc[base_data.index[test_idx], 'tenkan_sen'] = 1805
        base_data.loc[base_data.index[test_idx], 'kijun_sen'] = 1800

        # Price above cloud
        base_data.loc[base_data.index[test_idx], 'close'] = 1820
        base_data.loc[base_data.index[test_idx], 'senkou_span_a'] = 1790
        base_data.loc[base_data.index[test_idx], 'senkou_span_b'] = 1780

        # Cloud is bullish
        base_data.loc[base_data.index[test_idx], 'cloud_bullish'] = True

        # Chikou span is above price from 26 periods ago
        price_26_periods_ago = 1785
        base_data.loc[base_data.index[test_idx - 26], 'close'] = price_26_periods_ago

        # Add necessary ATR value
        base_data.loc[base_data.index[test_idx], 'atr'] = 10

        # Recalculate signals with our modified data
        result = self._identify_signals(base_data)

        # Verify signal recognition on our manipulated candle
        self.assertEqual(result.loc[result.index[test_idx], 'signal'], 1)  # Should detect a buy signal
        self.assertTrue(~np.isnan(result.loc[result.index[test_idx], 'stop_loss']))
        self.assertTrue(~np.isnan(result.loc[result.index[test_idx], 'take_profit']))

    def test_no_signal_generation(self):
        """Test that no signals are generated when conditions are not met."""
        # Create data with no clear pattern
        dates = [datetime.now() - timedelta(hours=i) for i in range(200, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 1, 200),  # Very tight range, no clear trend
            'high': np.random.normal(1802, 1, 200),
            'low': np.random.normal(1798, 1, 200),
            'close': np.random.normal(1800, 1, 200),
            'volume': np.random.normal(1000, 100, 200)
        }, index=dates)

        # Test with real calculation but mock the _identify_signals method
        with patch.object(self.strategy, '_identify_signals', return_value=data):
            # Calculate indicators and look for signals
            result = self.strategy.calculate_indicators(data)

            # Set no signal on last bar
            result.loc[result.index[-1], 'signal'] = 0

            # Configure analyze to use this mock data
            with patch.object(self.strategy, 'calculate_indicators', return_value=result):
                # Call analyze
                signals = self.strategy.analyze(data)

                # Verify we get no trading signals
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

    def _identify_signals(self, data):
        """Call the strategy's _identify_signals method on the data."""
        return self.strategy._identify_signals(data)


if __name__ == '__main__':
    unittest.main()