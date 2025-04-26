# tests/test_momentum_scalping_strategy.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from strategies.momentum_scalping import MomentumScalpingStrategy


class TestMomentumScalpingStrategy(unittest.TestCase):
    """Unit tests for the Momentum Scalping Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Initialize the strategy with the mock
        self.strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            ema_period=20,
            macd_fast=12,
            macd_slow=26,
            macd_signal=9,
            data_fetcher=self.mock_data_fetcher
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.symbol, "XAUUSD")
        self.assertEqual(self.strategy.timeframe, "M5")
        self.assertEqual(self.strategy.ema_period, 20)
        self.assertEqual(self.strategy.macd_fast, 12)
        self.assertEqual(self.strategy.macd_slow, 26)
        self.assertEqual(self.strategy.macd_signal, 9)
        self.assertEqual(self.strategy.name, "Momentum_Scalping")
        self.assertEqual(self.strategy.min_required_candles, 46)  # max(ema, macd_slow+signal) + 20

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises error."""
        with self.assertRaises(ValueError):
            # macd_fast >= macd_slow should raise ValueError
            MomentumScalpingStrategy(macd_fast=26, macd_slow=26)

    def test_calculate_indicators(self):
        """Test indicator calculation."""
        # Create sample data (100 candles)
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]
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
        self.assertIn('ema', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)
        self.assertIn('atr', result.columns)
        self.assertIn('swing_high', result.columns)
        self.assertIn('swing_low', result.columns)
        self.assertIn('signal', result.columns)
        self.assertIn('prior_trend', result.columns)
        self.assertIn('signal_strength', result.columns)
        self.assertIn('stop_loss', result.columns)
        self.assertIn('take_profit', result.columns)

    def create_bullish_momentum_data(self):
        """Create simulated data with a bullish momentum setup."""
        # Create 100 candles
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Create a downtrend for candles 50-80
        ema_value = 1820
        for i in range(50, 80):
            # Price below EMA in a downtrend
            data.loc[data.index[i], 'open'] = ema_value - np.random.uniform(5, 10)
            data.loc[data.index[i], 'close'] = ema_value - np.random.uniform(5, 10)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] + np.random.uniform(0, 3)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 3)

            # EMA sloping down
            ema_value -= 0.5

        # Create a bullish momentum shift on candles 81-85
        for i in range(80, 85):
            # Rising prices crossing above EMA
            cross_progress = (i - 80) / 4  # 0 to 1 as we progress through bars

            # Candle 80: still below, 81: crosses, 82-84: above
            if i == 80:
                data.loc[data.index[i], 'close'] = ema_value - 1  # Still below
            else:
                data.loc[data.index[i], 'close'] = ema_value + (cross_progress * 8)  # Crosses and continues above

            data.loc[data.index[i], 'open'] = data.loc[data.index[i], 'close'] - np.random.uniform(1, 4)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] + np.random.uniform(0, 2)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'open'] - np.random.uniform(0, 2)

        return data

    def test_bullish_signal_generation(self):
        """Test the generation of a bullish momentum signal."""
        # Create data with a bullish momentum setup
        data = self.create_bullish_momentum_data()

        # Process the data with real indicator calculation to set up signals properly
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare mock data with a buy signal
            result_data = self.strategy.calculate_indicators(data)

            # Make sure there is a buy signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = 1  # Buy signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.8
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = result_data['close'].iloc[-1] - 5
            result_data.loc[result_data.index[last_idx], 'take_profit'] = result_data['close'].iloc[-1] + 5
            result_data.loc[result_data.index[last_idx], 'prior_trend'] = -1
            result_data.loc[result_data.index[last_idx], 'ema'] = result_data['close'].iloc[-1] - 2
            result_data.loc[result_data.index[last_idx], 'macd_histogram'] = 0.01

            # Configure the mock to return this data
            mock_calc.return_value = result_data

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
            self.assertIn('take_profit_1r', metadata)
            self.assertIn('take_profit_2r', metadata)
            self.assertIn('risk_amount', metadata)
            self.assertIn('ema', metadata)
            self.assertIn('macd_histogram', metadata)
            self.assertEqual(metadata['reason'], 'Bullish momentum with EMA and MACD confirmation')

    def test_bearish_signal_generation(self):
        """Test the generation of a bearish momentum signal."""
        # Create data with a bearish momentum setup
        data = self.create_bearish_momentum_data()

        # Process the data with real indicator calculation to set up signals properly
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Prepare mock data with a sell signal
            result_data = self.strategy.calculate_indicators(data)

            # Make sure there is a sell signal on the last candle
            last_idx = len(result_data) - 1
            result_data.loc[result_data.index[last_idx], 'signal'] = -1  # Sell signal
            result_data.loc[result_data.index[last_idx], 'signal_strength'] = 0.7
            result_data.loc[result_data.index[last_idx], 'stop_loss'] = result_data['close'].iloc[-1] + 5
            result_data.loc[result_data.index[last_idx], 'take_profit'] = result_data['close'].iloc[-1] - 5
            result_data.loc[result_data.index[last_idx], 'prior_trend'] = 1
            result_data.loc[result_data.index[last_idx], 'ema'] = result_data['close'].iloc[-1] + 2
            result_data.loc[result_data.index[last_idx], 'macd_histogram'] = -0.01

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
            metadata = json.loads(signals[0].metadata)
            self.assertIn('stop_loss', metadata)
            self.assertIn('take_profit_1r', metadata)
            self.assertIn('take_profit_2r', metadata)
            self.assertIn('risk_amount', metadata)
            self.assertIn('ema', metadata)
            self.assertIn('macd_histogram', metadata)
            self.assertEqual(metadata['reason'], 'Bearish momentum with EMA and MACD confirmation')

    def test_signal_pattern_recognition(self):
        """Test that the strategy correctly identifies momentum patterns."""
        # Create a dataset with a clear pattern
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators first to create base data
        base_data = self.strategy.calculate_indicators(data)

        # Manually create a clear buy signal setup
        # 1. Price below EMA for several bars
        # 2. MACD histogram negative
        # 3. Then price crosses above EMA and MACD histogram crosses positive
        for i in range(90, 95):
            # Prior downtrend setup
            base_data.loc[base_data.index[i], 'ema'] = 1810
            base_data.loc[base_data.index[i], 'close'] = 1805
            base_data.loc[base_data.index[i], 'macd_histogram'] = -0.5
            base_data.loc[base_data.index[i], 'prior_trend'] = -1

        # Crossover bar
        i = 95
        base_data.loc[base_data.index[i], 'ema'] = 1810
        base_data.loc[base_data.index[i], 'close'] = 1815
        base_data.loc[base_data.index[i], 'macd_histogram'] = 0.1
        base_data.loc[base_data.index[i], 'prior_trend'] = -1

        # Setup necessary price action and ATR for correct signal calculation
        base_data.loc[base_data.index[i - 1], 'close'] = 1805
        base_data.loc[base_data.index[i - 1], 'ema'] = 1810
        base_data.loc[base_data.index[i - 1], 'macd_histogram'] = -0.05
        base_data.loc[base_data.index[90:95], 'low'] = 1800
        base_data.loc[base_data.index[i], 'atr'] = 10

        # Recalculate signals with our modified data
        with patch.object(self.strategy, 'calculate_indicators', return_value=base_data):
            # Call the indicator logic directly to check if it detects our pattern
            test_data = self.strategy._identify_signals(base_data)

            # Verify signal recognition
            self.assertEqual(test_data.loc[test_data.index[95], 'signal'], 1)  # Should detect a buy signal
            self.assertTrue(~np.isnan(test_data.loc[test_data.index[95], 'stop_loss']))
            self.assertTrue(~np.isnan(test_data.loc[test_data.index[95], 'take_profit']))

    def test_no_signal_generation(self):
        """Test that no signals are generated when conditions are not met."""
        # Create data with no clear momentum setup
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]
        data = pd.DataFrame({
            'open': np.random.normal(1800, 1, 100),  # Very tight range, no momentum
            'high': np.random.normal(1802, 1, 100),
            'low': np.random.normal(1798, 1, 100),
            'close': np.random.normal(1800, 1, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Calculate indicators and look for signals
        result = self.strategy.calculate_indicators(data)

        # Verify minimal signal detection in choppy data
        signal_count = len(result[result['signal'] != 0])
        self.assertLess(signal_count, 5)  # There should be very few if any signals

        # Test with analyze method
        with patch.object(self.strategy, 'calculate_indicators') as mock_calc:
            # Configure mock to return data with no signal
            result.loc[result.index[-1], 'signal'] = 0  # Ensure no signal on last bar
            mock_calc.return_value = result

            # Call analyze
            signals = self.strategy.analyze(data)

            # Verify we get no trading signals
            self.assertEqual(len(signals), 0)

    def create_bearish_momentum_data(self):
        """Create simulated data with a bearish momentum setup."""
        # Create 100 candles
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Create an uptrend for candles 50-80
        ema_value = 1780
        for i in range(50, 80):
            # Price above EMA in an uptrend
            data.loc[data.index[i], 'open'] = ema_value + np.random.uniform(5, 10)
            data.loc[data.index[i], 'close'] = ema_value + np.random.uniform(5, 10)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] + np.random.uniform(0, 3)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 3)

            # EMA sloping up
            ema_value += 0.5

        # Create a bearish momentum shift on candles 81-85
        for i in range(80, 85):
            # Falling prices crossing below EMA
            cross_progress = (i - 80) / 4  # 0 to 1 as we progress through bars

            # Candle 80: still above, 81: crosses, 82-84: below
            if i == 80:
                data.loc[data.index[i], 'close'] = ema_value + 1  # Still above
            else:
                data.loc[data.index[i], 'close'] = ema_value - (cross_progress * 8)  # Crosses and continues below

            data.loc[data.index[i], 'open'] = data.loc[data.index[i], 'close'] + np.random.uniform(1, 4)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'open'] + np.random.uniform(0, 2)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 2)

        return data

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Create small dataset with too few candles
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(10, 0, -1)]
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


    def create_bearish_momentum_data(self):
        """Create simulated data with a bearish momentum setup."""
        # Create 100 candles
        dates = [datetime.now() - timedelta(minutes=5 * i) for i in range(100, 0, -1)]

        # Initialize with random prices
        data = pd.DataFrame({
            'open': np.random.normal(1800, 10, 100),
            'high': np.random.normal(1810, 10, 100),
            'low': np.random.normal(1790, 10, 100),
            'close': np.random.normal(1800, 10, 100),
            'volume': np.random.normal(1000, 100, 100)
        }, index=dates)

        # Create an uptrend for candles 50-80
        ema_value = 1780
        for i in range(50, 80):
            # Price above EMA in an uptrend
            data.loc[data.index[i], 'open'] = ema_value + np.random.uniform(5, 10)
            data.loc[data.index[i], 'close'] = ema_value + np.random.uniform(5, 10)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'close'] + np.random.uniform(0, 3)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 3)

            # EMA sloping up
            ema_value += 0.5

        # Create a bearish momentum shift on candles 81-85
        for i in range(80, 85):
            # Falling prices crossing below EMA
            cross_progress = (i - 80) / 4  # 0 to 1 as we progress through bars

            # Candle 80: still above, 81: crosses, 82-84: below
            if i == 80:
                data.loc[data.index[i], 'close'] = ema_value + 1  # Still above
            else:
                data.loc[data.index[i], 'close'] = ema_value - (cross_progress * 8)  # Crosses and continues below

            data.loc[data.index[i], 'open'] = data.loc[data.index[i], 'close'] + np.random.uniform(1, 4)
            data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'open'] + np.random.uniform(0, 2)
            data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'close'] - np.random.uniform(0, 2)

        return data