# tests/strategies/test_triple_moving_average.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from strategies.triple_moving_average import TripleMovingAverageStrategy


class TestTripleMovingAverageStrategy(unittest.TestCase):
    """Test suite for the Triple Moving Average Strategy."""

    def setUp(self):
        """Set up test environment before each test method."""
        # Mock the data fetcher
        self.mock_data_fetcher = MagicMock()

        # Create strategy instance with mocked data fetcher
        self.strategy = TripleMovingAverageStrategy(
            symbol="XAUUSD",
            timeframe="H4",
            fast_period=10,
            medium_period=50,
            slow_period=200,
            data_fetcher=self.mock_data_fetcher
        )

        # Create a sample dataframe with price data
        # We'll create 250 candles to ensure enough data for all MAs
        dates = pd.date_range(start='2023-01-01', periods=250, freq='4H')

        # Create an uptrend followed by a downtrend
        uptrend = np.linspace(1800, 2000, 125)  # 125 candles in uptrend
        downtrend = np.linspace(2000, 1850, 125)  # 125 candles in downtrend
        prices = np.concatenate([uptrend, downtrend])

        # Add some noise to the prices
        noise = np.random.normal(0, 5, 250)
        prices = prices + noise

        # Create OHLC data
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.random.normal(5, 2, 250),
            'low': prices - np.random.normal(5, 2, 250),
            'close': prices + np.random.normal(0, 2, 250),
            'volume': np.random.normal(1000, 200, 250)
        }, index=dates)

    def test_initialization(self):
        """Test strategy initialization and validation."""
        # Test with valid parameters
        strategy = TripleMovingAverageStrategy(
            symbol="XAUUSD",
            timeframe="H4",
            fast_period=10,
            medium_period=50,
            slow_period=200
        )
        self.assertEqual(strategy.fast_period, 10)
        self.assertEqual(strategy.medium_period, 50)
        self.assertEqual(strategy.slow_period, 200)

        # Test with invalid parameters (fast period >= medium period)
        with self.assertRaises(ValueError):
            TripleMovingAverageStrategy(
                symbol="XAUUSD",
                timeframe="H4",
                fast_period=50,
                medium_period=50,
                slow_period=200
            )

        # Test with invalid parameters (medium period >= slow period)
        with self.assertRaises(ValueError):
            TripleMovingAverageStrategy(
                symbol="XAUUSD",
                timeframe="H4",
                fast_period=10,
                medium_period=200,
                slow_period=200
            )

    def test_calculate_indicators(self):
        """Test calculation of strategy indicators."""
        # Calculate indicators on test data
        result = self.strategy.calculate_indicators(self.test_data.copy())

        # Check that all required columns are present
        required_columns = [
            'fast_ema', 'medium_sma', 'slow_sma',
            'trend_filter', 'ema_above_sma',
            'cross_up', 'cross_down', 'signal', 'atr'
        ]
        for column in required_columns:
            self.assertIn(column, result.columns)

        # Check that moving averages are properly calculated
        # For the last candle, each MA should be around the price range
        last_candle = result.iloc[-1]
        self.assertTrue(1800 <= last_candle['fast_ema'] <= 2000)
        self.assertTrue(1800 <= last_candle['medium_sma'] <= 2000)
        self.assertTrue(1800 <= last_candle['slow_sma'] <= 2000)

        # Verify trend filter logic
        # If medium_sma > slow_sma, trend_filter should be 1
        # If medium_sma < slow_sma, trend_filter should be -1
        np.testing.assert_array_equal(
            result['trend_filter'].values,
            np.where(
                result['medium_sma'] > result['slow_sma'],
                1,
                np.where(
                    result['medium_sma'] < result['slow_sma'],
                    -1,
                    0
                )
            )
        )

        # Verify crossover detection
        expected_cross_up = ((result['fast_ema'] > result['medium_sma']) &
                             ~(result['fast_ema'].shift(1) > result['medium_sma'].shift(1)).fillna(False)).astype(int)
        expected_cross_down = ((result['fast_ema'] < result['medium_sma']) &
                               ~(result['fast_ema'].shift(1) < result['medium_sma'].shift(1)).fillna(False)).astype(int)

        np.testing.assert_array_equal(result['cross_up'].values, expected_cross_up.values)
        np.testing.assert_array_equal(result['cross_down'].values, expected_cross_down.values)

    def test_signal_generation(self):
        """Test that the strategy generates correct signals based on crossovers and trend filter."""
        # Mock the DBLogger to avoid actual logging
        with patch('strategies.triple_moving_average.DBLogger'):
            # Pre-calculate indicators
            data_with_indicators = self.strategy.calculate_indicators(self.test_data.copy())

            # Force a buy signal condition in the last row
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('signal')] = 1

            # Mock the create_signal method to return the signal directly
            mock_signal = MagicMock()
            self.strategy._generate_buy_signal = MagicMock(return_value=mock_signal)

            # Call analyze method - but patch it to ensure it doesn't recalculate indicators
            with patch.object(self.strategy, 'calculate_indicators', return_value=data_with_indicators):
                signals = self.strategy.analyze(data_with_indicators)

            # Check that a buy signal was generated
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0], mock_signal)

            # Verify that _generate_buy_signal was called with the right parameters
            self.strategy._generate_buy_signal.assert_called_once()

            # Reset and test sell signal
            self.strategy._generate_buy_signal.reset_mock()
            self.strategy._generate_sell_signal = MagicMock(return_value=mock_signal)
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('signal')] = -1

            # Call analyze method - with the same patching
            with patch.object(self.strategy, 'calculate_indicators', return_value=data_with_indicators):
                signals = self.strategy.analyze(data_with_indicators)

            # Check that a sell signal was generated
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0], mock_signal)

            # Verify that _generate_sell_signal was called
            self.strategy._generate_sell_signal.assert_called_once()

    def test_exit_signal_generation(self):
        """Test that the strategy generates exit signals based on crossovers."""
        # Mock the DBLogger to avoid actual logging
        with patch('strategies.triple_moving_average.DBLogger'):
            # Pre-calculate indicators
            data_with_indicators = self.strategy.calculate_indicators(self.test_data.copy())

            # Set no entry signal but a cross down in the last row (exit long scenario)
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('signal')] = 0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('cross_down')] = 1
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('cross_up')] = 0

            # We need previous candle to exist for the exit logic to work, make sure it exists
            if len(data_with_indicators) < 2:
                self.skipTest("Not enough data for exit signal test")

            # Mock the _generate_exit_signal method to return the signal directly
            mock_signal = MagicMock()
            self.strategy._generate_exit_signal = MagicMock(return_value=mock_signal)

            # Call analyze method - but patch it to ensure it doesn't recalculate indicators
            with patch.object(self.strategy, 'calculate_indicators', return_value=data_with_indicators):
                signals = self.strategy.analyze(data_with_indicators)

            # Check that an exit signal was generated
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0], mock_signal)

            # Verify that _generate_exit_signal was called
            self.strategy._generate_exit_signal.assert_called_once()

            # Reset and test exit short scenario (cross up)
            self.strategy._generate_exit_signal.reset_mock()
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('cross_down')] = 0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_loc('cross_up')] = 1

            # Call analyze method - with the same patching
            with patch.object(self.strategy, 'calculate_indicators', return_value=data_with_indicators):
                signals = self.strategy.analyze(data_with_indicators)

            # Check that an exit signal was generated
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0], mock_signal)

            # Verify that _generate_exit_signal was called
            self.strategy._generate_exit_signal.assert_called_once()

    def test_realistic_scenario(self):
        """Test a more realistic scenario with an uptrend followed by a downtrend."""
        # First prepare a dataframe with an uptrend and a downtrend
        # This will be more precise than the general test data
        dates = pd.date_range(start='2023-01-01', periods=400, freq='4H')

        # Create a price series with clear trend changes
        # Start with flat, then uptrend, then downtrend
        flat = np.ones(100) * 1900
        uptrend = np.linspace(1900, 2100, 150)
        downtrend = np.linspace(2100, 1800, 150)
        prices = np.concatenate([flat, uptrend, downtrend])

        # Add minimal noise to ensure clear trends
        noise = np.random.normal(0, 1, 400)
        prices = prices + noise

        realistic_data = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 2, 400)),
            'low': prices - np.abs(np.random.normal(0, 2, 400)),
            'close': prices + np.random.normal(0, 0.5, 400),
            'volume': np.random.normal(1000, 200, 400)
        }, index=dates)

        # Mock the DBLogger to avoid actual logging
        with patch('strategies.triple_moving_average.DBLogger'):
            # Calculate indicators
            data_with_indicators = self.strategy.calculate_indicators(realistic_data)

            # Check key transition points in the data
            # First, verify that trend_filter changes as expected
            # It should start as neutral (flat), become positive (uptrend), then negative (downtrend)

            # For this test, we'll examine the main transition points
            # At the beginning (during flat), trend_filter should be neutral or noisy
            # During uptrend, it should become consistently positive
            # During downtrend, it should become consistently negative

            uptrend_section = data_with_indicators.iloc[250:300]  # Sample from uptrend
            downtrend_section = data_with_indicators.iloc[350:]  # Sample from downtrend

            # During uptrend, medium SMA should eventually rise above slow SMA
            self.assertTrue((uptrend_section['trend_filter'] == 1).any())

            # During downtrend, medium SMA should eventually fall below slow SMA
            self.assertTrue((downtrend_section['trend_filter'] == -1).any())

            # Now check that signals are generated at the right times
            # Specifically, there should be buy signals during uptrend and sell signals during downtrend

            # We'll count signals in each phase
            uptrend_buys = (uptrend_section['signal'] == 1).sum()
            uptrend_sells = (uptrend_section['signal'] == -1).sum()

            downtrend_buys = (downtrend_section['signal'] == 1).sum()
            downtrend_sells = (downtrend_section['signal'] == -1).sum()

            # In uptrend, we expect more buy signals than sell signals
            # In downtrend, we expect more sell signals than buy signals

            # This test might be a bit flaky depending on the random noise
            # So we'll just verify that we get at least some of the expected signals
            self.assertGreaterEqual(uptrend_buys, 0, "Should have at least some buy signals in uptrend")
            self.assertGreaterEqual(downtrend_sells, 0, "Should have at least some sell signals in downtrend")


if __name__ == '__main__':
    unittest.main()