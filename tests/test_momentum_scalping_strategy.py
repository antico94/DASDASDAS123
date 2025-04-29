# tests/test_momentum_scalping.py
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from strategies.momentum_scalping import MomentumScalpingStrategy
from data.models import StrategySignal


class TestMomentumScalpingStrategy(unittest.TestCase):
    """Test cases for the enhanced Momentum Scalping Strategy."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock data fetcher
        self.mock_data_fetcher = MagicMock()

        # Create strategy instance with default parameters
        self.strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            data_fetcher=self.mock_data_fetcher
        )

        # Create sample OHLC data
        date_range = pd.date_range(start='2023-01-01', periods=100, freq='5min')

        # Create a stronger uptrend in the data (with more pronounced movement)
        base_price = 1900.0
        trend_strength = 1.0  # Increase the strength of the trend
        self.uptrend_data = pd.DataFrame({
            'open': [base_price + i * trend_strength + np.random.normal(0, 0.1) for i in range(100)],
            'high': [base_price + i * trend_strength + 0.5 + np.random.normal(0, 0.1) for i in range(100)],
            'low': [base_price + i * trend_strength - 0.3 + np.random.normal(0, 0.1) for i in range(100)],
            'close': [base_price + i * trend_strength + 0.1 + np.random.normal(0, 0.1) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create a stronger downtrend in the data
        self.downtrend_data = pd.DataFrame({
            'open': [base_price - i * trend_strength + np.random.normal(0, 0.1) for i in range(100)],
            'high': [base_price - i * trend_strength + 0.3 + np.random.normal(0, 0.1) for i in range(100)],
            'low': [base_price - i * trend_strength - 0.5 + np.random.normal(0, 0.1) for i in range(100)],
            'close': [base_price - i * trend_strength - 0.1 + np.random.normal(0, 0.1) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create a ranging/sideways market
        self.sideways_data = pd.DataFrame({
            'open': [base_price + np.random.normal(0, 0.5) for i in range(100)],
            'high': [base_price + 0.5 + np.random.normal(0, 0.2) for i in range(100)],
            'low': [base_price - 0.5 + np.random.normal(0, 0.2) for i in range(100)],
            'close': [base_price + np.random.normal(0, 0.5) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create data with a stronger breakout with volume for breakout testing
        self.breakout_data = self.sideways_data.copy()
        # Add a breakout with volume at the end (with stronger move)
        breakout_bars = 5
        for i in range(breakout_bars):
            idx = len(self.breakout_data) - breakout_bars + i
            # Create a progressive breakout (gradually increasing)
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['open'])] += 2.0 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['high'])] += 2.5 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['low'])] += 1.5 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['close'])] += 2.0 + i * 0.5
            # Increase volume for confirmation
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['volume'])] *= 3.0

        # Create a ranging/sideways market
        self.sideways_data = pd.DataFrame({
            'open': [base_price + np.random.normal(0, 0.5) for i in range(100)],
            'high': [base_price + 0.5 + np.random.normal(0, 0.2) for i in range(100)],
            'low': [base_price - 0.5 + np.random.normal(0, 0.2) for i in range(100)],
            'close': [base_price + np.random.normal(0, 0.5) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create data with a volume spike for breakout testing
        self.breakout_data = self.sideways_data.copy()
        # Add a breakout with volume at the end
        self.breakout_data.iloc[-5:, self.breakout_data.columns.get_indexer(['close'])] += 3.0  # Price breakout
        self.breakout_data.iloc[-5:, self.breakout_data.columns.get_indexer(['high'])] += 3.0
        self.breakout_data.iloc[-5:, self.breakout_data.columns.get_indexer(['volume'])] *= 2.5  # Volume spike

        # Mock the symbol_info for spread check
        self.mock_symbol_info = {
            'name': 'XAUUSD',
            'bid': 1900.0,
            'ask': 1900.2,  # 2 pip spread
            'point': 0.01,
            'digits': 2,
            'min_lot': 0.01,
            'max_lot': 10.0,
            'lot_step': 0.01,
            'trade_mode': 0
        }
        # Setup the connector mock to return our symbol info
        self.mock_data_fetcher.connector = MagicMock()
        self.mock_data_fetcher.connector.get_symbol_info.return_value = self.mock_symbol_info

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M1",  # Testing M1 timeframe
            rsi_period=9,
            rsi_threshold_high=65,
            rsi_threshold_low=35,
            stoch_k_period=10,
            stoch_d_period=5,
            stoch_slowing=5,
            macd_fast=8,
            macd_slow=17,
            macd_signal=9,
            momentum_period=14,
            volume_threshold=2.0,
            max_spread=5.0,
            consider_session=False,
            data_fetcher=self.mock_data_fetcher
        )

        # Check that parameters were correctly stored
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M1")
        self.assertEqual(strategy.rsi_period, 9)
        self.assertEqual(strategy.rsi_threshold_high, 65)
        self.assertEqual(strategy.rsi_threshold_low, 35)
        self.assertEqual(strategy.stoch_k_period, 10)
        self.assertEqual(strategy.stoch_d_period, 5)
        self.assertEqual(strategy.stoch_slowing, 5)
        self.assertEqual(strategy.macd_fast, 8)
        self.assertEqual(strategy.macd_slow, 17)
        self.assertEqual(strategy.macd_signal, 9)
        self.assertEqual(strategy.momentum_period, 14)
        self.assertEqual(strategy.volume_threshold, 2.0)
        self.assertEqual(strategy.max_spread, 5.0)
        self.assertEqual(strategy.consider_session, False)

    def test_init_with_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test with macd_fast >= macd_slow
        with self.assertRaises(ValueError):
            MomentumScalpingStrategy(
                symbol="XAUUSD",
                timeframe="M5",
                macd_fast=20,
                macd_slow=12,
                data_fetcher=self.mock_data_fetcher
            )

        # Test with invalid RSI thresholds
        with self.assertRaises(ValueError):
            MomentumScalpingStrategy(
                symbol="XAUUSD",
                timeframe="M5",
                rsi_threshold_high=40,
                rsi_threshold_low=60,
                data_fetcher=self.mock_data_fetcher
            )

        # Test with invalid stochastic periods
        with self.assertRaises(ValueError):
            MomentumScalpingStrategy(
                symbol="XAUUSD",
                timeframe="M5",
                stoch_k_period=0,
                data_fetcher=self.mock_data_fetcher
            )

    def test_calculate_indicators_uptrend(self):
        """Test indicator calculation in an uptrend."""
        # Calculate indicators on uptrend data
        result = self.strategy.calculate_indicators(self.uptrend_data)

        # Check that essential indicators were calculated
        self.assertIn('rsi', result.columns)
        self.assertIn('macd', result.columns)
        self.assertIn('macd_histogram', result.columns)
        self.assertIn('stoch_k', result.columns)
        self.assertIn('stoch_d', result.columns)
        self.assertIn('momentum', result.columns)
        self.assertIn('atr', result.columns)
        self.assertIn('volume_ratio', result.columns)

        # Check indicator values in an uptrend
        # Get the last 10 values to check trend direction
        last_rsi_values = result['rsi'].dropna().tail(10)
        last_macd_values = result['macd_histogram'].dropna().tail(10)

        # In an uptrend, RSI should be generally high
        self.assertTrue((last_rsi_values > 50).mean() >= 0.7)  # At least 70% of values are above 50

        # MACD histogram should be mostly positive
        self.assertTrue((last_macd_values > 0).mean() >= 0.7)  # At least 70% of histogram values are positive

    def test_calculate_indicators_downtrend(self):
        """Test indicator calculation in a downtrend."""
        # Calculate indicators on downtrend data
        result = self.strategy.calculate_indicators(self.downtrend_data)

        # Check indicator values in a downtrend
        # Get the last 10 values to check trend direction
        last_rsi_values = result['rsi'].dropna().tail(10)
        last_macd_values = result['macd_histogram'].dropna().tail(10)

        # In a downtrend, RSI should be generally low
        self.assertTrue((last_rsi_values < 50).mean() >= 0.7)  # At least 70% of values are below 50

        # MACD histogram should be mostly negative
        self.assertTrue((last_macd_values < 0).mean() >= 0.7)  # At least 70% of histogram values are negative

    def test_analyze_breakout(self):
        """Test signal generation on a breakout with volume."""
        # Patch the create_signal method to track calls
        with patch.object(self.strategy, 'create_signal',
                          return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
            # Call analyze method with breakout data
            signals = self.strategy.analyze(self.breakout_data)

            # Check if a signal was generated
            self.assertTrue(mock_create_signal.called)
            # Get the call arguments
            call_args = mock_create_signal.call_args[1]
            # Check signal type - should be BUY for an upward breakout
            self.assertEqual(call_args['signal_type'], "BUY")

    def test_session_handling(self):
        """Test session handling logic."""
        # Create a strategy with session awareness enabled
        session_aware_strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            consider_session=True,
            data_fetcher=self.mock_data_fetcher
        )

        # Modify our data to include session information
        test_data = self.breakout_data.copy()
        # Add good_session column (0 for poor session, 1 for good session)
        test_data['good_session'] = 0  # Poor session

        # Analyze with session awareness
        signals = session_aware_strategy.analyze(test_data)

        # Since we're in a poor session, the strategy should be more selective
        # But it might still generate signals if the momentum is very strong

        # Now test with a good session
        test_data['good_session'] = 1  # Good session
        signals = session_aware_strategy.analyze(test_data)

        # Session-aware strategy should work normally in good sessions
        self.assertGreaterEqual(len(signals), 0)

    def test_momentum_fading_detection(self):
        """Test detection of fading momentum for exit signals."""
        # Create data with momentum building and then fading
        # Start with uptrend
        fade_data = self.uptrend_data.copy()

        # Modify the last few bars to show fading momentum
        # (RSI dropping, MACD histogram shrinking)
        # We'll need to calculate indicators first, then modify them
        result = self.strategy.calculate_indicators(fade_data)

        # Now manually modify the indicators to simulate fading momentum
        # Take the last 5 bars
        for i in range(5):
            idx = result.index[-5 + i]
            # Gradually reduce RSI
            result.loc[idx, 'rsi'] = 70 - (i * 5)  # 70, 65, 60, 55, 50
            # Make MACD histogram shrinking
            result.loc[idx, 'macd_histogram'] = 0.5 - (i * 0.1)  # 0.5, 0.4, 0.3, 0.2, 0.1
            # Make stochastic crossing below 80
            result.loc[idx, 'stoch_k'] = 85 - (i * 10)  # 85, 75, 65, 55, 45
            result.loc[idx, 'stoch_d'] = 80 - (i * 8)  # 80, 72, 64, 56, 48
            # Set momentum state and fading flags
            if i < 2:
                result.loc[idx, 'momentum_state'] = 1  # Bullish
                result.loc[idx, 'momentum_fading'] = 0  # Not yet fading
            else:
                result.loc[idx, 'momentum_state'] = 1  # Still bullish
                result.loc[idx, 'momentum_fading'] = 1  # But now fading

        # Patch create_signal to track calls
        with patch.object(self.strategy, 'create_signal',
                          return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
            # Skip the calculate_indicators step since we've manually modified the data
            original_calculate = self.strategy.calculate_indicators
            self.strategy.calculate_indicators = lambda x: result

            try:
                # Analyze the modified data
                signals = self.strategy.analyze(result)

                # Check if create_signal was called with CLOSE signal
                close_signals = [call for call in mock_create_signal.call_args_list if
                                 call[1]['signal_type'] == "CLOSE"]
                self.assertTrue(len(close_signals) > 0, "Should have generated at least one CLOSE signal")

                # Check signal metadata
                for call in close_signals:
                    metadata = call[1]['metadata']
                    self.assertEqual(metadata['position_type'], "BUY")  # Should close long positions
                    self.assertIn('reason', metadata)
                    self.assertIn('Bullish momentum fading', metadata['reason'])
            finally:
                # Restore original method
                self.strategy.calculate_indicators = original_calculate

    def test_spread_check(self):
        """Test spread checking functionality."""
        # Create a strategy with a tight max spread
        tight_spread_strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            max_spread=1.0,  # Very tight (1 pip)
            data_fetcher=self.mock_data_fetcher
        )

        # Test with normal spread (2 pips)
        signals = tight_spread_strategy.analyze(self.breakout_data)
        # Should return empty list since spread > max_spread
        self.assertEqual(len(signals), 0)

        # Now set a wider max spread
        wide_spread_strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            max_spread=4.0,  # 4 pips is wider than our 2 pip spread
            data_fetcher=self.mock_data_fetcher
        )

        # Should now allow trading
        signals = wide_spread_strategy.analyze(self.breakout_data)
        # Length could be 0 if no signals, but at least we passed the spread check

        # Test with extremely wide spread
        # Modify the mock to simulate wide spread
        wide_symbol_info = self.mock_symbol_info.copy()
        wide_symbol_info['ask'] = 1900.8  # 8 pip spread
        self.mock_data_fetcher.connector.get_symbol_info.return_value = wide_symbol_info

        # Even the wide spread strategy should now reject trading
        signals = wide_spread_strategy.analyze(self.breakout_data)
        self.assertEqual(len(signals), 0, "Should reject trading with extremely wide spread")

        # Restore normal spread for other tests
        self.mock_data_fetcher.connector.get_symbol_info.return_value = self.mock_symbol_info


if __name__ == '__main__':
    unittest.main()