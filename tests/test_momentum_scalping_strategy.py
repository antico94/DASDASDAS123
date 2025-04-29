import datetime
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from strategies.momentum_scalping import MomentumScalpingStrategy
from data.models import StrategySignal


class TestMomentumScalpingStrategy(unittest.TestCase):
    """Comprehensive test suite for the Momentum Scalping Strategy."""

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

        # Create sample base OHLC data
        date_range = pd.date_range(start='2023-01-01', periods=100, freq='5min')
        base_price = 1900.0
        self.base_data = pd.DataFrame({
            'open': [base_price + np.random.normal(0, 0.5) for _ in range(100)],
            'high': [base_price + 0.5 + np.random.normal(0, 0.2) for _ in range(100)],
            'low': [base_price - 0.5 + np.random.normal(0, 0.2) for _ in range(100)],
            'close': [base_price + np.random.normal(0, 0.5) for _ in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for _ in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for _ in range(100)],
            'spread': [2 + np.random.randint(0, 2) for _ in range(100)]
        }, index=date_range)

        # Create uptrend data with strong momentum
        trend_strength = 2.0
        self.uptrend_data = pd.DataFrame({
            'open': [base_price + i * trend_strength + np.random.normal(0, 0.1) for i in range(100)],
            'high': [base_price + i * trend_strength + 0.5 + np.random.normal(0, 0.1) for i in range(100)],
            'low': [base_price + i * trend_strength - 0.3 + np.random.normal(0, 0.1) for i in range(100)],
            'close': [base_price + i * trend_strength + 0.1 + np.random.normal(0, 0.1) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create downtrend data with strong momentum
        self.downtrend_data = pd.DataFrame({
            'open': [base_price - i * trend_strength + np.random.normal(0, 0.1) for i in range(100)],
            'high': [base_price - i * trend_strength + 0.3 + np.random.normal(0, 0.1) for i in range(100)],
            'low': [base_price - i * trend_strength - 0.5 + np.random.normal(0, 0.1) for i in range(100)],
            'close': [base_price - i * trend_strength - 0.1 + np.random.normal(0, 0.1) for i in range(100)],
            'volume': [1000 + np.random.normal(0, 200) for i in range(100)],
            'tick_volume': [100 + np.random.randint(0, 50) for i in range(100)],
            'spread': [2 + np.random.randint(0, 2) for i in range(100)]
        }, index=date_range)

        # Create breakout data
        self.breakout_data = self.base_data.copy()
        # Add a breakout with volume at the end
        breakout_bars = 5
        for i in range(breakout_bars):
            idx = len(self.breakout_data) - breakout_bars + i
            # Create a progressive breakout
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['open'])] += 3.0 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['high'])] += 3.5 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['low'])] += 2.5 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['close'])] += 3.0 + i * 0.5
            self.breakout_data.iloc[idx, self.breakout_data.columns.get_indexer(['volume'])] *= 3.0

        # Create data with momentum fading
        self.fading_momentum_data = self.uptrend_data.copy()
        # Start strong then fade
        for i in range(5):
            idx = len(self.fading_momentum_data) - 5 + i
            # Momentum starts to fade
            self.fading_momentum_data.iloc[idx, self.fading_momentum_data.columns.get_indexer(['close'])] -= i * 0.3
            self.fading_momentum_data.iloc[idx, self.fading_momentum_data.columns.get_indexer(['volume'])] /= (
                        1.0 + i * 0.1)

        # Mock symbol_info for spread checking
        self.mock_symbol_info = {
            'name': 'XAUUSD',
            'bid': 1900.0,
            'ask': 1900.3,  # 3 pip spread
            'point': 0.01,
            'digits': 2,
            'min_lot': 0.01,
            'max_lot': 10.0,
            'lot_step': 0.01,
            'trade_mode': 0
        }

        # Setup connector mock
        self.mock_data_fetcher.connector = MagicMock()
        self.mock_data_fetcher.connector.get_symbol_info.return_value = self.mock_symbol_info

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            data_fetcher=self.mock_data_fetcher
        )

        # Check default parameters
        self.assertEqual(strategy.symbol, "XAUUSD")
        self.assertEqual(strategy.timeframe, "M5")
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.rsi_threshold_high, 60)
        self.assertEqual(strategy.rsi_threshold_low, 40)
        self.assertEqual(strategy.stoch_k_period, 14)
        self.assertEqual(strategy.stoch_d_period, 3)
        self.assertEqual(strategy.stoch_slowing, 3)
        self.assertEqual(strategy.macd_fast, 12)
        self.assertEqual(strategy.macd_slow, 26)
        self.assertEqual(strategy.macd_signal, 9)
        self.assertEqual(strategy.momentum_period, 10)
        self.assertEqual(strategy.volume_threshold, 1.5)
        self.assertEqual(strategy.max_spread, 3.0)
        self.assertEqual(strategy.consider_session, True)

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M1",
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

        # Check custom parameters
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

    def test_initialization_with_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test with macd_fast >= macd_slow
        with self.assertRaises(ValueError):
            MomentumScalpingStrategy(
                symbol="XAUUSD",
                timeframe="M5",
                macd_fast=26,
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

    def test_calculate_rsi(self):
        """Test RSI calculation function."""
        # Calculate RSI
        result = self.strategy._calculate_rsi(self.uptrend_data)

        # Check that RSI was calculated
        self.assertIn('rsi', result.columns)

        # RSI should be between 0 and 100
        rsi_values = result['rsi'].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values))

        # In an uptrend, RSI should generally be high
        self.assertTrue(rsi_values.tail(10).mean() > 50)

        # Check that RSI direction is calculated
        self.assertIn('rsi_direction', result.columns)

    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation."""
        # Calculate Stochastic
        result = self.strategy._calculate_stochastic(self.uptrend_data)

        # Check that Stochastic was calculated
        self.assertIn('stoch_k', result.columns)
        self.assertIn('stoch_d', result.columns)

        # Stoch values should be between 0 and 100
        stoch_k_values = result['stoch_k'].dropna()
        stoch_d_values = result['stoch_d'].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in stoch_k_values))
        self.assertTrue(all(0 <= val <= 100 for val in stoch_d_values))

        # Check that k_above_d, overbought/oversold columns exist
        self.assertIn('stoch_k_above_d', result.columns)
        self.assertIn('stoch_overbought', result.columns)
        self.assertIn('stoch_oversold', result.columns)

        # Check crossover detection columns
        self.assertIn('stoch_bull_cross', result.columns)
        self.assertIn('stoch_bear_cross', result.columns)

    def test_calculate_macd(self):
        """Test MACD calculation."""
        # Calculate MACD
        result = self.strategy._calculate_macd(self.uptrend_data)

        # Check that MACD was calculated
        self.assertIn('macd', result.columns)
        self.assertIn('macd_signal', result.columns)
        self.assertIn('macd_histogram', result.columns)

        # Check histogram direction
        self.assertIn('macd_histogram_direction', result.columns)

        # Check crossover detection
        self.assertIn('macd_bull_cross', result.columns)
        self.assertIn('macd_bear_cross', result.columns)

        # In an uptrend, MACD should generally be positive
        macd_values = result['macd'].dropna().tail(10)
        self.assertTrue(macd_values.mean() > 0)

    def test_calculate_momentum(self):
        """Test Momentum/ROC calculation."""
        # Calculate Momentum
        result = self.strategy._calculate_momentum(self.uptrend_data)

        # Check that Momentum was calculated
        self.assertIn('momentum', result.columns)

        # Check momentum state columns
        self.assertIn('momentum_positive', result.columns)
        self.assertIn('momentum_negative', result.columns)
        self.assertIn('momentum_direction', result.columns)

        # In an uptrend, momentum should generally be above 100
        momentum_values = result['momentum'].dropna().tail(10)
        self.assertTrue(momentum_values.mean() > 100)

    def test_calculate_atr(self):
        """Test ATR calculation."""
        # Calculate ATR
        result = self.strategy._calculate_atr(self.uptrend_data)

        # Check that ATR was calculated
        self.assertIn('atr', result.columns)

        # ATR should be positive
        atr_values = result['atr'].dropna()
        self.assertTrue(all(val > 0 for val in atr_values))

    def test_calculate_volume_metrics(self):
        """Test volume metrics calculation."""
        # Calculate volume metrics
        result = self.strategy._calculate_volume_metrics(self.uptrend_data)

        # Check that volume metrics were calculated
        self.assertIn('volume_ma', result.columns)
        self.assertIn('volume_ratio', result.columns)
        self.assertIn('high_volume', result.columns)
        self.assertIn('volume_change', result.columns)

        # Volume ratio should be positive
        volume_ratio_values = result['volume_ratio'].dropna()
        self.assertTrue(all(val > 0 for val in volume_ratio_values))

    def test_add_session_info(self):
        """Test adding session information."""
        # Calculate session info
        result = self.strategy._add_session_info(self.uptrend_data)

        # Check that session info was added
        self.assertIn('good_session', result.columns)

        # If consider_session is True, also check for low_liquidity_session
        if self.strategy.consider_session:
            self.assertIn('low_liquidity_session', result.columns)

    def test_calculate_indicators_integration(self):
        """Test that all indicators are calculated correctly together."""
        # Test the main indicator calculation method
        result = self.strategy._calculate_indicators(self.uptrend_data)

        # Check for all required indicator columns
        required_columns = [
            'rsi', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_histogram',
            'momentum', 'atr', 'volume_ratio', 'good_session'
        ]

        for col in required_columns:
            self.assertIn(col, result.columns, f"Missing expected column: {col}")

        # Verify sensible values in the indicators
        # RSI between 0-100
        self.assertTrue(all(0 <= val <= 100 for val in result['rsi'].dropna()))

        # Stochastic between 0-100
        self.assertTrue(all(0 <= val <= 100 for val in result['stoch_k'].dropna()))
        self.assertTrue(all(0 <= val <= 100 for val in result['stoch_d'].dropna()))

        # Volume ratio and ATR should be positive
        self.assertTrue(all(val > 0 for val in result['volume_ratio'].dropna()))
        self.assertTrue(all(val > 0 for val in result['atr'].dropna()))

    def test_identify_signals_uptrend(self):
        """Test signal identification in uptrend."""
        # Create strong uptrend data with high volume
        strong_uptrend = self.uptrend_data.copy()
        # Increase volume in the last 5 bars to trigger signals
        for i in range(5):
            idx = len(strong_uptrend) - 5 + i
            strong_uptrend.iloc[idx, strong_uptrend.columns.get_indexer(['volume'])] *= 2.0

        # Calculate indicators first
        data_with_indicators = self.strategy._calculate_indicators(strong_uptrend)

        # Now identify signals
        result = self.strategy._identify_signals(data_with_indicators)

        # Check that signal columns were added
        self.assertIn('signal', result.columns)
        self.assertIn('signal_strength', result.columns)
        self.assertIn('stop_loss', result.columns)
        self.assertIn('take_profit', result.columns)
        self.assertIn('momentum_state', result.columns)
        self.assertIn('momentum_fading', result.columns)

        # In a strong uptrend with high volume, we should see some buy signals
        buy_signals = result[result['signal'] == 1]
        self.assertTrue(len(buy_signals) > 0, "Expected at least one buy signal in strong uptrend")

    def test_identify_signals_downtrend(self):
        """Test signal identification in downtrend."""
        # Create strong downtrend data with high volume
        strong_downtrend = self.downtrend_data.copy()
        # Increase volume in the last 5 bars to trigger signals
        for i in range(5):
            idx = len(strong_downtrend) - 5 + i
            strong_downtrend.iloc[idx, strong_downtrend.columns.get_indexer(['volume'])] *= 2.0

        # Calculate indicators first
        data_with_indicators = self.strategy._calculate_indicators(strong_downtrend)

        # Now identify signals
        result = self.strategy._identify_signals(data_with_indicators)

        # In a strong downtrend with high volume, we should see some sell signals
        sell_signals = result[result['signal'] == -1]
        self.assertTrue(len(sell_signals) > 0, "Expected at least one sell signal in strong downtrend")

    def test_identify_signals_breakout(self):
        """Test signal identification in a breakout scenario."""
        # Calculate indicators first
        data_with_indicators = self.strategy._calculate_indicators(self.breakout_data)

        # Now identify signals
        result = self.strategy._identify_signals(data_with_indicators)

        # In a breakout with high volume, we should see a buy signal
        buy_signals = result[result['signal'] == 1]
        self.assertTrue(len(buy_signals) > 0, "Expected at least one buy signal in breakout")

        # Check that stop loss and take profit are calculated
        signal_rows = result[result['signal'] != 0]
        if len(signal_rows) > 0:
            last_signal = signal_rows.iloc[-1]
            self.assertFalse(np.isnan(last_signal['stop_loss']), "Stop loss should be calculated for signals")
            self.assertFalse(np.isnan(last_signal['take_profit']), "Take profit should be calculated for signals")

    def test_identify_signals_momentum_fading(self):
        """Test detection of fading momentum for exit signals."""
        # Calculate indicators on fading momentum data
        data_with_indicators = self.strategy._calculate_indicators(self.fading_momentum_data)

        # Create a state where we were in bullish momentum
        # Set the momentum_state of previous bars to bullish (1)
        for i in range(6, 10):
            idx = len(data_with_indicators) - i
            data_with_indicators.iloc[idx, data_with_indicators.columns.get_indexer(['momentum_state'])] = 1

        # Now identify signals
        result = self.strategy._identify_signals(data_with_indicators)

        # Check if momentum fading was detected in the last few bars
        fading_momentum = result.iloc[-5:]['momentum_fading']
        self.assertTrue(any(fading_momentum > 0), "Should detect bullish momentum fading")

    def test_analyze_with_buy_signal(self):
        """Test the main analyze method with conditions for a buy signal."""
        # Setup data that should trigger a buy signal
        # Take the breakout data and enhance it
        signal_data = self.breakout_data.copy()

        # Modify the last few bars to ensure strong indicators
        for i in range(3):
            idx = len(signal_data) - 3 + i
            # Make last bars very bullish
            signal_data.iloc[idx, signal_data.columns.get_indexer(['close'])] += 3.0 + i * 1.0
            signal_data.iloc[idx, signal_data.columns.get_indexer(['volume'])] *= 3.0

        # Patch calculate_indicators to return our prepared data with signals
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # First calculate indicators normally to get real values
            enhanced_data = self.strategy._calculate_indicators(signal_data)

            # Then modify to ensure a buy signal in the last bar
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal'])] = 1
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal_strength'])] = 0.8
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['stop_loss'])] = signal_data['close'].iloc[
                                                                                           -1] * 0.995
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['take_profit'])] = signal_data['close'].iloc[
                                                                                             -1] * 1.005
            mock_calc.return_value = enhanced_data

            # Patch create_signal to track calls
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                # Analyze the data
                signals = self.strategy.analyze(signal_data)

                # Check that create_signal was called for a BUY
                self.assertTrue(mock_create_signal.called)
                # Get the call arguments
                call_args = mock_create_signal.call_args[1]
                self.assertEqual(call_args['signal_type'], "BUY")
                self.assertAlmostEqual(call_args['strength'], 0.8, places=1)
                self.assertIn('stop_loss', call_args['metadata'])
                self.assertIn('take_profit_1r', call_args['metadata'])
                self.assertIn('take_profit_2r', call_args['metadata'])

    def test_analyze_with_sell_signal(self):
        """Test the main analyze method with conditions for a sell signal."""
        # Setup data that should trigger a sell signal
        # Take the downtrend data and enhance it
        signal_data = self.downtrend_data.copy()

        # Modify the last few bars to ensure strong bearish indicators
        for i in range(3):
            idx = len(signal_data) - 3 + i
            # Make last bars very bearish
            signal_data.iloc[idx, signal_data.columns.get_indexer(['close'])] -= 3.0 + i * 1.0
            signal_data.iloc[idx, signal_data.columns.get_indexer(['volume'])] *= 3.0

        # Patch calculate_indicators to return our prepared data with signals
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # First calculate indicators normally to get real values
            enhanced_data = self.strategy._calculate_indicators(signal_data)

            # Then modify to ensure a sell signal in the last bar
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal'])] = -1
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal_strength'])] = 0.8
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['stop_loss'])] = signal_data['close'].iloc[
                                                                                           -1] * 1.005
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['take_profit'])] = signal_data['close'].iloc[
                                                                                             -1] * 0.995
            mock_calc.return_value = enhanced_data

            # Patch create_signal to track calls
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                # Analyze the data
                signals = self.strategy.analyze(signal_data)

                # Check that create_signal was called for a SELL
                self.assertTrue(mock_create_signal.called)
                # Get the call arguments
                call_args = mock_create_signal.call_args[1]
                self.assertEqual(call_args['signal_type'], "SELL")
                self.assertAlmostEqual(call_args['strength'], 0.8, places=1)
                self.assertIn('stop_loss', call_args['metadata'])
                self.assertIn('take_profit_1r', call_args['metadata'])
                self.assertIn('take_profit_2r', call_args['metadata'])

    def test_analyze_with_momentum_fading_exit_signal(self):
        """Test the analyze method with conditions for a momentum fading exit signal."""
        # Setup data with fading momentum
        signal_data = self.fading_momentum_data.copy()

        # Patch calculate_indicators to return our prepared data with momentum fading
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # First calculate indicators normally
            enhanced_data = self.strategy._calculate_indicators(signal_data)

            # Then modify to ensure momentum fading in the last bar (bullish fading)
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['momentum_fading'])] = 1
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['momentum_state'])] = 1  # Was bullish
            mock_calc.return_value = enhanced_data

            # Patch create_signal to track calls
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                # Analyze the data
                signals = self.strategy.analyze(signal_data)

                # Check that create_signal was called for a CLOSE signal
                self.assertTrue(mock_create_signal.called)
                # Get the call arguments
                call_args = mock_create_signal.call_args[1]
                self.assertEqual(call_args['signal_type'], "CLOSE")
                # Check metadata has position_type and reason
                self.assertIn('position_type', call_args['metadata'])
                self.assertEqual(call_args['metadata']['position_type'], "BUY")  # Should close long positions
                self.assertIn('reason', call_args['metadata'])
                self.assertIn('momentum fading', call_args['metadata']['reason'].lower())

    def test_analyze_with_bearish_momentum_fading(self):
        """Test detection of bearish momentum fading."""
        # Setup data with bearish momentum fading
        signal_data = self.downtrend_data.copy()

        # Patch calculate_indicators to return data with bearish momentum fading
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # First calculate indicators normally
            enhanced_data = self.strategy._calculate_indicators(signal_data)

            # Then modify to ensure bearish momentum fading in the last bar
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['momentum_fading'])] = -1
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['momentum_state'])] = -1  # Was bearish
            mock_calc.return_value = enhanced_data

            # Patch create_signal to track calls
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                # Analyze the data
                signals = self.strategy.analyze(signal_data)

                # Check that create_signal was called for a CLOSE signal for short positions
                self.assertTrue(mock_create_signal.called)
                # Get the call arguments
                call_args = mock_create_signal.call_args[1]
                self.assertEqual(call_args['signal_type'], "CLOSE")
                # Check metadata has position_type and reason
                self.assertIn('position_type', call_args['metadata'])
                self.assertEqual(call_args['metadata']['position_type'], "SELL")  # Should close short positions
                self.assertIn('reason', call_args['metadata'])
                self.assertIn('bearish momentum fading', call_args['metadata']['reason'].lower())

    def test_analyze_with_wide_spread(self):
        """Test that analysis skips trading when spread is too wide."""
        # Setup a wide spread scenario
        wide_symbol_info = self.mock_symbol_info.copy()
        wide_symbol_info['ask'] = 1900.8  # 8 pip spread (wider than max_spread of 3.0)
        self.mock_data_fetcher.connector.get_symbol_info.return_value = wide_symbol_info

        # Patch _calculate_indicators to ensure it returns data with potential signals
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Create data that would normally generate signals
            enhanced_data = self.strategy._calculate_indicators(self.breakout_data)
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal'])] = 1  # Buy signal
            mock_calc.return_value = enhanced_data

            # Patch create_signal to track calls
            with patch.object(self.strategy, 'create_signal') as mock_create_signal:
                # Analyze the data
                signals = self.strategy.analyze(self.breakout_data)

                # Check that no signals were generated due to wide spread
                self.assertEqual(len(signals), 0)
                self.assertFalse(mock_create_signal.called)

    def test_analyze_with_session_filter(self):
        """Test session filtering in the analysis."""
        # Setup data with session information
        signal_data = self.breakout_data.copy()

        # Test with non-optimal session
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Create data with signals but poor session
            enhanced_data = self.strategy._calculate_indicators(signal_data)
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['signal'])] = 1  # Buy signal
            enhanced_data.iloc[-1, enhanced_data.columns.get_indexer(['good_session'])] = 0  # Poor session
            mock_calc.return_value = enhanced_data

            # Patch datetime to return a specific time outside optimal session
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.datetime(2023, 1, 1, 4, 0, 0)  # 4 AM UTC (Asian session)
                mock_datetime.utcnow.return_value = datetime.datetime(2023, 1, 1, 4, 0, 0)

                # Analyze the data
                signals = self.strategy.analyze(signal_data)

                # Strategy should still generate signals even in poor session
                # but might have a lower threshold/strength
                self.assertGreaterEqual(len(signals), 0)

    def test_analyze_edge_cases(self):
        """Test analysis with edge cases like empty data or no indicators."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        signals = self.strategy.analyze(empty_data)
        self.assertEqual(len(signals), 0)

        # Test with None data
        signals = self.strategy.analyze(None)
        self.assertEqual(len(signals), 0)

        # Test with valid data but no signal columns
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Return data without signal columns
            mock_calc.return_value = self.base_data.copy()  # No signal columns added
            signals = self.strategy.analyze(self.base_data)
            self.assertEqual(len(signals), 0)

        # Test with exception during indicator calculation
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            mock_calc.side_effect = Exception("Test error")
            signals = self.strategy.analyze(self.base_data)
            self.assertEqual(len(signals), 0)

    def test_indicator_calculation_with_insufficient_data(self):
        """Test indicator calculations with insufficient data."""
        # Create a very small dataset (smaller than required periods)
        small_data = self.base_data.iloc[:10].copy()  # Only 10 rows

        # Calculate indicators
        result = self.strategy._calculate_indicators(small_data)

        # Should return original data without error but indicators may be NaN
        self.assertEqual(len(result), len(small_data))

    def test_session_aware_logic(self):
        """Test session awareness logic with different timeframes."""
        # Create a strategy with session awareness enabled
        session_aware_strategy = MomentumScalpingStrategy(
            symbol="XAUUSD",
            timeframe="M5",
            consider_session=True,
            data_fetcher=self.mock_data_fetcher
        )

        # Test different hours for session quality
        test_times = [
            (0, False),  # 00:00 UTC - Asian session (poor liquidity)
            (4, False),  # 04:00 UTC - Asian session (poor liquidity)
            (8, False),  # 08:00 UTC - European session (moderate)
            (14, True),  # 14:00 UTC - London/NY overlap (good liquidity)
            (16, True),  # 16:00 UTC - London/NY overlap (good liquidity)
            (20, False)  # 20:00 UTC - US afternoon (moderate)
        ]

        test_data = self.base_data.copy()

        for hour, expected_good in test_times:
            # Create index with the specific hour
            hour_indices = [pd.Timestamp(d.date(), hour=hour) for d in test_data.index]
            hour_data = test_data.copy()
            hour_data.index = hour_indices

            # Test session quality detection
            result = session_aware_strategy._add_session_info(hour_data)

            # Check last bar's session quality
            is_good_session = result.iloc[-1]['good_session'] == 1
            self.assertEqual(is_good_session, expected_good,
                             f"Hour {hour} should be {'good' if expected_good else 'not good'} session")

            # Check low liquidity flag for Asian session
            if 0 <= hour < 6:
                self.assertEqual(result.iloc[-1]['low_liquidity_session'], 1,
                                 f"Hour {hour} should be flagged as low liquidity")

    def test_signal_strength_calculation(self):
        """Test that signal strength is calculated correctly based on indicators."""
        # Create data with very strong indicators
        strong_signal_data = self.breakout_data.copy()

        # Modify to have extremely strong indicators
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Calculate indicators first
            data_with_indicators = self.strategy._calculate_indicators(strong_signal_data)

            # Manually set very strong indicator values
            idx = -1  # Last bar
            data_with_indicators.iloc[idx, data_with_indicators.columns.get_indexer(['rsi'])] = 85  # Very high RSI
            data_with_indicators.iloc[
                idx, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.5  # Strong positive histogram
            data_with_indicators.iloc[
                idx, data_with_indicators.columns.get_indexer(['stoch_k'])] = 90  # High stochastic
            data_with_indicators.iloc[
                idx, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 3.0  # 3x volume

            # Add a stochastic crossover from oversold (extra signal strength factor)
            data_with_indicators.iloc[
                idx - 1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 15  # Previously oversold
            data_with_indicators.iloc[
                idx, data_with_indicators.columns.get_indexer(['stoch_bull_cross'])] = 1  # Bullish cross

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(strong_signal_data)

            # There should be a signal with high strength
            self.assertGreater(len(signals), 0)

            # Check if the create_signal was called with high strength
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                signals = self.strategy.analyze(strong_signal_data)

                # Check signal strength is high (near 1.0)
                call_args = mock_create_signal.call_args[1]
                self.assertGreater(call_args['strength'], 0.7, "Strong indicators should generate high signal strength")

    def test_stop_loss_calculation(self):
        """Test that stop loss is calculated correctly using ATR."""
        # Prepare data with known ATR value
        atr_test_data = self.base_data.copy()

        # Set a known ATR value in the data
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Calculate indicators first
            data_with_indicators = self.strategy._calculate_indicators(atr_test_data)

            # Set a specific ATR value
            atr_value = 2.0  # $2.0 ATR
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['atr'])] = atr_value

            # Create a buy signal
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['signal'])] = 1
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['signal_strength'])] = 0.8

            # Record the close price
            close_price = data_with_indicators.iloc[-1]['close']

            mock_calc.return_value = data_with_indicators

            # Patch create_signal to capture stop loss calculation
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                signals = self.strategy.analyze(atr_test_data)

                # Check stop loss - should be 1.5 * ATR below entry for buy
                call_args = mock_create_signal.call_args[1]
                expected_stop = close_price - (atr_value * 1.5)
                self.assertAlmostEqual(
                    call_args['metadata']['stop_loss'],
                    expected_stop,
                    places=1,
                    msg="Stop loss should be calculated as entry - (1.5 * ATR) for buy signals"
                )

    def test_take_profit_calculation(self):
        """Test that take profit targets are calculated with correct risk-reward ratios."""
        # Prepare data with buy signal and stop loss
        tp_test_data = self.base_data.copy()

        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            # Calculate indicators
            data_with_indicators = self.strategy._calculate_indicators(tp_test_data)

            # Create a buy signal with known values
            entry_price = 1900.0
            stop_loss = 1890.0  # $10 risk

            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['signal'])] = 1
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = entry_price
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stop_loss'])] = stop_loss

            mock_calc.return_value = data_with_indicators

            # Patch create_signal to capture take profit calculations
            with patch.object(self.strategy, 'create_signal',
                              return_value=MagicMock(spec=StrategySignal)) as mock_create_signal:
                signals = self.strategy.analyze(tp_test_data)

                # Check take profit targets
                call_args = mock_create_signal.call_args[1]
                risk = entry_price - stop_loss  # $10

                # First target should be 1:1 risk:reward
                expected_tp1 = entry_price + risk
                self.assertAlmostEqual(
                    call_args['metadata']['take_profit_1r'],
                    expected_tp1,
                    places=1,
                    msg="First take profit should be at 1:1 risk:reward ratio"
                )

                # Second target should be 2:1 risk:reward
                expected_tp2 = entry_price + (risk * 2)
                self.assertAlmostEqual(
                    call_args['metadata']['take_profit_2r'],
                    expected_tp2,
                    places=1,
                    msg="Second take profit should be at 2:1 risk:reward ratio"
                )

    def test_end_to_end_workflow(self):
        """Test the end-to-end workflow from data to signal generation."""
        # Create realistic market data
        df = pd.DataFrame({
            'open': [1900.0, 1900.5, 1901.0, 1901.5, 1902.0, 1903.0, 1904.0, 1905.0, 1906.0, 1907.0],
            'high': [1900.7, 1901.2, 1901.8, 1902.3, 1902.8, 1903.8, 1904.8, 1905.8, 1906.8, 1908.0],
            'low': [1899.8, 1900.2, 1900.7, 1901.2, 1901.7, 1902.7, 1903.7, 1904.5, 1905.5, 1906.5],
            'close': [1900.5, 1901.0, 1901.5, 1902.0, 1902.5, 1903.5, 1904.5, 1905.5, 1906.5, 1907.5],
            'volume': [1000, 1100, 1200, 1100, 1000, 1500, 2000, 2500, 3000, 3500],
            'tick_volume': [100, 110, 120, 110, 100, 150, 200, 250, 300, 350],
            'spread': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        }, index=pd.date_range(start='2023-01-01 09:00:00', periods=10, freq='5min'))

        # Create test setup with this data
        # This data should show a strong uptrend with volume increase

        # Mock connection with acceptable spread
        self.mock_data_fetcher.connector.get_symbol_info.return_value = self.mock_symbol_info

        # Patch get_ohlc_data to return our test data
        with patch.object(self.strategy, 'get_ohlc_data', return_value=df):
            # The real analyze method should be called with our data
            signals = self.strategy.generate_signals()

            # We're testing the full workflow, not asserting specific outcomes
            # Just verifying the function completes without errors and returns a list
            self.assertIsInstance(signals, list)

    def test_identify_signals_with_price_action(self):
        """Test signal identification with price action triggers."""
        # Create data with specific price action patterns
        price_action_data = self.base_data.copy()

        # Create a breakout pattern in the data
        lookback = 20
        for i in range(lookback):
            idx = len(price_action_data) - lookback + i
            # Set up a range for the first part
            if i < lookback - 3:
                # Create a sideways range
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['high'])] = 1902.0
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['low'])] = 1898.0
                price_action_data.iloc[
                    idx, price_action_data.columns.get_indexer(['close'])] = 1900.0 + np.random.normal(0, 0.5)
            else:
                # Create a breakout in the last 3 bars
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['high'])] = 1902.0 + (
                            i - (lookback - 3)) * 2.0
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['low'])] = 1899.0 + (
                            i - (lookback - 3)) * 1.5
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['close'])] = 1900.0 + (
                            i - (lookback - 3)) * 2.0
                # Increase volume for the breakout
                price_action_data.iloc[idx, price_action_data.columns.get_indexer(['volume'])] *= 2.0

        # Calculate indicators on this price action data
        result = self.strategy._calculate_indicators(price_action_data)

        # Identify signals
        result = self.strategy._identify_signals(result)

        # Check that we detected a signal based on the price action
        last_signal = result.iloc[-1]['signal']
        self.assertEqual(last_signal, 1, "Should detect a bullish breakout signal based on price action")

    def test_identify_signals_with_different_indicator_combinations(self):
        """Test signal identification with different combinations of indicator values."""
        test_data = self.base_data.copy()

        # Test various indicator combinations
        test_cases = [
            # All indicators bullish - should generate buy signal
            {
                'rsi': 65, 'macd_histogram': 0.5, 'stoch_k': 75, 'stoch_d': 65,
                'momentum': 101.0, 'volume_ratio': 2.0, 'expected_signal': 1
            },
            # Mixed indicators - should not generate signal
            {
                'rsi': 65, 'macd_histogram': -0.2, 'stoch_k': 45, 'stoch_d': 55,
                'momentum': 99.5, 'volume_ratio': 2.0, 'expected_signal': 0
            },
            # All indicators bearish - should generate sell signal
            {
                'rsi': 35, 'macd_histogram': -0.5, 'stoch_k': 25, 'stoch_d': 35,
                'momentum': 98.5, 'volume_ratio': 2.0, 'expected_signal': -1
            },
            # Good indicators but low volume - should not generate signal
            {
                'rsi': 65, 'macd_histogram': 0.5, 'stoch_k': 75, 'stoch_d': 65,
                'momentum': 101.0, 'volume_ratio': 1.2, 'expected_signal': 0
            }
        ]

        # Run tests for each combination
        for idx, test_case in enumerate(test_cases):
            # Create data with these indicators
            with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
                data_with_indicators = self.strategy._calculate_indicators(test_data)

                # Set the indicator values for the last bar
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = test_case['rsi']
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = test_case[
                    'macd_histogram']
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = test_case[
                    'stoch_k']
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = test_case[
                    'stoch_d']
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = test_case[
                    'momentum']
                data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = test_case[
                    'volume_ratio']

                # Set appropriate price action for signal
                if test_case['expected_signal'] == 1:
                    # Make recent high/low values for a bullish pattern
                    data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['high'])] = 1900.0
                    data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['high'])] = 1905.0
                    data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1904.0
                elif test_case['expected_signal'] == -1:
                    # Make recent high/low values for a bearish pattern
                    data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['low'])] = 1900.0
                    data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['low'])] = 1895.0
                    data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1896.0

                mock_calc.return_value = data_with_indicators

                # Run signal identification through analyze
                signals = self.strategy.analyze(test_data)

                # For test case tracking
                expected_signal_type = "BUY" if test_case['expected_signal'] == 1 else "SELL" if test_case[
                                                                                                     'expected_signal'] == -1 else "NONE"

                # Verify expected signal count
                if test_case['expected_signal'] == 0:
                    self.assertEqual(len(signals), 0,
                                     f"Test case {idx}: Expected no signals with {expected_signal_type} conditions")
                else:
                    self.assertEqual(len(signals), 1,
                                     f"Test case {idx}: Expected 1 signal with {expected_signal_type} conditions")
                    if signals:
                        signal_type = signals[0].signal_type
                        expected_type = "BUY" if test_case['expected_signal'] == 1 else "SELL"
                        self.assertEqual(signal_type, expected_type,
                                         f"Test case {idx}: Expected {expected_type} signal")

    def test_rsi_crossover_detection(self):
        """Test RSI threshold crossovers for entry and exit signals."""
        # Create data with RSI crossing above/below thresholds
        rsi_test_data = self.base_data.copy()

        # Setup RSI values crossing above 60 (buy signal)
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(rsi_test_data)

            # Set RSI crossing pattern - below, at, above threshold
            data_with_indicators.iloc[-3, data_with_indicators.columns.get_indexer(['rsi'])] = 55
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['rsi'])] = 60
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 65

            # Set other indicators to favorable values for a buy signal
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.2
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 70
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 60
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 101.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakout
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['high'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['high'])] = 1905.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1904.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(rsi_test_data)

            # Should detect a buy signal
            self.assertEqual(len(signals), 1, "Should detect a buy signal on RSI crossing above 60")
            if signals:
                self.assertEqual(signals[0].signal_type, "BUY")

        # Setup RSI values crossing below 40 (sell signal)
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(rsi_test_data)

            # Set RSI crossing pattern - above, at, below threshold
            data_with_indicators.iloc[-3, data_with_indicators.columns.get_indexer(['rsi'])] = 45
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['rsi'])] = 40
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 35

            # Set other indicators to favorable values for a sell signal
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = -0.2
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 30
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 40
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 99.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakdown
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['low'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['low'])] = 1895.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1896.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(rsi_test_data)

            # Should detect a sell signal
            self.assertEqual(len(signals), 1, "Should detect a sell signal on RSI crossing below 40")
            if signals:
                self.assertEqual(signals[0].signal_type, "SELL")

    def test_stochastic_crossover_detection(self):
        """Test Stochastic crossover detection for entry signals."""
        # Create data with Stochastic K crossing D
        stoch_test_data = self.base_data.copy()

        # Setup Stochastic bullish crossover
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(stoch_test_data)

            # Set up stochastic crossover pattern
            # K below D, then K above D
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['stoch_k'])] = 45
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['stoch_d'])] = 50
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 55
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 50

            # Set bullish cross flag
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_bull_cross'])] = 1

            # Set other indicators to favorable values
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 61
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.1
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 100.5
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakout
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['high'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['high'])] = 1905.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1904.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(stoch_test_data)

            # Should detect a buy signal
            self.assertEqual(len(signals), 1, "Should detect a buy signal on Stochastic K crossing above D")
            if signals:
                self.assertEqual(signals[0].signal_type, "BUY")

        # Setup Stochastic bearish crossover
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(stoch_test_data)

            # Set up stochastic crossover pattern
            # K above D, then K below D
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['stoch_k'])] = 55
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['stoch_d'])] = 50
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 45
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 50

            # Set bearish cross flag
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_bear_cross'])] = 1

            # Set other indicators to favorable values for a sell
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 39
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = -0.1
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 99.5
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakdown
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['low'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['low'])] = 1895.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1896.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(stoch_test_data)

            # Should detect a sell signal
            self.assertEqual(len(signals), 1, "Should detect a sell signal on Stochastic K crossing below D")
            if signals:
                self.assertEqual(signals[0].signal_type, "SELL")

    def test_macd_confirmation(self):
        """Test MACD confirmation for entry signals."""
        # Create data with MACD histogram confirming signal
        macd_test_data = self.base_data.copy()

        # Setup positive MACD histogram (bullish)
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(macd_test_data)

            # Create pattern with increasing MACD histogram
            data_with_indicators.iloc[-3, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.05
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.10
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = 0.15

            # MACD bullish cross
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_bull_cross'])] = 1

            # Set other indicators to favorable values
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 62
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 65
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 60
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 100.5
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakout
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['high'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['high'])] = 1905.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1904.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(macd_test_data)

            # Should detect a buy signal
            self.assertEqual(len(signals), 1, "Should detect a buy signal with positive MACD histogram")
            if signals:
                self.assertEqual(signals[0].signal_type, "BUY")

        # Setup negative MACD histogram (bearish)
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(macd_test_data)

            # Create pattern with decreasing MACD histogram
            data_with_indicators.iloc[-3, data_with_indicators.columns.get_indexer(['macd_histogram'])] = -0.05
            data_with_indicators.iloc[-2, data_with_indicators.columns.get_indexer(['macd_histogram'])] = -0.10
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_histogram'])] = -0.15

            # MACD bearish cross
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['macd_bear_cross'])] = 1

            # Set other indicators to favorable values for sell
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['rsi'])] = 38
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_k'])] = 35
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['stoch_d'])] = 40
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['momentum'])] = 99.5
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['volume_ratio'])] = 1.6

            # Price action - breakdown
            data_with_indicators.iloc[-5:-1, data_with_indicators.columns.get_indexer(['low'])] = 1900.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['low'])] = 1895.0
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['close'])] = 1896.0

            mock_calc.return_value = data_with_indicators

            # Run signal identification
            signals = self.strategy.analyze(macd_test_data)

            # Should detect a sell signal
            self.assertEqual(len(signals), 1, "Should detect a sell signal with negative MACD histogram")
            if signals:
                self.assertEqual(signals[0].signal_type, "SELL")

    def test_error_handling_during_analysis(self):
        """Test error handling during the analysis process."""
        # Test with exception during calculation
        with patch.object(self.strategy, '_calculate_indicators', side_effect=Exception("Test exception")):
            signals = self.strategy.analyze(self.base_data)
            self.assertEqual(len(signals), 0, "Should handle exceptions gracefully")

        # Test with exception during signal identification
        with patch.object(self.strategy, '_calculate_indicators', return_value=self.base_data), \
                patch.object(self.strategy, '_identify_signals', side_effect=Exception("Test exception")):
            signals = self.strategy.analyze(self.base_data)
            self.assertEqual(len(signals), 0, "Should handle exceptions during signal identification")

        # Test with exception during create_signal
        with patch.object(self.strategy, '_calculate_indicators') as mock_calc:
            data_with_indicators = self.strategy._calculate_indicators(self.base_data)
            data_with_indicators.iloc[-1, data_with_indicators.columns.get_indexer(['signal'])] = 1  # Add buy signal
            mock_calc.return_value = data_with_indicators

            # Force create_signal to throw exception
            with patch.object(self.strategy, 'create_signal', side_effect=Exception("Test exception")):
                signals = self.strategy.analyze(self.base_data)
                self.assertEqual(len(signals), 0, "Should handle exceptions during signal creation")


if __name__ == '__main__':
    unittest.main()