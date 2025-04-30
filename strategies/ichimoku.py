# strategies/ichimoku.py
import numpy as np
import pandas as pd
from db_logger.db_logger import DBLogger
from strategies.base_strategy import BaseStrategy


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Kinko Hyo Strategy for XAU/USD.

    This strategy uses the Ichimoku indicator to trade XAU/USD with a blend of
    trend-following and reversal signals. It identifies high-probability entry and exit
    signals using multiple Ichimoku components (Cloud, Tenkan/Kijun crosses, Chikou Span).

    The strategy operates on H4 timeframe to balance signal reliability and frequency,
    using default Ichimoku settings (9, 26, 52) which work well for gold's behavior.
    """

    def __init__(self, symbol="XAUUSD", timeframe="H4",
                 tenkan_period=9, kijun_period=26, senkou_b_period=52,
                 data_fetcher=None):
        """Initialize the Ichimoku Cloud strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H4".
            tenkan_period (int, optional): Tenkan-sen period. Defaults to 9.
            kijun_period (int, optional): Kijun-sen period. Defaults to 26.
            senkou_b_period (int, optional): Senkou Span B period. Defaults to 52.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Ichimoku_Cloud", data_fetcher=data_fetcher)

        # Validate inputs
        if tenkan_period >= kijun_period:
            error_msg = f"tenkan_period ({tenkan_period}) must be < kijun_period ({kijun_period})"
            DBLogger.log_error("IchimokuStrategy", error_msg)
            raise ValueError(error_msg)
        if kijun_period >= senkou_b_period:
            error_msg = f"kijun_period ({kijun_period}) must be < senkou_b_period ({senkou_b_period})"
            DBLogger.log_error("IchimokuStrategy", error_msg)
            raise ValueError(error_msg)

        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

        # Ensure we fetch enough data for calculations
        # We need enough data for:
        # 1. Senkou Span B calculation (52 periods)
        # 2. The displacement (26 periods forward)
        # 3. Chikou Span (26 periods back)
        # 4. Extra bars for analysis
        self.min_required_candles = self.senkou_b_period + self.kijun_period * 2 + 30

        DBLogger.log_event("INFO",
                           f"Initialized Ichimoku Cloud strategy: {symbol} {timeframe}, "
                           f"Tenkan: {tenkan_period}, Kijun: {kijun_period}, Senkou B: {senkou_b_period}",
                           "IchimokuStrategy")

    def calculate_indicators(self, data):
        """Calculate Ichimoku indicators on OHLC data.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with Ichimoku indicators added
        """
        if len(data) < self.min_required_candles:
            DBLogger.log_event("WARNING",
                               f"Insufficient data for Ichimoku calculations. "
                               f"Need at least {self.min_required_candles} candles.",
                               "IchimokuStrategy")
            return data

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        period_high = data['high'].rolling(window=self.tenkan_period).max()
        period_low = data['low'].rolling(window=self.tenkan_period).min()
        data['tenkan_sen'] = (period_high + period_low) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        period_high = data['high'].rolling(window=self.kijun_period).max()
        period_low = data['low'].rolling(window=self.kijun_period).min()
        data['kijun_sen'] = (period_high + period_low) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(self.kijun_period)

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, plotted 26 periods ahead
        period_high = data['high'].rolling(window=self.senkou_b_period).max()
        period_low = data['low'].rolling(window=self.senkou_b_period).min()
        data['senkou_span_b'] = ((period_high + period_low) / 2).shift(self.kijun_period)

        # Calculate Chikou Span (Lagging Span): Current closing price, plotted 26 periods back
        data['chikou_span'] = data['close'].shift(-self.kijun_period)

        # Calculate average true range for stop loss placement
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Identify Kumo (Cloud) color - Bullish when Senkou A > Senkou B
        data['cloud_bullish'] = data['senkou_span_a'] > data['senkou_span_b']

        # Calculate cloud top and bottom at each position
        data['cloud_top'] = data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        data['cloud_bottom'] = data[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        # Identify if price is above, below, or inside the cloud
        data['price_above_cloud'] = data['close'] > data['cloud_top']
        data['price_below_cloud'] = data['close'] < data['cloud_bottom']
        data['price_in_cloud'] = (~data['price_above_cloud']) & (~data['price_below_cloud'])

        # Check Chikou Span position relative to price 26 periods ago
        data['chikou_above_price'] = data['chikou_span'].shift(self.kijun_period) > data['close']
        data['chikou_below_price'] = data['chikou_span'].shift(self.kijun_period) < data['close']

        # Identify TK cross signals and all types of signals
        data = self._identify_signals(data)

        return data

    def _identify_signals(self, data):
        """Identify both trend-following and reversal signals based on Ichimoku components.

        Args:
            data (pandas.DataFrame): OHLC data with Ichimoku indicators

        Returns:
            pandas.DataFrame: Data with signal information
        """
        # Initialize signal columns
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data['signal_type'] = ""  # "TREND" or "REVERSAL"
        data['signal_strength'] = 0.0
        data['stop_loss'] = np.nan
        data['take_profit_1'] = np.nan
        data['take_profit_2'] = np.nan

        for i in range(self.kijun_period * 2, len(data)):
            # Skip if not enough prior data
            if i < self.kijun_period * 2:
                continue

            # Get current values (current completed candle)
            current_close = data.iloc[i]['close']
            current_tenkan = data.iloc[i]['tenkan_sen']
            current_kijun = data.iloc[i]['kijun_sen']
            current_span_a = data.iloc[i]['senkou_span_a']
            current_span_b = data.iloc[i]['senkou_span_b']
            current_chikou = data.iloc[i - self.kijun_period]['chikou_span']  # Chikou is plotted 26 periods back

            # Get previous values
            prev_tenkan = data.iloc[i - 1]['tenkan_sen']
            prev_kijun = data.iloc[i - 1]['kijun_sen']
            prev_chikou = data.iloc[i - self.kijun_period - 1]['chikou_span']

            # Calculate cloud top and bottom at current position
            cloud_top = max(current_span_a, current_span_b)
            cloud_bottom = min(current_span_a, current_span_b)

            # Determine if Chikou Span is above/below price 26 periods ago
            # Note: To properly check this, we need to look at current price vs. price that was 26 periods before current
            price_26_periods_ago = data.iloc[i - self.kijun_period]['close']
            chikou_above_price = current_close > price_26_periods_ago

            # Determine if we have a TK cross
            # Tenkan crosses above Kijun (bullish cross)
            tk_bullish_cross = prev_tenkan <= prev_kijun and current_tenkan > current_kijun

            # Tenkan crosses below Kijun (bearish cross)
            tk_bearish_cross = prev_tenkan >= prev_kijun and current_tenkan < current_kijun

            # Check if Tenkan is already above/below Kijun (continuing trend)
            tk_bullish_alignment = current_tenkan > current_kijun
            tk_bearish_alignment = current_tenkan < current_kijun

            # Check cloud direction (is future cloud bullish/bearish)
            future_cloud_bullish = current_span_a > current_span_b

            # Check if cloud is very thin (potential ranging market)
            cloud_thickness = abs(current_span_a - current_span_b) / ((current_span_a + current_span_b) / 2)
            cloud_is_thin = cloud_thickness < 0.01  # Less than 1% difference between spans

            # ----------------------------------------------------------------------
            # TREND-FOLLOWING SIGNALS
            # ----------------------------------------------------------------------

            # Bullish Trend Signal Conditions:
            # 1. Price is above the Cloud
            # 2. Cloud ahead is bullish (Senkou A > Senkou B)
            # 3. Tenkan crosses above Kijun OR is already above (bullish alignment)
            # 4. Chikou Span is above price from 26 periods ago
            if (current_close > cloud_top and
                    future_cloud_bullish and
                    (tk_bullish_cross or tk_bullish_alignment) and
                    chikou_above_price and
                    not cloud_is_thin):  # Skip if cloud is too thin (ranging market)

                # Find a good stop loss level based on Ichimoku
                # For bullish trend trades, stop loss should be below the Kijun-sen or cloud bottom
                stop_loss = min(current_kijun, cloud_bottom)

                # If Kijun/cloud is too far for a reasonable stop, use ATR
                risk_distance = current_close - stop_loss
                if risk_distance > data.iloc[i]['atr'] * 3:
                    stop_loss = current_close - data.iloc[i]['atr'] * 2

                # Calculate take profit levels (1.5:1 and 3:1 reward-to-risk) as per the plan
                risk = current_close - stop_loss
                take_profit_1 = current_close + (risk * 1.5)  # 1.5:1 reward-to-risk
                take_profit_2 = current_close + (risk * 3)  # 3:1 reward-to-risk

                # Store both take profit levels
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2

                # Calculate signal strength based on multiple factors
                cloud_thickness_factor = min(1.0, cloud_thickness * 5)  # Thicker cloud = stronger trend
                tk_decisiveness = min(1.0, abs((current_tenkan - current_kijun) / current_kijun) * 10)
                chikou_strength = min(1.0, abs((current_close - price_26_periods_ago) / price_26_periods_ago) * 10)

                strength = min(1.0, (0.4 * cloud_thickness_factor + 0.3 * tk_decisiveness + 0.3 * chikou_strength))

                # Generate TREND buy signal
                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_type'] = "TREND"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss

            # Bearish Trend Signal Conditions:
            # 1. Price is below the Cloud
            # 2. Cloud ahead is bearish (Senkou A < Senkou B)
            # 3. Tenkan crosses below Kijun OR is already below (bearish alignment)
            # 4. Chikou Span is below price from 26 periods ago
            elif (current_close < cloud_bottom and
                  not future_cloud_bullish and
                  (tk_bearish_cross or tk_bearish_alignment) and
                  not chikou_above_price and
                  not cloud_is_thin):  # Skip if cloud is too thin (ranging market)

                # Find a good stop loss level based on Ichimoku
                # For bearish trend trades, stop loss should be above the Kijun-sen or cloud top
                stop_loss = max(current_kijun, cloud_top)

                # If Kijun/cloud is too far for a reasonable stop, use ATR
                risk_distance = stop_loss - current_close
                if risk_distance > data.iloc[i]['atr'] * 3:
                    stop_loss = current_close + data.iloc[i]['atr'] * 2

                # Calculate take profit levels (1.5:1 and 3:1 reward-to-risk) as per the plan
                risk = stop_loss - current_close
                take_profit_1 = current_close - (risk * 1.5)  # 1.5:1 reward-to-risk
                take_profit_2 = current_close - (risk * 3)  # 3:1 reward-to-risk

                # Store both take profit levels
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2

                # Calculate signal strength based on multiple factors
                cloud_thickness_factor = min(1.0, cloud_thickness * 5)  # Thicker cloud = stronger trend
                tk_decisiveness = min(1.0, abs((current_kijun - current_tenkan) / current_kijun) * 10)
                chikou_strength = min(1.0, abs((price_26_periods_ago - current_close) / price_26_periods_ago) * 10)

                strength = min(1.0, (0.4 * cloud_thickness_factor + 0.3 * tk_decisiveness + 0.3 * chikou_strength))

                # Generate TREND sell signal
                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_type'] = "TREND"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss

            # ----------------------------------------------------------------------
            # REVERSAL SIGNALS
            # ----------------------------------------------------------------------

            # Bullish Reversal Signal Conditions:
            # 1. Prior trend was bearish (price below cloud)
            # 2. Bullish TK cross occurs while price is still below/inside cloud
            # 3. Chikou Span is rising and crossing above past price
            elif (current_close <= cloud_top and  # Price below or inside cloud
                  tk_bullish_cross and  # Tenkan crosses above Kijun
                  current_chikou > prev_chikou and  # Chikou is rising
                  (current_chikou > price_26_periods_ago or  # Chikou above price 26 bars ago
                   abs(current_chikou - price_26_periods_ago) < data.iloc[i]['atr'])):  # Or very close to it

                # For reversal trades, use a tighter stop loss since they're higher risk
                stop_loss = current_close - data.iloc[i]['atr'] * 1.5

                # Calculate take profit at the cloud top - this is a logical target for a reversal
                take_profit_1 = cloud_top

                # If cloud top is too close, extend target to 1.5x risk
                risk = current_close - stop_loss
                if (take_profit_1 - current_close) < risk * 1.5:
                    take_profit_1 = current_close + (risk * 1.5)

                # Second take profit at 2:1 reward-to-risk for reversals
                take_profit_2 = current_close + (risk * 2)

                # Store both take profit levels
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2

                # Reversal trades have lower strength initially
                strength = min(0.7, abs((current_tenkan - current_kijun) / current_kijun) * 5)

                # Generate REVERSAL buy signal
                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_type'] = "REVERSAL"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss

            # Bearish Reversal Signal Conditions:
            # 1. Prior trend was bullish (price above cloud)
            # 2. Bearish TK cross occurs while price is still above/inside cloud
            # 3. Chikou Span is falling and crossing below past price
            elif (current_close >= cloud_bottom and  # Price above or inside cloud
                  tk_bearish_cross and  # Tenkan crosses below Kijun
                  current_chikou < prev_chikou and  # Chikou is falling
                  (current_chikou < price_26_periods_ago or  # Chikou below price 26 bars ago
                   abs(current_chikou - price_26_periods_ago) < data.iloc[i]['atr'])):  # Or very close to it

                # For reversal trades, use a tighter stop loss since they're higher risk
                stop_loss = current_close + data.iloc[i]['atr'] * 1.5

                # Calculate take profit at the cloud bottom - logical target for a reversal
                take_profit_1 = cloud_bottom

                # If cloud bottom is too close, extend target to 1.5x risk
                risk = stop_loss - current_close
                if (current_close - take_profit_1) < risk * 1.5:
                    take_profit_1 = current_close - (risk * 1.5)

                # Second take profit at 2:1 reward-to-risk for reversals
                take_profit_2 = current_close - (risk * 2)

                # Store both take profit levels
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2

                # Reversal trades have lower strength initially
                strength = min(0.7, abs((current_kijun - current_tenkan) / current_kijun) * 5)

                # Generate REVERSAL sell signal
                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_type'] = "REVERSAL"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss

            # ----------------------------------------------------------------------
            # EXIT SIGNALS
            # ----------------------------------------------------------------------

            # Check for potential exit signals for current positions
            # These would be implemented in the order_manager or trailing_stop_manager
            # For example:

            # Exit longs on:
            # - Bearish TK Cross
            # - Price moving back below Kijun
            # - Price entering the cloud from above

            # Exit shorts on:
            # - Bullish TK Cross
            # - Price moving back above Kijun
            # - Price entering the cloud from below

            # Note: Actual position management is handled separately, but we're
            # identifying potential exit points for the trailing stop manager to use

        return data

    def analyze(self, data):
        """Analyze market data and generate trading signals.

        Args:
            data (pandas.DataFrame): OHLC data for analysis

        Returns:
            list: Generated trading signals
        """
        # Calculate indicators
        data = self.calculate_indicators(data)

        # Check if we have sufficient data after calculations
        if data.empty or 'signal' not in data.columns:
            DBLogger.log_event("Warning", "Insufficient data for Ichimoku analysis after calculations")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]

        # Check for trading signal on the last candle
        if last_candle['signal'] == 1:  # Buy signal
            signal_type = last_candle['signal_type']

            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit = last_candle['take_profit']

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate risk in dollars
            risk = entry_price - stop_loss

            # Calculate multiple take profit levels
            take_profit_1 = last_candle['take_profit']  # Use the calculated take profit
            take_profit_2 = entry_price + (risk * 3) if signal_type == "TREND" else entry_price + (risk * 2)

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'tenkan_sen': last_candle['tenkan_sen'],
                    'kijun_sen': last_candle['kijun_sen'],
                    'senkou_span_a': last_candle['senkou_span_a'],
                    'senkou_span_b': last_candle['senkou_span_b'],
                    'cloud_bullish': bool(last_candle['cloud_bullish']),
                    'strategy_signal_type': signal_type,
                    'reason': f'Bullish {signal_type} signal: TK alignment with Chikou confirmation'
                }
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                               f"Generated BUY ({signal_type}) signal for {self.symbol} at {entry_price}. "
                               f"Tenkan: {last_candle['tenkan_sen']:.2f}, Kijun: {last_candle['kijun_sen']:.2f}, "
                               f"Cloud: {'Bullish' if last_candle['cloud_bullish'] else 'Bearish'}, "
                               f"Stop: {stop_loss:.2f}, Target: {take_profit_1:.2f}",
                               "IchimokuStrategy"
                               )

        elif last_candle['signal'] == -1:  # Sell signal
            signal_type = last_candle['signal_type']

            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit = last_candle['take_profit']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate risk in dollars
            risk = stop_loss - entry_price

            # Use the take profit calculated during signal identification
            take_profit_1 = last_candle['take_profit']

            # Calculate the second take profit target based on signal type
            take_profit_2 = entry_price - (risk * 3) if signal_type == "TREND" else entry_price - (risk * 2)

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'tenkan_sen': last_candle['tenkan_sen'],
                    'kijun_sen': last_candle['kijun_sen'],
                    'senkou_span_a': last_candle['senkou_span_a'],
                    'senkou_span_b': last_candle['senkou_span_b'],
                    'cloud_bullish': bool(last_candle['cloud_bullish']),
                    'strategy_signal_type': signal_type,
                    'reason': f'Bearish {signal_type} signal: TK alignment with Chikou confirmation'
                }
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                               f"Generated SELL ({signal_type}) signal for {self.symbol} at {entry_price}. "
                               f"Tenkan: {last_candle['tenkan_sen']:.2f}, Kijun: {last_candle['kijun_sen']:.2f}, "
                               f"Cloud: {'Bullish' if last_candle['cloud_bullish'] else 'Bearish'}, "
                               f"Stop: {stop_loss:.2f}, Target: {take_profit_1:.2f}",
                               "IchimokuStrategy"
                               )

        return signals