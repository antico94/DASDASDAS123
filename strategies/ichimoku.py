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
                 data_fetcher=None, debug_mode=True):  # Added debug_mode here
        """Initialize the Ichimoku Cloud strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H4".
            tenkan_period (int, optional): Tenkan-sen period. Defaults to 9.
            kijun_period (int, optional): Kijun-sen period. Defaults to 26.
            senkou_b_period (int, optional): Senkou Span B period. Defaults to 52.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
            debug_mode (bool, optional): Enable detailed debugging output. Defaults to False.
        """
        super().__init__(symbol, timeframe, name="Ichimoku_Cloud", data_fetcher=data_fetcher)

        # Validate inputs - standard Ichimoku assumes Tenkan < Kijun < Senkou B
        if not (tenkan_period < kijun_period < senkou_b_period):
            error_msg = (f"Ichimoku periods must satisfy Tenkan ({tenkan_period}) < Kijun ({kijun_period}) "
                         f"< Senkou B ({senkou_b_period}) for standard interpretation.")
            DBLogger.log_error("IchimokuStrategy", error_msg)
            raise ValueError(error_msg)

        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.debug_mode = debug_mode  # Set the debug flag

        # Ensure we fetch enough data for calculations
        # We need enough data for:
        # 1. Senkou Span B calculation (senkou_b_period)
        # 2. The displacement (kijun_period periods forward) for Senkou A/B
        # 3. The Chikou Span comparison requires price from kijun_period periods back.
        # To have valid Senkou A/B *at the current bar*, we need history going back
        # senkou_b_period + kijun_period.
        # To compare current close with price kijun_period bars ago, we need kijun_period history.
        # The maximum lookback is senkou_b_period + kijun_period. Add a buffer.
        self.min_required_candles = self.senkou_b_period + self.kijun_period + 30

        DBLogger.log_event("INFO",
                           f"Initialized Ichimoku Cloud strategy: {symbol} {timeframe}, "
                           f"Tenkan: {tenkan_period}, Kijun: {kijun_period}, Senkou B: {senkou_b_period}",
                           "IchimokuStrategy")
        if self.debug_mode:
            DBLogger.log_event("INFO", "Ichimoku Strategy Debugging Mode is ENABLED.", "IchimokuStrategy")

    def calculate_indicators(self, data):
        """Calculate Ichimoku indicators on OHLC data.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with Ichimoku indicators added
        """
        # Create a copy to avoid modifying the original DataFrame passed in
        data = data.copy()

        # Define columns to be added and their default types/values
        indicator_cols_numeric = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
                                  'chikou_span', 'atr', 'cloud_top', 'cloud_bottom',
                                  'price_above_cloud', 'price_below_cloud', 'price_in_cloud',
                                  'signal', 'signal_strength', 'stop_loss', 'take_profit_1', 'take_profit_2']
        indicator_cols_boolean = ['cloud_bullish']
        indicator_cols_object = ['signal_type']  # Use object dtype for strings/None

        if len(data) < self.min_required_candles:
            DBLogger.log_event("WARNING",
                               f"Insufficient data for Ichimoku calculations. "
                               f"Need at least {self.min_required_candles} candles, but got {len(data)}.",
                               "IchimokuStrategy")
            # Add columns with appropriate dtypes, initialized to NaN/None
            for col in indicator_cols_numeric + indicator_cols_boolean:
                if col not in data.columns:
                    data[col] = np.nan  # Numeric and boolean columns can use NaN
            for col in indicator_cols_object:
                if col not in data.columns:
                    data[col] = None  # Use None for object columns (string) to avoid type issues

            return data

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        data['tenkan_sen'] = (data['high'].rolling(window=self.tenkan_period).max() +
                              data['low'].rolling(window=self.tenkan_period).min()) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        data['kijun_sen'] = (data['high'].rolling(window=self.kijun_period).max() +
                             data['low'].rolling(window=self.kijun_period).min()) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        # Shift forward by kijun_period
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(self.kijun_period)

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, plotted 26 periods ahead
        # Shift forward by kijun_period
        longest_period_hh = data['high'].rolling(window=self.senkou_b_period).max()
        longest_period_ll = data['low'].rolling(window=self.senkou_b_period).min()
        data['senkou_span_b'] = ((longest_period_hh + longest_period_ll) / 2).shift(self.kijun_period)

        # Calculate Chikou Span (Lagging Span): Current closing price, plotted 26 periods back
        # Shift backward by kijun_period
        # CORRECTED: Shift(self.kijun_period) moves the current value to kijun_period bars ago
        data['chikou_span'] = data['close'].shift(self.kijun_period)

        # Calculate average true range for stop loss placement
        # This is a standard ATR calculation, common in trading strategies
        high = data['high']
        low = data['low']
        # Ensure we have a previous close; default to current close if not enough data
        close = data['close'].shift(1).fillna(data['close'])

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()  # Using a standard 14-period window for ATR

        # Identify Kumo (Cloud) color - Bullish when Senkou A > Senkou B
        data['cloud_bullish'] = data['senkou_span_a'] > data['senkou_span_b']

        # Calculate cloud top and bottom at each position
        data['cloud_top'] = data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        data['cloud_bottom'] = data[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        # Identify if price is above, below, or inside the cloud
        data['price_above_cloud'] = data['close'] > data['cloud_top']
        data['price_below_cloud'] = data['close'] < data['cloud_bottom']
        data['price_in_cloud'] = (~data['price_above_cloud']) & (~data['price_below_cloud'])

        # Note: Chikou Span position relative to price is better checked
        # directly in the signal logic loop using current close vs close N periods ago.
        # The previous calculation using the shifted column was incorrect.
        # We will perform this check directly in _identify_signals.

        # Identify TK cross signals and all types of signals
        # This step is handled in a separate method for clarity and modularity
        data = self._identify_signals(data)

        return data

    def _identify_signals(self, data):
        """Identify both trend-following and reversal signals based on Ichimoku components.

        Args:
            data (pandas.DataFrame): OHLC data with Ichimoku indicators

        Returns:
            pandas.DataFrame: Data with signal information
        """
        # Initialize signal columns if they don't exist with appropriate dtypes
        if 'signal' not in data.columns:
            data['signal'] = np.nan  # Numeric column
        if 'signal_type' not in data.columns:
            # Initialize with None to ensure object dtype for strings/None
            data['signal_type'] = None  # Object column
        if 'signal_strength' not in data.columns:
            data['signal_strength'] = np.nan  # Numeric column
        if 'stop_loss' not in data.columns:
            data['stop_loss'] = np.nan  # Numeric column
        if 'take_profit_1' not in data.columns:
            data['take_profit_1'] = np.nan  # Numeric column
        if 'take_profit_2' not in data.columns:
            data['take_profit_2'] = np.nan  # Numeric column

        # Need enough data for indicators AND enough history for comparisons
        # Loop needs to start after indicators have values and previous data is available
        # Starting after senkou_b_period + kijun_period ensures shifted spans are available
        # and price N periods ago is available for Chikou comparison.
        start_index = max(self.senkou_b_period + self.kijun_period, 14)  # Need at least 14 for ATR

        # Ensure there's enough data even after determining the start_index
        if len(data) <= start_index:
            # If not enough data to even start the loop, return data as is (likely with NaNs)
            return data

        for i in range(start_index, len(data)):
            # Ensure required data points are not NaN for current bar
            if (pd.isna(data.iloc[i]['tenkan_sen']) or pd.isna(data.iloc[i]['kijun_sen']) or
                    pd.isna(data.iloc[i]['senkou_span_a']) or pd.isna(data.iloc[i]['senkou_span_b']) or
                    pd.isna(data.iloc[i]['atr'])):
                # Explicitly reset signal for this bar if indicators are missing
                data.loc[data.index[i], ['signal', 'signal_type', 'signal_strength', 'stop_loss', 'take_profit_1',
                                         'take_profit_2']] = [np.nan, None, np.nan, np.nan, np.nan, np.nan]
                continue  # Skip if indicators are not fully calculated yet

            # Get current values (current completed candle)
            current_close = data.iloc[i]['close']
            current_tenkan = data.iloc[i]['tenkan_sen']
            current_kijun = data.iloc[i]['kijun_sen']
            current_span_a = data.iloc[i]['senkou_span_a']
            current_span_b = data.iloc[i]['senkou_span_b']
            current_atr = data.iloc[i]['atr']
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']

            # Get previous values for crosses
            # Ensure previous data exists (checked by start_index)
            prev_tenkan = data.iloc[i - 1]['tenkan_sen']
            prev_kijun = data.iloc[i - 1]['kijun_sen']

            # Get price kijun_period bars ago for Chikou comparison
            # Ensure data from i - kijun_period exists (checked by start_index)
            price_kijun_periods_ago = data.iloc[i - self.kijun_period]['close']

            # Calculate cloud top and bottom at current position
            cloud_top = max(current_span_a, current_span_b)
            cloud_bottom = min(current_span_a, current_span_b)

            # Determine if Chikou Span (current close) is above/below price kijun periods ago
            # This is the standard Chikou signal check
            chikou_above_price_signal = current_close > price_kijun_periods_ago
            chikou_below_price_signal = current_close < price_kijun_periods_ago

            # Determine if we have a TK cross
            # Tenkan crosses above Kijun (bullish cross) - check at the close of the current bar
            tk_bullish_cross = prev_tenkan <= prev_kijun and current_tenkan > current_kijun

            # Tenkan crosses below Kijun (bearish cross) - check at the close of the current bar
            tk_bearish_cross = prev_tenkan >= prev_kijun and current_tenkan < current_kijun

            # Check if Tenkan is already above/below Kijun (continuing trend)
            tk_bullish_alignment = current_tenkan > current_kijun
            tk_bearish_alignment = current_tenkan < current_kijun

            # Check cloud direction (is future cloud bullish/bearish) - Senkou A vs B at current bar
            future_cloud_bullish = current_span_a > current_span_b

            # Check if cloud is very thin (potential ranging market)
            cloud_thickness = abs(current_span_a - current_span_b)
            average_span = (current_span_a + current_span_b) / 2
            # Avoid division by zero if spans are near zero (unlikely with price data)
            cloud_is_thin = (cloud_thickness / average_span) < 0.01 if average_span != 0 else False

            # ---------- ENHANCED DEBUGGING START ----------
            if self.debug_mode:
                # Prepare condition checklist for bullish signals
                bullish_checks = [
                    f"Price > Cloud Top: {'✅' if current_close > cloud_top else '❌'} ({current_close:.5f} > {cloud_top:.5f})",
                    f"Future Cloud Bullish (A > B): {'✅' if future_cloud_bullish else '❌'} ({current_span_a:.5f} > {current_span_b:.5f})",
                    f"TK Bullish Cross: {'✅' if tk_bullish_cross else '❌'} (Prev:{prev_tenkan:.5f}/{prev_kijun:.5f}, Curr:{current_tenkan:.5f}/{current_kijun:.5f})",
                    f"TK Bullish Alignment: {'✅' if tk_bullish_alignment else '❌'} ({current_tenkan:.5f} > {current_kijun:.5f})",
                    f"Chikou (Current Close) > Price {self.kijun_period} periods ago: {'✅' if chikou_above_price_signal else '❌'} ({current_close:.5f} > {price_kijun_periods_ago:.5f})",
                    f"Cloud is NOT Thin: {'✅' if not cloud_is_thin else '❌'} (Thickness: {cloud_thickness:.5f}, Avg:{average_span:.5f})"
                ]

                # Prepare condition checklist for bearish signals
                bearish_checks = [
                    f"Price < Cloud Bottom: {'✅' if current_close < cloud_bottom else '❌'} ({current_close:.5f} < {cloud_bottom:.5f})",
                    f"Future Cloud Bearish (A < B): {'✅' if not future_cloud_bullish else '❌'} ({current_span_a:.5f} < {current_span_b:.5f})",
                    f"TK Bearish Cross: {'✅' if tk_bearish_cross else '❌'} (Prev:{prev_tenkan:.5f}/{prev_kijun:.5f}, Curr:{current_tenkan:.5f}/{current_kijun:.5f})",
                    f"TK Bearish Alignment: {'✅' if tk_bearish_alignment else '❌'} ({current_tenkan:.5f} < {current_kijun:.5f})",
                    f"Chikou (Current Close) < Price {self.kijun_period} periods ago: {'✅' if chikou_below_price_signal else '❌'} ({current_close:.5f} < {price_kijun_periods_ago:.5f})",
                    f"Cloud is NOT Thin: {'✅' if not cloud_is_thin else '❌'} (Thickness: {cloud_thickness:.5f}, Avg:{average_span:.5f})"
                ]

                # Prepare condition checklist for bullish REVERSAL signals
                bullish_reversal_checks = [
                    f"Bullish TK Cross: {'✅' if tk_bullish_cross else '❌'} (Prev:{prev_tenkan:.5f}/{prev_kijun:.5f}, Curr:{current_tenkan:.5f}/{current_kijun:.5f})",
                    f"Price <= Cloud Top: {'✅' if current_close <= cloud_top else '❌'} ({current_close:.5f} <= {cloud_top:.5f})",
                    f"Chikou (Current Close) >= Price {self.kijun_period} periods ago OR near: {'✅' if (chikou_above_price_signal or abs(current_close - price_kijun_periods_ago) < current_atr) else '❌'} (Curr:{current_close:.5f}, Past:{price_kijun_periods_ago:.5f}, ATR:{current_atr:.5f})"
                ]

                # Prepare condition checklist for bearish REVERSAL signals
                bearish_reversal_checks = [
                    f"Bearish TK Cross: {'✅' if tk_bearish_cross else '❌'} (Prev:{prev_tenkan:.5f}/{prev_kijun:.5f}, Curr:{current_tenkan:.5f}/{current_kijun:.5f})",
                    f"Price >= Cloud Bottom: {'✅' if current_close >= cloud_bottom else '❌'} ({current_close:.5f} >= {cloud_bottom:.5f})",
                    f"Chikou (Current Close) <= Price {self.kijun_period} periods ago OR near: {'✅' if (chikou_below_price_signal or abs(current_close - price_kijun_periods_ago) < current_atr) else '❌'} (Curr:{current_close:.5f}, Past:{price_kijun_periods_ago:.5f}, ATR:{current_atr:.5f})"
                ]

                # Get the timestamp for the current bar
                current_time = data.index[i]
                if i == len(data) - 1:
                    print(f"\n=== Ichimoku Strategy - Bar: {current_time} ===")
                    print("--- Bullish Signal Conditions ---")
                    # Check if ALL bullish TREND conditions are met
                    all_bullish_trend_met = (current_close > cloud_top and
                                             future_cloud_bullish and
                                             (tk_bullish_cross or tk_bullish_alignment) and
                                             chikou_above_price_signal and
                                             not cloud_is_thin)

                    print(f"Overall Bullish TREND Signal Possible: {'✅' if all_bullish_trend_met else '❌'}")
                    for idx, check in enumerate(bullish_checks, start=1):
                        print(f"  {idx}. {check}")

                    print("\n--- Bearish Signal Conditions ---")
                    # Check if ALL bearish TREND conditions are met
                    all_bearish_trend_met = (current_close < cloud_bottom and
                                             not future_cloud_bullish and
                                             (tk_bearish_cross or tk_bearish_alignment) and
                                             chikou_below_price_signal and
                                             not cloud_is_thin)
                    print(f"Overall Bearish TREND Signal Possible: {'✅' if all_bearish_trend_met else '❌'}")
                    for idx, check in enumerate(bearish_checks, start=1):
                        print(f"  {idx}. {check}")

                    print("\n--- Bullish REVERSAL Signal Conditions ---")
                    # Check if ALL bullish REVERSAL conditions are met
                    all_bullish_reversal_met = (tk_bullish_cross and
                                                current_close <= cloud_top and
                                                (chikou_above_price_signal or abs(
                                                    current_close - price_kijun_periods_ago) < current_atr))
                    print(f"Overall Bullish REVERSAL Signal Possible: {'✅' if all_bullish_reversal_met else '❌'}")
                    for idx, check in enumerate(bullish_reversal_checks, start=1):
                        print(f"  {idx}. {check}")

                    print("\n--- Bearish REVERSAL Signal Conditions ---")
                    # Check if ALL bearish REVERSAL conditions are met
                    all_bearish_reversal_met = (tk_bearish_cross and
                                                current_close >= cloud_bottom and
                                                (chikou_below_price_signal or abs(
                                                    current_close - price_kijun_periods_ago) < current_atr))
                    print(f"Overall Bearish REVERSAL Signal Possible: {'✅' if all_bearish_reversal_met else '❌'}")
                    for idx, check in enumerate(bearish_reversal_checks, start=1):
                        print(f"  {idx}. {check}")

                    print("-------------------------------------------")

                # ---------- ENHANCED DEBUGGING END ----------

            # Reset signal columns for the current bar before potentially setting a new signal
            data.loc[data.index[i], ['signal', 'signal_type', 'signal_strength', 'stop_loss', 'take_profit_1',
                                     'take_profit_2']] = [np.nan, None, np.nan, np.nan, np.nan, np.nan]

            # ----------------------------------------------------------------------
            # TREND-FOLLOWING SIGNALS - STRONG SIGNALS
            # Require alignment of Price, Cloud, TK, and Chikou
            # ----------------------------------------------------------------------

            # Bullish Trend Signal Conditions:
            # 1. Price is above the Cloud
            # 2. Cloud ahead is bullish (Senkou A > Senkou B)
            # 3. Tenkan crosses above Kijun OR is already above (bullish alignment)
            # 4. Chikou Span (current close) is above price from kijun periods ago
            if (current_close > cloud_top and
                    future_cloud_bullish and
                    (tk_bullish_cross or tk_bullish_alignment) and
                    chikou_above_price_signal and
                    not cloud_is_thin):  # Avoid thin cloud (ranging)

                # Stop loss below Kijun or cloud bottom, whichever is higher (safer)
                # Corrected for buy signal SL below price
                stop_loss_option1 = current_kijun
                stop_loss_option2 = cloud_bottom
                # Choose the higher of the two levels below current price
                stop_loss = max(stop_loss_option1,
                                stop_loss_option2) if stop_loss_option1 < current_close and stop_loss_option2 < current_close else current_close - current_atr * 2  # Fallback to ATR if both levels are above price

                # Ensure stop loss is below current price for a buy and is not NaN
                if pd.isna(stop_loss) or stop_loss >= current_close:
                    stop_loss = current_close - current_atr * 2  # Use ATR if Ichimoku level is not suitable or NaN

                # Calculate take profit levels (1.5:1 and 3:1 reward-to-risk)
                risk = current_close - stop_loss
                take_profit_1 = current_close + (risk * 1.5)
                take_profit_2 = current_close + (risk * 3.0)

                # Calculate signal strength
                # Strength factors: Price vs Cloud distance, TK spread, Chikou vs Price distance, Cloud thickness
                price_cloud_dist_factor = (current_close - cloud_top) / cloud_top if cloud_top > 0 else 0
                tk_spread_factor = (current_tenkan - current_kijun) / current_kijun if current_kijun > 0 else 0
                chikou_dist_factor = (
                                                 current_close - price_kijun_periods_ago) / price_kijun_periods_ago if price_kijun_periods_ago > 0 else 0
                cloud_thickness_factor = (cloud_thickness / average_span) if average_span > 0 else 0

                strength = min(1.0, max(0, (
                            price_cloud_dist_factor * 0.3 + tk_spread_factor * 0.3 + chikou_dist_factor * 0.2 + cloud_thickness_factor * 0.2)))

                # Generate TREND buy signal
                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_type'] = "TREND"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2


            # Bearish Trend Signal Conditions:
            # 1. Price is below the Cloud
            # 2. Cloud ahead is bearish (Senkou A < Senkou B)
            # 3. Tenkan crosses below Kijun OR is already below (bearish alignment)
            # 4. Chikou Span (current close) is below price from kijun periods ago
            elif (current_close < cloud_bottom and
                  not future_cloud_bullish and
                  (tk_bearish_cross or tk_bearish_alignment) and
                  chikou_below_price_signal and
                  not cloud_is_thin):  # Avoid thin cloud (ranging)

                # Stop loss above Kijun or cloud top, whichever is lower (safer)
                # Corrected for sell signal SL above price
                stop_loss_option1 = current_kijun
                stop_loss_option2 = cloud_top
                # Choose the lower of the two levels above current price
                stop_loss = min(stop_loss_option1,
                                stop_loss_option2) if stop_loss_option1 > current_close and stop_loss_option2 > current_close else current_close + current_atr * 2  # Fallback to ATR if both levels are below price

                # Ensure stop loss is above current price for a sell and is not NaN
                if pd.isna(stop_loss) or stop_loss <= current_close:
                    stop_loss = current_close + current_atr * 2  # Use ATR if Ichimoku level is not suitable or NaN

                # Calculate take profit levels (1.5:1 and 3:1 reward-to-risk)
                risk = stop_loss - current_close
                take_profit_1 = current_close - (risk * 1.5)
                take_profit_2 = current_close - (risk * 3.0)

                # Calculate signal strength
                # Strength factors: Price vs Cloud distance, TK spread, Chikou vs Price distance, Cloud thickness
                price_cloud_dist_factor = (
                                                      cloud_bottom - current_close) / cloud_bottom if cloud_bottom > 0 else 0  # Use absolute difference
                tk_spread_factor = (
                                               current_kijun - current_tenkan) / current_kijun if current_kijun > 0 else 0  # Absolute difference for strength
                chikou_dist_factor = (
                                                 price_kijun_periods_ago - current_close) / price_kijun_periods_ago if price_kijun_periods_ago > 0 else 0

                strength = min(1.0, max(0, (price_cloud_dist_factor * 0.3 + abs(
                    tk_spread_factor) * 0.3 + chikou_dist_factor * 0.2 + cloud_thickness_factor * 0.2)))

                # Generate TREND sell signal
                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_type'] = "TREND"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2


            # ----------------------------------------------------------------------
            # REVERSAL SIGNALS - Can be higher risk
            # Often involve TK cross inside or near the cloud, with Chikou confirmation
            # ----------------------------------------------------------------------

            # Bullish Reversal Signal Conditions:
            # 1. Bullish TK cross occurs
            # 2. Price is currently below or inside the Cloud
            # 3. Chikou Span (current close) crosses above or is near price from kijun periods ago
            # 4. Optional: Previous price action was bearish (e.g., below cloud) - implied by price < cloud_top
            elif (tk_bullish_cross and
                  current_close <= cloud_top and  # Price below or inside cloud
                  (chikou_above_price_signal or abs(current_close - price_kijun_periods_ago) < current_atr)
            # Chikou crossing or near price
            ):

                # Tighter stop loss below recent low or using ATR or Kijun
                stop_loss_option1 = current_close - current_atr * 1.5
                stop_loss_option2 = current_kijun
                stop_loss_option3 = cloud_bottom  # Cloud bottom can also act as support

                # Choose the highest of these levels that is still below current price
                stop_loss = current_close - current_atr * 1.5  # Start with ATR
                if stop_loss_option2 is not None and stop_loss_option2 < current_close and stop_loss_option2 > stop_loss:
                    stop_loss = stop_loss_option2
                if stop_loss_option3 is not None and stop_loss_option3 < current_close and stop_loss_option3 > stop_loss:
                    stop_loss = stop_loss_option3

                # Ensure stop loss is below current price for a buy and is not NaN
                if pd.isna(stop_loss) or stop_loss >= current_close:
                    stop_loss = current_close - current_atr * 1.5

                # Calculate take profit 1 at Cloud Top or 1.5x risk, whichever is further
                risk = current_close - stop_loss
                take_profit_1_option1 = cloud_top
                take_profit_1_option2 = current_close + risk * 1.5
                # Ensure cloud_top is not NaN if used
                if pd.isna(take_profit_1_option1):
                    take_profit_1 = take_profit_1_option2
                else:
                    take_profit_1 = max(take_profit_1_option1, take_profit_1_option2)

                # Calculate take profit 2 at 2:1 reward-to-risk
                take_profit_2 = current_close + risk * 2.0

                # Reversal trades have potentially lower initial strength than strong trend trades
                strength = min(0.7, abs((
                                                    current_tenkan - current_kijun) / current_kijun) * 5)  # Strength based on TK separation at cross
                if chikou_above_price_signal:
                    strength = min(1.0, strength * 1.2)  # Boost strength if Chikou confirms

                # Generate REVERSAL buy signal
                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_type'] = "REVERSAL"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2


            # Bearish Reversal Signal Conditions:
            # 1. Bearish TK cross occurs
            # 2. Price is currently above or inside the Cloud
            # 3. Chikou Span (current close) crosses below or is near price from kijun periods ago
            # 4. Optional: Previous price action was bullish (e.g., above cloud) - implied by price > cloud_bottom
            elif (tk_bearish_cross and
                  current_close >= cloud_bottom and  # Price above or inside cloud
                  (chikou_below_price_signal or abs(current_close - price_kijun_periods_ago) < current_atr)
            # Chikou crossing or near price
            ):

                # Tighter stop loss above recent high or using ATR or Kijun
                stop_loss_option1 = current_close + current_atr * 1.5
                stop_loss_option2 = current_kijun
                stop_loss_option3 = cloud_top  # Cloud top can also act as resistance

                # Choose the lowest of these levels that is still above current price
                stop_loss = current_close + current_atr * 1.5  # Start with ATR
                if stop_loss_option2 is not None and stop_loss_option2 > current_close and stop_loss_option2 < stop_loss:
                    stop_loss = stop_loss_option2
                if stop_loss_option3 is not None and stop_loss_option3 > current_close and stop_loss_option3 < stop_loss:
                    stop_loss = stop_loss_option3

                # Ensure stop loss is above current price for a sell and is not NaN
                if pd.isna(stop_loss) or stop_loss <= current_close:
                    stop_loss = current_close + current_atr * 1.5

                # Calculate take profit 1 at Cloud Bottom or 1.5x risk, whichever is further
                risk = stop_loss - current_close
                take_profit_1_option1 = cloud_bottom
                take_profit_1_option2 = current_close - risk * 1.5
                # Ensure cloud_bottom is not NaN if used
                if pd.isna(take_profit_1_option1):
                    take_profit_1 = take_profit_1_option2
                else:
                    take_profit_1 = min(take_profit_1_option1, take_profit_1_option2)

                # Calculate take profit 2 at 2:1 reward-to-risk
                take_profit_2 = current_close - risk * 2.0

                # Reversal trades have potentially lower initial strength than strong trend trades
                strength = min(0.7, abs((
                                                    current_kijun - current_tenkan) / current_kijun) * 5)  # Strength based on TK separation at cross
                if chikou_below_price_signal:
                    strength = min(1.0, strength * 1.2)  # Boost strength if Chikou confirms

                # Generate REVERSAL sell signal
                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_type'] = "REVERSAL"
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit_1'] = take_profit_1
                data.loc[data.index[i], 'take_profit_2'] = take_profit_2

            # If no signal conditions are met, ensure signal columns are reset for this bar
            else:
                data.loc[data.index[i], ['signal', 'signal_type', 'signal_strength', 'stop_loss', 'take_profit_1',
                                         'take_profit_2']] = [np.nan, None, np.nan, np.nan, np.nan, np.nan]

        return data

    def analyze(self, data):
        """Analyze market data and generate trading signals.

        Args:
            data (pandas.DataFrame): OHLC data for analysis

        Returns:
            list: Generated trading signals
        """
        # Calculate indicators and identify signals
        data_with_indicators = self.calculate_indicators(data)

        # Check if we have sufficient data after calculations or if calculation failed
        if data_with_indicators.empty or 'signal' not in data_with_indicators.columns:
            DBLogger.log_event("Warning", "Insufficient data for Ichimoku analysis or indicator calculation failed.",
                               "IchimokuStrategy")
            return []

        signals = []

        # Get the last complete candle where signals were potentially generated
        # Ensure the index is valid
        if data_with_indicators.empty:
            return []

        last_candle_index = len(data_with_indicators) - 1
        if last_candle_index < 0:  # Should not happen if data_with_indicators is not empty
            return []

        last_candle = data_with_indicators.iloc[last_candle_index]

        # Check for trading signal on the last candle
        # Check if signal is not NaN and not 0
        if not pd.isna(last_candle['signal']) and last_candle['signal'] != 0:
            signal_direction = "BUY" if last_candle['signal'] == 1 else "SELL"
            signal_type = last_candle['signal_type']
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit_1 = last_candle['take_profit_1']
            take_profit_2 = last_candle['take_profit_2']
            signal_strength = last_candle['signal_strength']

            # Basic validation for SL/TP values before creating signal
            # These checks are also in _identify_signals, but good to double-check
            is_valid = True
            if pd.isna(stop_loss) or pd.isna(take_profit_1) or pd.isna(take_profit_2):
                DBLogger.log_event("WARNING", f"Signal found but SL/TP values are NaN for {self.symbol}",
                                   "IchimokuStrategy")
                is_valid = False

            if signal_direction == "BUY" and (
                    stop_loss >= entry_price or take_profit_1 <= entry_price or take_profit_2 <= entry_price):
                DBLogger.log_event("WARNING",
                                   f"Invalid BUY SL/TP values for {self.symbol}: SL={stop_loss}, TP1={take_profit_1}, TP2={take_profit_2}",
                                   "IchimokuStrategy")
                is_valid = False
            elif signal_direction == "SELL" and (
                    stop_loss <= entry_price or take_profit_1 >= entry_price or take_profit_2 >= entry_price):
                DBLogger.log_event("WARNING",
                                   f"Invalid SELL SL/TP values for {self.symbol}: SL={stop_loss}, TP1={take_profit_1}, TP2={take_profit_2}",
                                   "IchimokuStrategy")
                is_valid = False

            # Also check if signal_type is not None or empty string if a signal exists
            if is_valid and (signal_type is None or signal_type == ""):
                DBLogger.log_event("WARNING", f"Signal found but signal_type is missing for {self.symbol}",
                                   "IchimokuStrategy")
                is_valid = False

            if is_valid:
                signal = self.create_signal(
                    signal_type=signal_direction,  # Use BUY/SELL for the signal type dictionary
                    price=entry_price,
                    strength=signal_strength,
                    metadata={
                        'stop_loss': stop_loss,
                        'take_profit_1': take_profit_1,
                        'take_profit_2': take_profit_2,
                        'tenkan_sen': last_candle['tenkan_sen'],
                        'kijun_sen': last_candle['kijun_sen'],
                        'senkou_span_a': last_candle['senkou_span_a'],
                        'senkou_span_b': last_candle['senkou_span_b'],
                        'cloud_bullish': bool(last_candle['cloud_bullish']),
                        'strategy_signal_type': signal_type,  # Store the TREND/REVERSAL type in metadata
                        'reason': f'{signal_direction} {signal_type} signal based on Ichimoku components.'
                    }
                )
                signals.append(signal)

                DBLogger.log_event("INFO",
                                   f"Generated {signal_direction} ({signal_type}) signal for {self.symbol} at {entry_price:.5f}. "
                                   f"Strength: {signal_strength:.2f}, "
                                   f"Stop: {stop_loss:.5f}, TP1: {take_profit_1:.5f}, TP2: {take_profit_2:.5f}",
                                   "IchimokuStrategy"
                                   )
            else:
                DBLogger.log_event("INFO",
                                   f"Skipped generating potential {signal_direction} signal for {self.symbol} due to validation failure.",
                                   "IchimokuStrategy")

        return signals
