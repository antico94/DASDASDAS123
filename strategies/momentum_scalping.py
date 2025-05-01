# strategies/momentum_scalping.py
from datetime import datetime

import numpy as np
import pandas as pd

from db_logger.db_logger import DBLogger
from strategies.base_strategy import BaseStrategy


class MomentumScalpingStrategy(BaseStrategy):
    """Enhanced Momentum Scalping Strategy for XAU/USD.

    This strategy focuses on capturing rapid price moves by trading in the direction
    of strong price momentum and exiting as soon as that momentum fades. It uses
    several momentum indicators (RSI, MACD, Stochastic, Momentum/ROC) and volume
    confirmation to trigger entries and exits.

    The strategy is optimized for short timeframes (M1, M5) and specifically
    adapted for XAU/USD (Gold) trading characteristics.
    """

    def __init__(self, symbol="XAUUSD", timeframe="M5",
                 rsi_period=14, rsi_threshold_high=60, rsi_threshold_low=40,
                 stoch_k_period=14, stoch_d_period=3, stoch_slowing=3,
                 macd_fast=12, macd_slow=26, macd_signal=9,
                 momentum_period=10, volume_threshold=1.5,
                 max_spread=3.0, consider_session=True,
                 data_fetcher=None):
        """Initialize the Enhanced Momentum Scalping strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "M5".
            rsi_period (int, optional): RSI period. Defaults to 14.
            rsi_threshold_high (int, optional): RSI threshold for bullish momentum. Defaults to 60.
            rsi_threshold_low (int, optional): RSI threshold for bearish momentum. Defaults to 40.
            stoch_k_period (int, optional): Stochastic %K period. Defaults to 14.
            stoch_d_period (int, optional): Stochastic %D period. Defaults to 3.
            stoch_slowing (int, optional): Stochastic slowing period. Defaults to 3.
            macd_fast (int, optional): MACD fast period. Defaults to 12.
            macd_slow (int, optional): MACD slow period. Defaults to 26.
            macd_signal (int, optional): MACD signal period. Defaults to 9.
            momentum_period (int, optional): Momentum/ROC period. Defaults to 10.
            volume_threshold (float, optional): Volume multiplier to confirm signals. Defaults to 1.5.
            max_spread (float, optional): Maximum allowed spread in pips. Defaults to 3.0.
            consider_session (bool, optional): Consider trading session. Defaults to True.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Momentum_Scalping", data_fetcher=data_fetcher)

        # Validate inputs
        if macd_fast >= macd_slow:
            error_msg = f"macd_fast ({macd_fast}) must be < macd_slow ({macd_slow})"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        if stoch_k_period <= 0 or stoch_d_period <= 0:
            error_msg = f"Stochastic periods must be positive"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        if rsi_threshold_high <= rsi_threshold_low:
            error_msg = f"RSI high threshold ({rsi_threshold_high}) must be > RSI low threshold ({rsi_threshold_low})"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        # Store parameters
        self.rsi_period = rsi_period
        self.rsi_threshold_high = rsi_threshold_high
        self.rsi_threshold_low = rsi_threshold_low

        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_slowing = stoch_slowing

        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.momentum_period = momentum_period
        self.volume_threshold = volume_threshold

        self.max_spread = max_spread
        self.consider_session = consider_session

        # Ensure we fetch enough data for calculations
        self.min_required_candles = max(
            rsi_period,
            stoch_k_period + stoch_d_period,
            macd_slow + macd_signal,
            momentum_period
        ) + 30  # Extra bars for better analysis

        DBLogger.log_event("INFO",
                           f"Initialized Enhanced Momentum Scalping strategy: {symbol} {timeframe}, "
                           f"RSI: {rsi_period}/{rsi_threshold_low}/{rsi_threshold_high}, "
                           f"Stoch: {stoch_k_period}/{stoch_d_period}/{stoch_slowing}, "
                           f"MACD: {macd_fast}/{macd_slow}/{macd_signal}, "
                           f"Momentum: {momentum_period}, Volume threshold: {volume_threshold}",
                           "MomentumScalpingStrategy")

    def _calculate_indicators(self, data):
        """Calculate strategy indicators on OHLC data."""
        if len(data) < self.min_required_candles:
            DBLogger.log_event("WARNING",
                               f"Insufficient data for momentum calculations. "
                               f"Need at least {self.min_required_candles} candles.",
                               "MomentumScalpingStrategy")
            return data

        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()

        # Calculate RSI
        df = self._calculate_rsi(df)

        # Calculate Stochastic Oscillator
        df = self._calculate_stochastic(df)

        # Calculate MACD
        df = self._calculate_macd(df)

        # Calculate Momentum/ROC
        df = self._calculate_momentum(df)

        # Calculate Average True Range (ATR) for volatility
        df = self._calculate_atr(df)

        # Calculate Volume analysis
        df = self._calculate_volume_metrics(df)

        # Calculate recent swing highs and lows for stop loss placement
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()

        # Identify if we're in a favorable trading session
        df = self._add_session_info(df)

        # Identify signal conditions
        df = self._identify_signals(df)

        return df

    def _calculate_rsi(self, data):
        """Calculate Relative Strength Index (RSI).

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with RSI indicator added
        """
        # Calculate price differences
        delta = data['close'].diff()

        # Get gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gain and loss over RSI period
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        # Calculate RS
        rs = avg_gain / avg_loss

        # Calculate RSI
        data['rsi'] = 100 - (100 / (1 + rs))

        # Calculate short-term RSI direction (3-bar)
        data['rsi_direction'] = data['rsi'].diff(3)

        return data

    def _calculate_stochastic(self, data):
        """Calculate Stochastic Oscillator."""
        # Calculate %K (Fast Stochastic)
        low_min = data['low'].rolling(window=self.stoch_k_period).min()
        high_max = data['high'].rolling(window=self.stoch_k_period).max()

        # Handle potential division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)

        data['stoch_k_raw'] = ((data['close'] - low_min) / denominator) * 100

        # Apply slowing if specified (typically 3)
        if self.stoch_slowing > 1:
            data['stoch_k'] = data['stoch_k_raw'].rolling(window=self.stoch_slowing).mean()
        else:
            data['stoch_k'] = data['stoch_k_raw']

        # Calculate %D (Slow Stochastic - signal line)
        data['stoch_d'] = data['stoch_k'].rolling(window=self.stoch_d_period).mean()

        # Calculate if K is above D (bullish) or below D (bearish)
        # Correct: Use numpy for array-like operations
        data['stoch_k_above_d'] = np.where(data['stoch_k'] > data['stoch_d'], 1, 0)

        # Calculate if we're in overbought/oversold territories
        data['stoch_overbought'] = np.where(data['stoch_k'] > 80, 1, 0)
        data['stoch_oversold'] = np.where(data['stoch_k'] < 20, 1, 0)

        # Stochastic crossovers
        stoch_bull_cross = ((data['stoch_k'] > data['stoch_d']) &
                            (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)))
        data['stoch_bull_cross'] = np.where(stoch_bull_cross, 1, 0)

        stoch_bear_cross = ((data['stoch_k'] < data['stoch_d']) &
                            (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)))
        data['stoch_bear_cross'] = np.where(stoch_bear_cross, 1, 0)

        return data

    def _calculate_macd(self, data):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        ema_fast = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow, adjust=False).mean()

        # Calculate MACD line and signal
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.macd_signal, adjust=False).mean()

        # Calculate MACD histogram
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Calculate MACD momentum (histogram direction)
        data['macd_histogram_direction'] = data['macd_histogram'].diff(1)

        # Calculate MACD crosses - fix astype error
        macd_bull_cross = ((data['macd'] > data['macd_signal']) &
                           (data['macd'].shift(1) <= data['macd_signal'].shift(1)))
        data['macd_bull_cross'] = np.where(macd_bull_cross, 1, 0)

        macd_bear_cross = ((data['macd'] < data['macd_signal']) &
                           (data['macd'].shift(1) >= data['macd_signal'].shift(1)))
        data['macd_bear_cross'] = np.where(macd_bear_cross, 1, 0)

        return data

    def _calculate_momentum(self, data):
        """Calculate Momentum/Rate of Change (ROC) indicator.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with Momentum indicator added
        """
        # Calculate Momentum as percentage change over period
        # Formula: (Current Price / Price N periods ago) * 100
        data['momentum'] = (data['close'] / data['close'].shift(self.momentum_period)) * 100

        # Calculate if momentum is positive (>100) or negative (<100)
        data['momentum_positive'] = (data['momentum'] > 100).astype(int)
        data['momentum_negative'] = (data['momentum'] < 100).astype(int)

        # Calculate momentum change to detect acceleration/deceleration
        data['momentum_direction'] = data['momentum'].diff(1)

        return data

    def _calculate_atr(self, data):
        """Calculate Average True Range (ATR).

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with ATR indicator added
        """
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        return data

    def _calculate_volume_metrics(self, data):
        """Calculate volume-based indicators.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with volume metrics added
        """
        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(window=20).mean()

        # Calculate relative volume (current volume vs average)
        # Initialize with zeros to avoid NaN issues
        data['volume_ratio'] = 0.0

        # Only calculate ratio where we have valid moving averages
        valid_indices = data.index[~data['volume_ma'].isna() & (data['volume_ma'] > 0)]
        if len(valid_indices) > 0:
            data.loc[valid_indices, 'volume_ratio'] = data.loc[valid_indices, 'volume'] / data.loc[
                valid_indices, 'volume_ma']

        # Flag high volume bars - using numpy for consistent array operations
        # Only mark high volume where we have valid ratios
        data['high_volume'] = 0
        valid_ratio_indices = data.index[data['volume_ratio'] > 0]
        if len(valid_ratio_indices) > 0:
            data.loc[valid_ratio_indices, 'high_volume'] = np.where(
                data.loc[valid_ratio_indices, 'volume_ratio'] >= self.volume_threshold,
                1,
                0
            )

        # Calculate volume change
        data['volume_change'] = data['volume'].pct_change()
        # Fill NaN in first row of volume_change with 0
        data['volume_change'] = data['volume_change'].fillna(0)

        # Log warning when insufficient data is available
        invalid_count = len(data) - len(valid_indices)
        if invalid_count > 0:
            DBLogger.log_event(
                "WARNING",
                f"Insufficient volume data for {invalid_count} of {len(data)} bars. Some volume metrics unavailable.",
                "MomentumScalpingStrategy"
            )

        return data

    def _add_session_info(self, data):
        """Add trading session information.

        Gold typically has the best momentum during London/NY overlap.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with session info added
        """
        if not self.consider_session or 'timestamp' not in data.columns:
            # Create a default column to avoid errors
            data['good_session'] = 1
            return data

        # Extract hour from timestamp
        hours = pd.Series([t.hour if hasattr(t, 'hour') else t.to_pydatetime().hour
                           for t in data.index])

        # London/NY overlap (13-17 UTC is optimal) - fix astype error
        data['good_session'] = np.where((hours >= 13) & (hours < 17), 1, 0)

        # Asian session (lower liquidity, avoid) - fix astype error
        data['low_liquidity_session'] = np.where((hours >= 0) & (hours < 6), 1, 0)

        return data

    def _identify_signals(self, data):
        """Identify momentum scalping signals according to the plan specifications.

        This method implements a high-precision momentum scalping strategy for XAU/USD,
        focusing on capturing rapid price moves by trading in the direction of strong
        price momentum and exiting as soon as that momentum fades.

        Args:
            data (pandas.DataFrame): OHLC data with required indicators

        Returns:
            pandas.DataFrame: Data with signal columns added

        Raises:
            ValueError: If required data is missing or conditions can't be evaluated
            RuntimeError: If critical signal setting operations fail
        """
        # Validate input data
        if data is None:
            error_msg = "Input data cannot be None"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        if not isinstance(data, pd.DataFrame):
            error_msg = f"Input data must be a pandas DataFrame, got {type(data)}"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        if len(data) < 5:
            error_msg = f"Input data must have at least 5 rows, got {len(data)}"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        # Verify required indicator columns exist
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'momentum', 'volume_ratio'
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg)
            raise ValueError(error_msg)

        # Make a copy to avoid modifying the original
        try:
            processed_data = data.copy()
        except Exception as e:
            error_msg = f"Failed to create data copy: {str(e)}"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg, exception=e)
            raise ValueError(error_msg) from e

        # Initialize signal columns or reset them if they exist
        try:
            processed_data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
            processed_data['signal_strength'] = 0.0
            processed_data['stop_loss'] = np.nan
            processed_data['take_profit'] = np.nan
            processed_data['momentum_state'] = 0  # 0: Neutral, 1: Bullish, -1: Bearish
            processed_data['momentum_fading'] = 0  # 0: Not fading, 1: Bullish fading, -1: Bearish fading
        except Exception as e:
            error_msg = f"Failed to initialize signal columns: {str(e)}"
            DBLogger.log_error("MomentumScalpingStrategy", error_msg, exception=e)
            raise ValueError(error_msg) from e

        DBLogger.log_event("DEBUG", f"Processing {len(processed_data)} bars for signal identification",
                           "MomentumScalpingStrategy")

        # Calculate the minimum required bar for valid indicators
        # For a 20-period volume MA, we need at least 20 bars
        min_bar_required = 20

        # Process each bar starting from the min_bar_required (to have enough lookback)
        bar_count = 0
        signal_count = 0

        for i in range(min_bar_required, len(processed_data)):
            try:
                bar_count += 1

                # Get current values
                current_close = processed_data.iloc[i]['close']
                current_high = processed_data.iloc[i]['high']
                current_low = processed_data.iloc[i]['low']
                current_open = processed_data.iloc[i]['open']

                # Previous values for comparison - crucial for identifying momentum patterns
                prev_close = processed_data.iloc[i - 1]['close']
                prev_high = processed_data.iloc[i - 1]['high']
                prev_low = processed_data.iloc[i - 1]['low']
                prev_open = processed_data.iloc[i - 1]['open']

                # Calculate price change metrics
                price_change = current_close - prev_close
                price_change_percent = (price_change / prev_close) * 100

                # Calculate candle patterns for additional confirmation
                current_candle_body = abs(current_close - current_open)
                prev_candle_body = abs(prev_close - prev_open)

                # Get indicator values with explicit error checking
                try:
                    current_rsi = processed_data.iloc[i]['rsi']
                    current_macd = processed_data.iloc[i]['macd']
                    current_macd_signal = processed_data.iloc[i]['macd_signal']
                    current_macd_hist = processed_data.iloc[i]['macd_histogram']
                    prev_macd_hist = processed_data.iloc[i - 1]['macd_histogram']
                    current_stoch_k = processed_data.iloc[i]['stoch_k']
                    current_stoch_d = processed_data.iloc[i]['stoch_d']
                    prev_stoch_k = processed_data.iloc[i - 1]['stoch_k']
                    prev_stoch_d = processed_data.iloc[i - 1]['stoch_d']
                    current_momentum = processed_data.iloc[i]['momentum']
                    current_volume_ratio = processed_data.iloc[i]['volume_ratio']
                except KeyError as e:
                    error_msg = f"Failed to access indicator column: {str(e)}"
                    DBLogger.log_error("MomentumScalpingStrategy", error_msg, exception=e)
                    continue  # Skip this bar but continue processing others

                # Check for invalid indicator values (common in early bars)
                if (pd.isna(current_rsi) or pd.isna(current_macd_hist) or pd.isna(prev_macd_hist) or
                        pd.isna(current_stoch_k) or pd.isna(current_momentum)):
                    DBLogger.log_event("DEBUG", f"Skipping bar {i} due to NaN indicator values",
                                       "MomentumScalpingStrategy")
                    continue

                # Check if session consideration is enabled
                good_session = True  # Default to good session
                if self.consider_session:
                    if 'good_session' in processed_data.columns:
                        good_session = bool(processed_data.iloc[i]['good_session'] == 1)
                    else:
                        # If column is missing but feature is enabled, create it
                        # from index timestamp if available
                        try:
                            if hasattr(processed_data.index, 'hour') or hasattr(processed_data.index[i], 'hour'):
                                hour = processed_data.index[i].hour if hasattr(processed_data.index[i], 'hour') else 0
                                good_session = 13 <= hour < 17  # London/NY overlap

                                # If we're creating this on the fly, add it to the dataframe
                                if 'good_session' not in processed_data.columns:
                                    processed_data['good_session'] = 0
                                    # Set session quality for all bars based on hour
                                    for idx, timestamp in enumerate(processed_data.index):
                                        if hasattr(timestamp, 'hour'):
                                            hour_val = timestamp.hour
                                            processed_data.iloc[idx, processed_data.columns.get_loc(
                                                'good_session')] = 1 if 13 <= hour_val < 17 else 0
                        except Exception as e:
                            DBLogger.log_event("WARNING", f"Failed to determine session quality: {str(e)}",
                                               "MomentumScalpingStrategy")
                            # Continue with default good_session=True

                # PRICE ACTION TRIGGER: Check for breakout or strong momentum candle
                try:
                    # Look back for recent high/low levels
                    lookback = 20  # Look back 20 bars for significant levels
                    lookback_start = max(0, i - lookback)
                    recent_high = processed_data.iloc[lookback_start:i]['high'].max()  # Exclude current bar
                    recent_low = processed_data.iloc[lookback_start:i]['low'].min()  # Exclude current bar

                    # Calculate average candle size for comparison
                    avg_candle_size = (
                            processed_data.iloc[lookback_start:i]['high'] - processed_data.iloc[lookback_start:i][
                        'low']).mean()
                    current_candle_size = current_high - current_low

                    # Check if current candle is significantly larger (breakout condition)
                    large_candle = current_candle_size > (avg_candle_size * 1.5)

                    # Check for breakout above recent high or below recent low
                    # Using previous values as reference points for breakout confirmation
                    breakout_up = current_close > recent_high and current_close > prev_high
                    breakout_down = current_close < recent_low and current_close < prev_low

                    # Check for strong momentum candles - using previous price for comparison
                    strong_bullish_candle = (
                            current_close > prev_close and
                            price_change_percent > 0.2 and  # 0.2% move is significant for XAU/USD
                            current_candle_body > prev_candle_body * 1.5
                        # Current candle body is 50% larger than previous
                    )

                    strong_bearish_candle = (
                            current_close < prev_close and
                            price_change_percent < -0.2 and  # -0.2% move is significant for XAU/USD
                            current_candle_body > prev_candle_body * 1.5
                        # Current candle body is 50% larger than previous
                    )

                    # Enhanced price action conditions
                    bullish_price_action = breakout_up or large_candle or strong_bullish_candle
                    bearish_price_action = breakout_down or large_candle or strong_bearish_candle

                    # Log price action for debugging
                    if bullish_price_action or bearish_price_action:
                        action_type = []
                        if large_candle:
                            action_type.append("large candle")
                        if breakout_up:
                            action_type.append("upward breakout")
                        if breakout_down:
                            action_type.append("downward breakout")
                        if strong_bullish_candle:
                            action_type.append("strong bullish candle")
                        if strong_bearish_candle:
                            action_type.append("strong bearish candle")

                        DBLogger.log_event("DEBUG",
                                           f"Price action detected at bar {i}: {', '.join(action_type)}. "
                                           f"Close: {current_close}, Prev close: {prev_close}, "
                                           f"Recent high: {recent_high}, Recent low: {recent_low}, "
                                           f"Change %: {price_change_percent:.2f}%, "
                                           f"Candle size: {current_candle_size:.2f}, Avg size: {avg_candle_size:.2f}",
                                           "MomentumScalpingStrategy")
                except Exception as e:
                    DBLogger.log_error("MomentumScalpingStrategy",
                                       f"Error calculating price action at bar {i}: {str(e)}",
                                       exception=e)
                    continue  # Skip this bar if price action can't be determined

                # MOMENTUM INDICATOR CONFIRMATION
                # Define conditions with explicit error handling
                try:
                    # RSI condition with trend confirmation
                    rsi_rising = current_rsi > processed_data.iloc[i - 2]['rsi']  # Compare with 2 bars ago for trend
                    rsi_falling = current_rsi < processed_data.iloc[i - 2]['rsi']
                    bullish_rsi = current_rsi > self.rsi_threshold_high and rsi_rising
                    bearish_rsi = current_rsi < self.rsi_threshold_low and rsi_falling

                    # MACD confirmation with trend confirmation
                    macd_rising = current_macd_hist > prev_macd_hist
                    macd_falling = current_macd_hist < prev_macd_hist
                    bullish_macd = current_macd_hist > 0 and current_macd > current_macd_signal and macd_rising
                    bearish_macd = current_macd_hist < 0 and current_macd < current_macd_signal and macd_falling

                    # Stochastic confirmation
                    stoch_bullish_cross = (current_stoch_k > current_stoch_d) and (prev_stoch_k <= prev_stoch_d)
                    stoch_bearish_cross = (current_stoch_k < current_stoch_d) and (prev_stoch_k >= prev_stoch_d)
                    stoch_bullish = current_stoch_k > current_stoch_d or current_stoch_k > 80
                    stoch_bearish = current_stoch_k < current_stoch_d or current_stoch_k < 20

                    # Momentum/ROC with confirmation from price
                    bullish_momentum = current_momentum > 100.2 and current_close > prev_close
                    bearish_momentum = current_momentum < 99.8 and current_close < prev_close

                    # Volume confirmation with surge detection - with proper handling for zeros
                    # A zero value means invalid/missing data
                    if current_volume_ratio <= 0:
                        volume_surge = False
                        high_volume = False
                        DBLogger.log_event("DEBUG",
                                           f"Bar {i}: Invalid volume ratio data - can't evaluate volume conditions",
                                           "MomentumScalpingStrategy")
                    else:
                        # Check previous volume ratio for surge calculation
                        prev_volume_ratio = processed_data.iloc[i - 1]['volume_ratio']
                        if prev_volume_ratio <= 0:
                            volume_surge = False  # Can't calculate surge without previous data
                        else:
                            volume_surge = current_volume_ratio > prev_volume_ratio * 1.2  # 20% volume increase

                        high_volume = current_volume_ratio >= self.volume_threshold and volume_surge

                    # Log indicator conditions for debugging the last few bars
                    if i >= len(processed_data) - 3:  # Only log last 3 bars for efficiency
                        DBLogger.log_event("DEBUG",
                                           f"Bar {i} conditions: RSI={current_rsi:.1f}({bullish_rsi}), "
                                           f"MACD Hist={current_macd_hist:.4f}({bullish_macd}), "
                                           f"Stoch K/D={current_stoch_k:.1f}/{current_stoch_d:.1f}({stoch_bullish}), "
                                           f"Momentum={current_momentum:.2f}({bullish_momentum}), "
                                           f"Volume ratio={current_volume_ratio:.1f}({high_volume}), "
                                           f"Session={good_session}, "
                                           f"Price action: BP={bullish_price_action}, BP={bearish_price_action}",
                                           "MomentumScalpingStrategy")
                except Exception as e:
                    DBLogger.log_error("MomentumScalpingStrategy",
                                       f"Error evaluating indicator conditions at bar {i}: {str(e)}",
                                       exception=e)
                    continue  # Skip this bar if conditions can't be evaluated

                # COMBINED ENTRY CONDITIONS
                # For a long entry: RSI>60, MACD>0, Stochastic bullish, Volume high, Price breakout up
                bullish_conditions = (
                        bullish_rsi and
                        bullish_macd and
                        (stoch_bullish or stoch_bullish_cross) and  # Using stochastic crossover signal
                        bullish_momentum and
                        high_volume and
                        bullish_price_action and
                        good_session
                )

                # For a short entry: RSI<40, MACD<0, Stochastic bearish, Volume high, Price breakout down
                bearish_conditions = (
                        bearish_rsi and
                        bearish_macd and
                        (stoch_bearish or stoch_bearish_cross) and  # Using stochastic crossover signal
                        bearish_momentum and
                        high_volume and
                        bearish_price_action and
                        good_session
                )

                # ---------- ENHANCED DEBUGGING START ----------
                # Create detailed condition check logging for every bar
                # Format: "Condition ✅/❌ (Actual Value)"

                # Prepare condition checklist for bullish signals
                bullish_checks = [
                    f"RSI > {self.rsi_threshold_high} and rising: {'✅' if bullish_rsi else '❌'} ({current_rsi:.1f}, {'rising' if rsi_rising else 'not rising'})",
                    f"MACD Histogram > 0 and rising: {'✅' if bullish_macd else '❌'} ({current_macd_hist:.4f}, {'rising' if macd_rising else 'not rising'})",
                    f"Stochastic bullish: {'✅' if stoch_bullish or stoch_bullish_cross else '❌'} (K:{current_stoch_k:.1f}, D:{current_stoch_d:.1f}, Cross:{stoch_bullish_cross})",
                    f"Momentum > 100.2: {'✅' if bullish_momentum else '❌'} ({current_momentum:.2f})",
                    f"Volume high & surging: {'✅' if high_volume else '❌'} (Ratio:{current_volume_ratio:.2f}, Surge:{volume_surge})",
                    f"Bullish price action: {'✅' if bullish_price_action else '❌'} (Breakout:{breakout_up}, Large:{large_candle}, Strong:{strong_bullish_candle})",
                    f"Good session: {'✅' if good_session else '❌'}"
                ]

                # Prepare condition checklist for bearish signals
                bearish_checks = [
                    f"RSI < {self.rsi_threshold_low} and falling: {'✅' if bearish_rsi else '❌'} ({current_rsi:.1f}, {'falling' if rsi_falling else 'not falling'})",
                    f"MACD Histogram < 0 and falling: {'✅' if bearish_macd else '❌'} ({current_macd_hist:.4f}, {'falling' if macd_falling else 'not falling'})",
                    f"Stochastic bearish: {'✅' if stoch_bearish or stoch_bearish_cross else '❌'} (K:{current_stoch_k:.1f}, D:{current_stoch_d:.1f}, Cross:{stoch_bearish_cross})",
                    f"Momentum < 99.8: {'✅' if bearish_momentum else '❌'} ({current_momentum:.2f})",
                    f"Volume high & surging: {'✅' if high_volume else '❌'} (Ratio:{current_volume_ratio:.2f}, Surge:{volume_surge})",
                    f"Bearish price action: {'✅' if bearish_price_action else '❌'} (Breakout:{breakout_down}, Large:{large_candle}, Strong:{strong_bearish_candle})",
                    f"Good session: {'✅' if good_session else '❌'}"
                ]

                # After your bullish_checks and bearish_checks definitions...

                # Print Bullish Checks
                print("\n=== Momentum Scalping - Bullish Checks ===")
                for idx, check in enumerate(bullish_checks, start=1):
                    print(f"{idx}. {check}")

                # Print Bearish Checks
                print("\n=== Momentum Scalping - Bearish Checks ===")
                for idx, check in enumerate(bearish_checks, start=1):
                    print(f"{idx}. {check}")

                # Log detailed conditions for every bar being analyzed
                DBLogger.log_event("DEBUG",
                                   f"Signal conditions at bar {i} (Time: {processed_data.index[i] if hasattr(processed_data, 'index') else 'N/A'}) - Price: {current_close:.2f}\n"
                                   f"BULLISH CHECKS:\n" + "\n".join(bullish_checks) + "\n"
                                                                                      f"BEARISH CHECKS:\n" + "\n".join(
                                       bearish_checks) + "\n"
                                                         f"FINAL RESULT: {'BUY SIGNAL' if bullish_conditions else 'SELL SIGNAL' if bearish_conditions else 'NO SIGNAL'}",
                                   "MomentumScalpingStrategy")
                # ---------- ENHANCED DEBUGGING END ----------

                # Debug log bullish conditions for the last few bars
                if i >= len(processed_data) - 3:  # Only log last 3 bars
                    DBLogger.log_event("DEBUG",
                                       f"Bar {i} bullish_conditions={bullish_conditions}, bearish_conditions={bearish_conditions}",
                                       "MomentumScalpingStrategy")

                # Check for BUY signal
                if bullish_conditions:
                    signal_count += 1
                    DBLogger.log_event("INFO",
                                       f"BUY SIGNAL triggered at bar {i}. "
                                       f"Close: {current_close}, RSI: {current_rsi:.1f}, "
                                       f"MACD Hist: {current_macd_hist:.4f}, Momentum: {current_momentum:.2f}",
                                       "MomentumScalpingStrategy")

                    # Calculate signal strength based on multiple factors
                    try:
                        strength_factors = [
                            min(1.0, max(0, (current_rsi - 50) / 25)),  # Scale 0-1 based on RSI strength
                            min(1.0, max(0, current_macd_hist / processed_data.iloc[i][
                                'atr'] if 'atr' in processed_data.columns and processed_data.iloc[i][
                                'atr'] > 0 else 0.5)),  # Scale relative to volatility
                            min(1.0, max(0, (current_stoch_k - 50) / 50)),  # Scale based on stochastic
                            min(1.0, max(0, (current_volume_ratio - 1) / self.volume_threshold))  # Volume strength
                        ]

                        # Add extra strength for stochastic crossover, especially from oversold
                        if stoch_bullish_cross:
                            crossover_from_oversold = prev_stoch_k < 20
                            strength_factors.append(0.2 if crossover_from_oversold else 0.1)

                        # Average the factors
                        signal_strength = min(1.0, sum(strength_factors) / len(strength_factors))

                        DBLogger.log_event("DEBUG",
                                           f"Signal strength factors: {strength_factors}, "
                                           f"Final strength: {signal_strength:.2f}",
                                           "MomentumScalpingStrategy")
                    except Exception as e:
                        DBLogger.log_error("MomentumScalpingStrategy",
                                           f"Error calculating signal strength: {str(e)}",
                                           exception=e)
                        signal_strength = 0.6  # Default if calculation fails

                    # Calculate stop loss using ATR if available
                    try:
                        if 'atr' in processed_data.columns and not pd.isna(processed_data.iloc[i]['atr']):
                            atr = processed_data.iloc[i]['atr']
                        else:
                            # Estimate ATR if not available
                            atr = (current_high - current_low) * 0.5

                        stop_loss = current_close - (atr * 1.5)

                        # Calculate take profit (1:1 risk-reward for now)
                        risk = current_close - stop_loss
                        take_profit = current_close + risk
                    except Exception as e:
                        DBLogger.log_error("MomentumScalpingStrategy",
                                           f"Error calculating stop loss/take profit: {str(e)}",
                                           exception=e)
                        # Use default values as fallback
                        stop_loss = current_close * 0.995
                        take_profit = current_close * 1.005

                    # Set signal values - with explicit error handling
                    try:
                        processed_data.iloc[i, processed_data.columns.get_loc('signal')] = 1  # Buy signal
                        processed_data.iloc[i, processed_data.columns.get_loc('signal_strength')] = signal_strength
                        processed_data.iloc[i, processed_data.columns.get_loc('stop_loss')] = stop_loss
                        processed_data.iloc[i, processed_data.columns.get_loc('take_profit')] = take_profit
                        processed_data.iloc[i, processed_data.columns.get_loc('momentum_state')] = 1  # Bullish state
                    except Exception as e:
                        critical_error = f"CRITICAL: Failed to set signal values in DataFrame at index {i}: {str(e)}"
                        DBLogger.log_error("MomentumScalpingStrategy", critical_error, exception=e)
                        raise RuntimeError(critical_error) from e

                # Check for SELL signal
                elif bearish_conditions:
                    signal_count += 1
                    DBLogger.log_event("INFO",
                                       f"SELL SIGNAL triggered at bar {i}. "
                                       f"Close: {current_close}, RSI: {current_rsi:.1f}, "
                                       f"MACD Hist: {current_macd_hist:.4f}, Momentum: {current_momentum:.2f}",
                                       "MomentumScalpingStrategy")

                    # Calculate signal strength based on multiple factors
                    try:
                        strength_factors = [
                            min(1.0, max(0, (50 - current_rsi) / 25)),  # Scale 0-1 based on RSI strength
                            min(1.0, max(0, -current_macd_hist / processed_data.iloc[i][
                                'atr'] if 'atr' in processed_data.columns and processed_data.iloc[i][
                                'atr'] > 0 else 0.5)),  # Scale relative to volatility
                            min(1.0, max(0, (50 - current_stoch_k) / 50)),  # Scale based on stochastic
                            min(1.0, max(0, (current_volume_ratio - 1) / self.volume_threshold))  # Volume strength
                        ]

                        # Add extra strength for stochastic crossover, especially from overbought
                        if stoch_bearish_cross:
                            crossover_from_overbought = prev_stoch_k > 80
                            strength_factors.append(0.2 if crossover_from_overbought else 0.1)

                        # Average the factors
                        signal_strength = min(1.0, sum(strength_factors) / len(strength_factors))

                        DBLogger.log_event("DEBUG",
                                           f"Signal strength factors: {strength_factors}, "
                                           f"Final strength: {signal_strength:.2f}",
                                           "MomentumScalpingStrategy")
                    except Exception as e:
                        DBLogger.log_error("MomentumScalpingStrategy",
                                           f"Error calculating signal strength: {str(e)}",
                                           exception=e)
                        signal_strength = 0.6  # Default if calculation fails

                    # Calculate stop loss using ATR if available
                    try:
                        if 'atr' in processed_data.columns and not pd.isna(processed_data.iloc[i]['atr']):
                            atr = processed_data.iloc[i]['atr']
                        else:
                            # Estimate ATR if not available
                            atr = (current_high - current_low) * 0.5

                        stop_loss = current_close + (atr * 1.5)

                        # Calculate take profit (1:1 risk-reward for now)
                        risk = stop_loss - current_close
                        take_profit = current_close - risk
                    except Exception as e:
                        DBLogger.log_error("MomentumScalpingStrategy",
                                           f"Error calculating stop loss/take profit: {str(e)}",
                                           exception=e)
                        # Use default values as fallback
                        stop_loss = current_close * 1.005
                        take_profit = current_close * 0.995

                    # Set signal values - with explicit error handling
                    try:
                        processed_data.iloc[i, processed_data.columns.get_loc('signal')] = -1  # Sell signal
                        processed_data.iloc[i, processed_data.columns.get_loc('signal_strength')] = signal_strength
                        processed_data.iloc[i, processed_data.columns.get_loc('stop_loss')] = stop_loss
                        processed_data.iloc[i, processed_data.columns.get_loc('take_profit')] = take_profit
                        processed_data.iloc[i, processed_data.columns.get_loc('momentum_state')] = -1  # Bearish state
                    except Exception as e:
                        critical_error = f"CRITICAL: Failed to set signal values in DataFrame at index {i}: {str(e)}"
                        DBLogger.log_error("MomentumScalpingStrategy", critical_error, exception=e)
                        raise RuntimeError(critical_error) from e

                # MOMENTUM FADING DETECTION FOR EXITS
                # Per plan: "exit when RSI drops back below 50, MACD histogram shrinks, etc."
                try:
                    if i > 0 and processed_data.iloc[i - 1]['momentum_state'] == 1:  # Previous bar was bullish
                        # Check if momentum is fading
                        momentum_fading = (
                                current_rsi < 50 or
                                current_macd_hist < prev_macd_hist or  # Histogram shrinking
                                (current_stoch_k < current_stoch_d and prev_stoch_k >= prev_stoch_d)
                            # Bearish stoch cross
                        )

                        if momentum_fading:
                            DBLogger.log_event("DEBUG",
                                               f"Bullish momentum fading detected at bar {i}. "
                                               f"RSI: {current_rsi:.1f}, MACD hist: {current_macd_hist:.4f}",
                                               "MomentumScalpingStrategy")
                            processed_data.iloc[
                                i, processed_data.columns.get_loc('momentum_fading')] = 1  # Bullish momentum fading

                    elif i > 0 and processed_data.iloc[i - 1]['momentum_state'] == -1:  # Previous bar was bearish
                        momentum_fading = (
                                current_rsi > 50 or
                                current_macd_hist > prev_macd_hist or  # Histogram shrinking (in negative)
                                (current_stoch_k > current_stoch_d and prev_stoch_k <= prev_stoch_d)
                            # Bullish stoch cross
                        )

                        if momentum_fading:
                            DBLogger.log_event("DEBUG",
                                               f"Bearish momentum fading detected at bar {i}. "
                                               f"RSI: {current_rsi:.1f}, MACD hist: {current_macd_hist:.4f}",
                                               "MomentumScalpingStrategy")
                            processed_data.iloc[
                                i, processed_data.columns.get_loc('momentum_fading')] = -1  # Bearish momentum fading
                except Exception as e:
                    DBLogger.log_error("MomentumScalpingStrategy",
                                       f"Error detecting momentum fading at bar {i}: {str(e)}",
                                       exception=e)
                    # Continue - momentum fading detection is not critical
            except Exception as e:
                DBLogger.log_error("MomentumScalpingStrategy",
                                   f"Error processing bar {i}: {str(e)}",
                                   exception=e)
                # Continue to next bar

        # Log signal processing summary
        DBLogger.log_event("INFO",
                           f"Signal identification complete. Processed {bar_count} bars, found {signal_count} signals.",
                           "MomentumScalpingStrategy")

        # Verify after processing
        if signal_count > 0:
            signal_bars = processed_data[processed_data['signal'] != 0]
            if len(signal_bars) != signal_count:
                DBLogger.log_event("WARNING",
                                   f"Signal count mismatch. Expected {signal_count}, got {len(signal_bars)}.",
                                   "MomentumScalpingStrategy")

        return processed_data

    def analyze(self, data):
        """Analyze market data and generate trading signals.

        Args:
            data (pandas.DataFrame): OHLC data for analysis

        Returns:
            list: Generated trading signals
        """
        # Check if data is None or empty
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            DBLogger.log_event("WARNING", "No data provided for momentum scalping analysis", "MomentumScalpingStrategy")
            return []

        # Calculate indicators
        try:
            data = self._calculate_indicators(data)
        except Exception as e:
            DBLogger.log_error("MomentumScalpingStrategy", "Error calculating indicators", exception=e)
            return []

        # Check if we have sufficient data after calculations
        if data.empty or 'signal' not in data.columns:
            DBLogger.log_event("WARNING", "Insufficient data for momentum scalping analysis after calculations",
                               "MomentumScalpingStrategy")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]

        # Check current spread if available
        if hasattr(self.data_fetcher, 'connector') and self.data_fetcher.connector is not None:
            try:
                symbol_info = self.data_fetcher.connector.get_symbol_info(self.symbol)
                if symbol_info and 'ask' in symbol_info and 'bid' in symbol_info:
                    current_spread = symbol_info['ask'] - symbol_info['bid']
                    # For XAU/USD, spread is typically in dollars
                    # Convert to pips (1 pip = $0.1 for gold)
                    spread_pips = current_spread * 10

                    # Skip if spread is too wide
                    if spread_pips > self.max_spread:
                        DBLogger.log_event("WARNING",
                                           f"Current spread ({spread_pips:.1f} pips) exceeds maximum allowed ({self.max_spread:.1f} pips)",
                                           "MomentumScalpingStrategy")
                        return []
            except Exception as e:
                # Log but don't fail if spread check has an error
                DBLogger.log_event("WARNING",
                                   f"Error checking spread, continuing analysis: {str(e)}",
                                   "MomentumScalpingStrategy")

        # Check if we're in a good session time (if session analysis is enabled)
        session_ok = True
        if self.consider_session and 'good_session' in data.columns:
            if last_candle['good_session'] != 1:
                session_ok = False
                current_time = datetime.now().time()
                DBLogger.log_event("DEBUG",
                                   f"Current time {current_time} is not in optimal trading session for momentum scalping",
                                   "MomentumScalpingStrategy")
                # We'll still continue with analysis but note the suboptimal session

        # Debug log the signal value
        DBLogger.log_event("DEBUG", f"Signal in last candle: {last_candle['signal']}", "MomentumScalpingStrategy")

        # Check for trading signal on the last candle
        if last_candle['signal'] == 1:  # Buy signal
            # Create BUY signal
            entry_price = last_candle['close']

            # FIXED: Calculate stop loss directly here if needed, don't rely on potentially stale value
            # First, we check if the stop_loss value is valid (not 0.0 or NaN)
            stop_loss = last_candle['stop_loss'] if 'stop_loss' in last_candle else None
            if stop_loss is None or pd.isna(stop_loss) or stop_loss == 0 or stop_loss >= entry_price:
                # Recalculate using ATR
                atr = last_candle['atr'] if 'atr' in last_candle else (last_candle['high'] - last_candle['low']) * 0.1
                stop_loss = entry_price - (atr * 1.5)  # Default 1.5 * ATR below entry

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate multiple take profit levels for scaling out
            risk = entry_price - stop_loss
            take_profit_1r = entry_price + risk  # 1:1 reward-to-risk
            take_profit_2r = entry_price + (risk * 2)  # 2:1 reward-to-risk

            # Get signal strength or default to 0.6 if not available
            signal_strength = last_candle['signal_strength'] if 'signal_strength' in last_candle else 0.6

            # Prepare metadata with safe fallbacks
            metadata = {
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'risk_amount': risk,
                'rsi': last_candle.get('rsi', 0),
                'macd_histogram': last_candle.get('macd_histogram', 0),
                'stoch_k': last_candle.get('stoch_k', 0),
                'stoch_d': last_candle.get('stoch_d', 0),
                'momentum': last_candle.get('momentum', 0),
                'volume_ratio': last_candle.get('volume_ratio', 0),
                'atr': last_candle.get('atr', 0),
                'session_quality': last_candle.get('good_session', 1) if 'good_session' in last_candle else session_ok,
                'reason': 'Bullish momentum with RSI, MACD, and Stochastic confirmation'
            }

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=signal_strength,
                metadata=metadata
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                               f"Generated BUY signal for {self.symbol} at {entry_price}. "
                               f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                               f"RSI: {last_candle.get('rsi', 0):.1f}, MACD hist: {last_candle.get('macd_histogram', 0):.6f}, "
                               f"Volume ratio: {last_candle.get('volume_ratio', 0):.1f}",
                               "MomentumScalpingStrategy")

        elif last_candle['signal'] == -1:  # Sell signal
            # Create SELL signal
            entry_price = last_candle['close']

            # FIXED: Calculate stop loss directly here if needed, don't rely on potentially stale value
            # First, we check if the stop_loss value is valid (not 0.0 or NaN)
            stop_loss = last_candle['stop_loss'] if 'stop_loss' in last_candle else None
            if stop_loss is None or pd.isna(stop_loss) or stop_loss == 0 or stop_loss <= entry_price:
                # Recalculate using ATR
                atr = last_candle['atr'] if 'atr' in last_candle else (last_candle['high'] - last_candle['low']) * 0.1
                stop_loss = entry_price + (atr * 1.5)  # Default 1.5 * ATR above entry

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate multiple take profit levels for scaling out
            risk = stop_loss - entry_price
            take_profit_1r = entry_price - risk  # 1:1 reward-to-risk
            take_profit_2r = entry_price - (risk * 2)  # 2:1 reward-to-risk

            # Get signal strength or default to 0.6 if not available
            signal_strength = last_candle['signal_strength'] if 'signal_strength' in last_candle else 0.6

            # Prepare metadata with safe fallbacks
            metadata = {
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'risk_amount': risk,
                'rsi': last_candle.get('rsi', 0),
                'macd_histogram': last_candle.get('macd_histogram', 0),
                'stoch_k': last_candle.get('stoch_k', 0),
                'stoch_d': last_candle.get('stoch_d', 0),
                'momentum': last_candle.get('momentum', 0),
                'volume_ratio': last_candle.get('volume_ratio', 0),
                'atr': last_candle.get('atr', 0),
                'session_quality': last_candle.get('good_session', 1) if 'good_session' in last_candle else session_ok,
                'reason': 'Bearish momentum with RSI, MACD, and Stochastic confirmation'
            }

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=signal_strength,
                metadata=metadata
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                               f"Generated SELL signal for {self.symbol} at {entry_price}. "
                               f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                               f"RSI: {last_candle.get('rsi', 0):.1f}, MACD hist: {last_candle.get('macd_histogram', 0):.6f}, "
                               f"Volume ratio: {last_candle.get('volume_ratio', 0):.1f}",
                               "MomentumScalpingStrategy")

        # Also check for momentum fading signals (for exit purposes)
        # These can be used by the order manager to exit open trades
        if 'momentum_fading' in last_candle and last_candle['momentum_fading'] == 1:  # Bullish momentum fading
            DBLogger.log_event("DEBUG",
                               f"Detected bullish momentum fading on {self.symbol}. "
                               f"RSI: {last_candle.get('rsi', 0):.1f}, MACD hist: {last_candle.get('macd_histogram', 0):.6f}",
                               "MomentumScalpingStrategy")

            # Create "CLOSE" signal for any LONG positions
            signal = self.create_signal(
                signal_type="CLOSE",
                price=last_candle['close'],
                strength=0.8,  # High priority for exit signals
                metadata={
                    'position_type': "BUY",  # Close long positions
                    'reason': 'Bullish momentum fading',
                    'rsi': last_candle.get('rsi', 0),
                    'macd_histogram': last_candle.get('macd_histogram', 0)
                }
            )
            signals.append(signal)

        elif 'momentum_fading' in last_candle and last_candle['momentum_fading'] == -1:  # Bearish momentum fading
            DBLogger.log_event("DEBUG",
                               f"Detected bearish momentum fading on {self.symbol}. "
                               f"RSI: {last_candle.get('rsi', 0):.1f}, MACD hist: {last_candle.get('macd_histogram', 0):.6f}",
                               "MomentumScalpingStrategy")

            # Create "CLOSE" signal for any SHORT positions
            signal = self.create_signal(
                signal_type="CLOSE",
                price=last_candle['close'],
                strength=0.8,  # High priority for exit signals
                metadata={
                    'position_type': "SELL",  # Close short positions
                    'reason': 'Bearish momentum fading',
                    'rsi': last_candle.get('rsi', 0),
                    'macd_histogram': last_candle.get('macd_histogram', 0)
                }
            )
            signals.append(signal)

        return signals
