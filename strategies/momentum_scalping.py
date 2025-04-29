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
        data['volume_ratio'] = data['volume'] / data['volume_ma']

        # Flag high volume bars - fix astype error
        data['high_volume'] = np.where(data['volume_ratio'] >= self.volume_threshold, 1, 0)

        # Calculate volume change
        data['volume_change'] = data['volume'].pct_change()

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

    # strategies/momentum_scalping.py - Key fixes
    def _identify_signals(self, data):
        """Identify momentum scalping signals according to the plan specifications."""
        # Initialize signal columns
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data['signal_strength'] = 0.0
        data['stop_loss'] = np.nan
        data['take_profit'] = np.nan
        data['momentum_state'] = 0  # 0: Neutral, 1: Bullish, -1: Bearish
        data['momentum_fading'] = 0  # 0: Not fading, 1: Bullish fading, -1: Bearish fading

        for i in range(5, len(data)):
            # Skip if not enough prior data
            if i < 5:
                continue

            # Get current values
            current_close = data.iloc[i]['close']
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']

            # Get indicator values - check they exist first
            required_columns = ['rsi', 'macd', 'macd_signal', 'macd_histogram',
                                'stoch_k', 'stoch_d', 'momentum', 'volume_ratio']
            if not all(col in data.columns for col in required_columns):
                continue

            current_rsi = data.iloc[i]['rsi']
            current_macd = data.iloc[i]['macd']
            current_macd_signal = data.iloc[i]['macd_signal']
            current_macd_hist = data.iloc[i]['macd_histogram']
            current_stoch_k = data.iloc[i]['stoch_k']
            current_stoch_d = data.iloc[i]['stoch_d']
            current_momentum = data.iloc[i]['momentum']
            current_volume_ratio = data.iloc[i]['volume_ratio']

            # Previous values for comparison
            prev_close = data.iloc[i - 1]['close']
            prev_high = data.iloc[i - 1]['high']
            prev_low = data.iloc[i - 1]['low']
            prev_stoch_k = data.iloc[i - 1]['stoch_k']
            prev_stoch_d = data.iloc[i - 1]['stoch_d']
            prev_macd_hist = data.iloc[i - 1]['macd_histogram']

            # Check session time if enabled
            good_session = True
            if self.consider_session and 'good_session' in data.columns:
                good_session = data.iloc[i]['good_session'] == 1

            # PRICE ACTION TRIGGER: Check for breakout or strong momentum candle
            # Per plan: "a breakout is defined as price moving beyond a key level and holding beyond it"
            # Look back for recent high/low levels
            lookback = 20  # Look back 20 bars for significant levels
            lookback_start = max(0, i - lookback)
            recent_high = data.iloc[lookback_start:i]['high'].max()
            recent_low = data.iloc[lookback_start:i]['low'].min()

            # Calculate average candle size for comparison
            avg_candle_size = (data.iloc[lookback_start:i]['high'] - data.iloc[lookback_start:i]['low']).mean()
            current_candle_size = current_high - current_low

            # Check if current candle is significantly larger (breakout condition)
            large_candle = current_candle_size > (avg_candle_size * 1.5)

            # Check for breakout above recent high or below recent low
            breakout_up = current_close > recent_high and current_close > prev_high
            breakout_down = current_close < recent_low and current_close < prev_low

            # MOMENTUM INDICATOR CONFIRMATION
            # Per plan: "RSI > 60 and rising for bullish momentum"
            bullish_rsi = current_rsi > self.rsi_threshold_high
            bearish_rsi = current_rsi < self.rsi_threshold_low

            # MACD confirmation
            # Per plan: "MACD histogram should be > 0 and preferably increasing"
            bullish_macd = current_macd_hist > 0 and current_macd > current_macd_signal
            bearish_macd = current_macd_hist < 0 and current_macd < current_macd_signal

            # Stochastic confirmation
            # Per plan: "Stochastic %K crossing above %D, or already in the 80+ zone"
            stoch_bullish_cross = (current_stoch_k > current_stoch_d) and (prev_stoch_k <= prev_stoch_d)
            stoch_bearish_cross = (current_stoch_k < current_stoch_d) and (prev_stoch_k >= prev_stoch_d)
            stoch_bullish = current_stoch_k > current_stoch_d or current_stoch_k > 80
            stoch_bearish = current_stoch_k < current_stoch_d or current_stoch_k < 20

            # Momentum/ROC
            # Per plan: "Momentum > 100.20 (+0.2%) for long"
            bullish_momentum = current_momentum > 100.2
            bearish_momentum = current_momentum < 99.8

            # VOLUME SURGE CONFIRMATION
            # Per plan: "Volume should exceed recent averages (>150% of 20-bar average)"
            high_volume = current_volume_ratio >= self.volume_threshold  # Already calculated in preparation

            # COMBINED ENTRY CONDITIONS
            # For a long entry: RSI>60, MACD>0, Stochastic bullish, Volume high, Price breakout up
            bullish_conditions = (
                    bullish_rsi and
                    bullish_macd and
                    (stoch_bullish or stoch_bullish_cross) and  # Using stochastic crossover signal
                    bullish_momentum and
                    high_volume and
                    (breakout_up or large_candle) and
                    good_session
            )

            # For a short entry: RSI<40, MACD<0, Stochastic bearish, Volume high, Price breakout down
            bearish_conditions = (
                    bearish_rsi and
                    bearish_macd and
                    (stoch_bearish or stoch_bearish_cross) and  # Using stochastic crossover signal
                    bearish_momentum and
                    high_volume and
                    (breakout_down or large_candle) and
                    good_session
            )

            # Check for BUY signal
            if bullish_conditions:
                # Calculate signal strength based on multiple factors
                strength_factors = [
                    (current_rsi - 50) / 25,  # Scale 0-1 based on RSI strength
                    current_macd_hist / data.iloc[i]['atr'],  # Scale relative to volatility
                    (current_stoch_k - 50) / 50,  # Scale based on stochastic
                    (current_volume_ratio - 1) / self.volume_threshold  # Volume strength
                ]

                # Add extra strength for stochastic crossover, especially from oversold
                if stoch_bullish_cross:
                    crossover_from_oversold = prev_stoch_k < 20
                    strength_factors.append(0.2 if crossover_from_oversold else 0.1)

                # Average the factors and cap at 1.0
                signal_strength = min(1.0, sum([max(0, s) for s in strength_factors]) / len(strength_factors))

                # Calculate stop loss (use ATR for this example)
                atr = data.iloc[i]['atr']
                stop_loss = current_close - (atr * 1.5)

                # Calculate take profit (1:1 risk-reward for now)
                risk = current_close - stop_loss
                take_profit = current_close + risk

                # Set signal values
                data.iloc[i, data.columns.get_loc('signal')] = 1  # Buy signal
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                data.iloc[i, data.columns.get_loc('stop_loss')] = stop_loss
                data.iloc[i, data.columns.get_loc('take_profit')] = take_profit
                data.iloc[i, data.columns.get_loc('momentum_state')] = 1  # Bullish state

            # Check for SELL signal
            elif bearish_conditions:
                # Calculate signal strength based on multiple factors
                strength_factors = [
                    (50 - current_rsi) / 25,  # Scale 0-1 based on RSI strength
                    -current_macd_hist / data.iloc[i]['atr'],  # Scale relative to volatility
                    (50 - current_stoch_k) / 50,  # Scale based on stochastic
                    (current_volume_ratio - 1) / self.volume_threshold  # Volume strength
                ]

                # Add extra strength for stochastic crossover, especially from overbought
                if stoch_bearish_cross:
                    crossover_from_overbought = prev_stoch_k > 80
                    strength_factors.append(0.2 if crossover_from_overbought else 0.1)

                # Average the factors and cap at 1.0
                signal_strength = min(1.0, sum([max(0, s) for s in strength_factors]) / len(strength_factors))

                # Calculate stop loss
                atr = data.iloc[i]['atr']
                stop_loss = current_close + (atr * 1.5)

                # Calculate take profit (1:1 risk-reward for now)
                risk = stop_loss - current_close
                take_profit = current_close - risk

                # Set signal values
                data.iloc[i, data.columns.get_loc('signal')] = -1  # Sell signal
                data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                data.iloc[i, data.columns.get_loc('stop_loss')] = stop_loss
                data.iloc[i, data.columns.get_loc('take_profit')] = take_profit
                data.iloc[i, data.columns.get_loc('momentum_state')] = -1  # Bearish state

            # MOMENTUM FADING DETECTION FOR EXITS
            # Per plan: "exit when RSI drops back below 50, MACD histogram shrinks, etc."
            if i > 0 and data.iloc[i - 1]['momentum_state'] == 1:  # Previous bar was bullish
                # Check if momentum is fading
                momentum_fading = (
                        current_rsi < 50 or
                        current_macd_hist < prev_macd_hist or  # Histogram shrinking
                        (current_stoch_k < current_stoch_d and prev_stoch_k >= prev_stoch_d)  # Bearish stoch cross
                )

                if momentum_fading:
                    data.iloc[i, data.columns.get_loc('momentum_fading')] = 1  # Bullish momentum fading

            elif i > 0 and data.iloc[i - 1]['momentum_state'] == -1:  # Previous bar was bearish
                # Check if momentum is fading
                momentum_fading = (
                        current_rsi > 50 or
                        current_macd_hist > prev_macd_hist or  # Histogram shrinking (in negative)
                        (current_stoch_k > current_stoch_d and prev_stoch_k <= prev_stoch_d)  # Bullish stoch cross
                )

                if momentum_fading:
                    data.iloc[i, data.columns.get_loc('momentum_fading')] = -1  # Bearish momentum fading

        return data

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
        if hasattr(self.data_fetcher, 'connector'):
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

        # Check if we're in a good session time (if session analysis is enabled)
        if self.consider_session and 'good_session' in data.columns:
            if last_candle['good_session'] != 1:
                current_time = datetime.now().time()
                DBLogger.log_event("DEBUG",
                                f"Current time {current_time} is not in optimal trading session for momentum scalping",
                                "MomentumScalpingStrategy")
                # Continue with analysis but be more selective on signals
                # (signal strength will be lower outside optimal sessions)

        # Check for trading signal on the last candle
        if last_candle['signal'] == 1:  # Buy signal
            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate multiple take profit levels for scaling out
            risk = entry_price - stop_loss
            take_profit_1r = entry_price + risk  # 1:1 reward-to-risk
            take_profit_2r = entry_price + (risk * 2)  # 2:1 reward-to-risk

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1r': take_profit_1r,
                    'take_profit_2r': take_profit_2r,
                    'risk_amount': risk,
                    'rsi': last_candle['rsi'],
                    'macd_histogram': last_candle['macd_histogram'],
                    'stoch_k': last_candle['stoch_k'],
                    'stoch_d': last_candle['stoch_d'],
                    'momentum': last_candle['momentum'],
                    'volume_ratio': last_candle['volume_ratio'],
                    'atr': last_candle['atr'],
                    'session_quality': last_candle['good_session'] if 'good_session' in last_candle else 1,
                    'reason': 'Bullish momentum with RSI, MACD, and Stochastic confirmation'
                }
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                           f"Generated BUY signal for {self.symbol} at {entry_price}. "
                           f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                           f"RSI: {last_candle['rsi']:.1f}, MACD hist: {last_candle['macd_histogram']:.6f}, "
                           f"Volume ratio: {last_candle['volume_ratio']:.1f}",
                           "MomentumScalpingStrategy")

        elif last_candle['signal'] == -1:  # Sell signal
            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate multiple take profit levels for scaling out
            risk = stop_loss - entry_price
            take_profit_1r = entry_price - risk  # 1:1 reward-to-risk
            take_profit_2r = entry_price - (risk * 2)  # 2:1 reward-to-risk

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1r': take_profit_1r,
                    'take_profit_2r': take_profit_2r,
                    'risk_amount': risk,
                    'rsi': last_candle['rsi'],
                    'macd_histogram': last_candle['macd_histogram'],
                    'stoch_k': last_candle['stoch_k'],
                    'stoch_d': last_candle['stoch_d'],
                    'momentum': last_candle['momentum'],
                    'volume_ratio': last_candle['volume_ratio'],
                    'atr': last_candle['atr'],
                    'session_quality': last_candle['good_session'] if 'good_session' in last_candle else 1,
                    'reason': 'Bearish momentum with RSI, MACD, and Stochastic confirmation'
                }
            )
            signals.append(signal)

            DBLogger.log_event("INFO",
                           f"Generated SELL signal for {self.symbol} at {entry_price}. "
                           f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                           f"RSI: {last_candle['rsi']:.1f}, MACD hist: {last_candle['macd_histogram']:.6f}, "
                           f"Volume ratio: {last_candle['volume_ratio']:.1f}",
                           "MomentumScalpingStrategy")

        # Also check for momentum fading signals (for exit purposes)
        # These can be used by the order manager to exit open trades
        if last_candle['momentum_fading'] == 1:  # Bullish momentum fading
            DBLogger.log_event("DEBUG",
                           f"Detected bullish momentum fading on {self.symbol}. "
                           f"RSI: {last_candle['rsi']:.1f}, MACD hist: {last_candle['macd_histogram']:.6f}",
                           "MomentumScalpingStrategy")

            # Create "CLOSE" signal for any LONG positions
            signal = self.create_signal(
                signal_type="CLOSE",
                price=last_candle['close'],
                strength=0.8,  # High priority for exit signals
                metadata={
                    'position_type': "BUY",  # Close long positions
                    'reason': 'Bullish momentum fading',
                    'rsi': last_candle['rsi'],
                    'macd_histogram': last_candle['macd_histogram']
                }
            )
            signals.append(signal)

        elif last_candle['momentum_fading'] == -1:  # Bearish momentum fading
            DBLogger.log_event("DEBUG",
                           f"Detected bearish momentum fading on {self.symbol}. "
                           f"RSI: {last_candle['rsi']:.1f}, MACD hist: {last_candle['macd_histogram']:.6f}",
                           "MomentumScalpingStrategy")

            # Create "CLOSE" signal for any SHORT positions
            signal = self.create_signal(
                signal_type="CLOSE",
                price=last_candle['close'],
                strength=0.8,  # High priority for exit signals
                metadata={
                    'position_type': "SELL",  # Close short positions
                    'reason': 'Bearish momentum fading',
                    'rsi': last_candle['rsi'],
                    'macd_histogram': last_candle['macd_histogram']
                }
            )
            signals.append(signal)

        return signals