# strategies/moving_average.py
import numpy as np
from strategies.base_strategy import BaseStrategy


class EnhancedMovingAverageStrategy(BaseStrategy):
    """Enhanced Moving Average Trend-Following Strategy for XAU/USD.

    This strategy is based on two EMAs (fast and slow) and includes:
    1. Crossover entry signals
    2. Pullback entry signals for improved entry timing
    3. Advanced stop-loss placement using swing levels
    4. Partial profit-taking and trailing stop management

    For H1 timeframe, recommended values are fast_period=20, slow_period=50.
    """

    def __init__(self, symbol="XAUUSD", timeframe="H1",
                 fast_period=20, slow_period=50, data_fetcher=None):
        """Initialize the Enhanced Moving Average strategy."""
        super().__init__(symbol, timeframe, name="EnhancedMA_Trend", data_fetcher=data_fetcher)

        # Validate inputs
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")

        self.fast_period = fast_period
        self.slow_period = slow_period

        # Ensure we fetch enough data for calculations
        self.min_required_candles = slow_period + 30  # Need extra bars for swing high/low detection

        self.logger.info(
            f"Initialized Enhanced Moving Average strategy: {symbol} {timeframe}, "
            f"Fast EMA: {fast_period}, Slow EMA: {slow_period}"
        )

    def calculate_indicators(self, data):
        """Calculate strategy indicators on OHLC data."""
        if len(data) < self.min_required_candles:
            self.logger.warning(
                f"Insufficient data for MA calculations. "
                f"Need at least {self.min_required_candles} candles."
            )
            return data

        # Calculate EMAs
        data['fast_ema'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=self.slow_period, adjust=False).mean()

        # Calculate crossover signals
        data['ema_diff'] = data['fast_ema'] - data['slow_ema']
        data['crossover'] = np.where(
            (data['ema_diff'] > 0) & (data['ema_diff'].shift(1) <= 0),
            1,  # Bullish crossover
            np.where(
                (data['ema_diff'] < 0) & (data['ema_diff'].shift(1) >= 0),
                -1,  # Bearish crossover
                0  # No crossover
            )
        )

        # Identify trend bias based on EMAs
        data['trend_bias'] = np.where(
            data['fast_ema'] > data['slow_ema'],
            1,  # Bullish trend
            np.where(
                data['fast_ema'] < data['slow_ema'],
                -1,  # Bearish trend
                0  # Neutral
            )
        )

        # Calculate ATR for volatility measurement (useful for stops and targets)
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['close'].shift(1))
        data['tr3'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()

        # Identify swing highs and lows for stop placement
        # We'll use a 5-bar window centered on the current bar
        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Enhanced pullback detection (key improvement)
        # A pullback is when price retraces toward the EMA in an established trend
        data = self._calculate_pullback_signals(data)

        # Check for higher highs/higher lows (trend confirmation)
        data = self._calculate_trend_confirmation(data)

        return data

    def _calculate_pullback_signals(self, data):
        """Calculate pullback signals based on price interaction with EMAs.

        This is a more sophisticated approach to detect pullbacks to the EMAs
        in established trends per the original plan.
        """
        # Initialize pullback column
        data['pullback_to_ema'] = 0

        # Need at least 5 bars of data
        if len(data) < 5:
            return data

        # Loop through the data to detect pullbacks
        for i in range(5, len(data)):
            # Check for bullish trend
            if data.iloc[i - 3:i]['trend_bias'].mean() > 0.8:  # Strong bullish trend
                # Check if price pulled back to the fast EMA
                if (data.iloc[i]['low'] <= data.iloc[i]['fast_ema'] and
                        data.iloc[i]['close'] > data.iloc[i]['fast_ema']):
                    # This is a pullback to fast EMA in uptrend
                    data.loc[data.index[i], 'pullback_to_ema'] = 1

                # Check if price pulled back to the slow EMA (deeper pullback)
                elif (data.iloc[i]['low'] <= data.iloc[i]['slow_ema'] and
                      data.iloc[i]['close'] > data.iloc[i]['slow_ema']):
                    # This is a pullback to slow EMA in uptrend
                    data.loc[data.index[i], 'pullback_to_ema'] = 2

            # Check for bearish trend
            elif data.iloc[i - 3:i]['trend_bias'].mean() < -0.8:  # Strong bearish trend
                # Check if price pulled back to the fast EMA
                if (data.iloc[i]['high'] >= data.iloc[i]['fast_ema'] and
                        data.iloc[i]['close'] < data.iloc[i]['fast_ema']):
                    # This is a pullback to fast EMA in downtrend
                    data.loc[data.index[i], 'pullback_to_ema'] = -1

                # Check if price pulled back to the slow EMA (deeper pullback)
                elif (data.iloc[i]['high'] >= data.iloc[i]['slow_ema'] and
                      data.iloc[i]['close'] < data.iloc[i]['slow_ema']):
                    # This is a pullback to slow EMA in downtrend
                    data.loc[data.index[i], 'pullback_to_ema'] = -2

        return data

    def _calculate_trend_confirmation(self, data):
        """Calculate trend confirmation based on price making higher highs/higher lows.

        This implements the "higher highs and higher lows" confirmation
        mentioned in the original plan.
        """
        # Initialize trend confirmation columns
        data['higher_highs'] = False
        data['higher_lows'] = False
        data['lower_highs'] = False
        data['lower_lows'] = False

        # Need at least 10 bars to confirm trend
        if len(data) < 10:
            return data

        # Analyze for higher highs and higher lows (uptrend)
        for i in range(10, len(data)):
            # Get recent highs and lows
            recent_highs = data.iloc[i - 10:i]['high'].rolling(window=3).max()
            recent_lows = data.iloc[i - 10:i]['low'].rolling(window=3).min()

            # Check for higher highs (last 3 swing highs are rising)
            if (len(recent_highs.dropna()) >= 3 and
                    recent_highs.iloc[-1] > recent_highs.iloc[-4] and
                    recent_highs.iloc[-4] > recent_highs.iloc[-7]):
                data.loc[data.index[i], 'higher_highs'] = True

            # Check for higher lows (last 3 swing lows are rising)
            if (len(recent_lows.dropna()) >= 3 and
                    recent_lows.iloc[-1] > recent_lows.iloc[-4] and
                    recent_lows.iloc[-4] > recent_lows.iloc[-7]):
                data.loc[data.index[i], 'higher_lows'] = True

            # Check for lower highs (downtrend)
            if (len(recent_highs.dropna()) >= 3 and
                    recent_highs.iloc[-1] < recent_highs.iloc[-4] and
                    recent_highs.iloc[-4] < recent_highs.iloc[-7]):
                data.loc[data.index[i], 'lower_highs'] = True

            # Check for lower lows (downtrend)
            if (len(recent_lows.dropna()) >= 3 and
                    recent_lows.iloc[-1] < recent_lows.iloc[-4] and
                    recent_lows.iloc[-4] < recent_lows.iloc[-7]):
                data.loc[data.index[i], 'lower_lows'] = True

        return data

    def analyze(self, data):
        """Analyze market data and generate trading signals."""
        # Calculate indicators
        data = self.calculate_indicators(data)

        # Check if we have sufficient data after calculations
        if data.empty or 'crossover' not in data.columns:
            self.logger.warning("Insufficient data for analysis after calculations")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]
        previous_candle = data.iloc[-2] if len(data) > 1 else None

        # Check for signals

        # 1. Crossover Signals (as in original plan)
        if last_candle['crossover'] == 1:  # Bullish crossover
            # Enhance with trend confirmation
            if last_candle['higher_lows']:  # Confirm uptrend with higher lows
                signal = self._generate_bullish_crossover_signal(data, last_candle)
                if signal:
                    signals.append(signal)

        elif last_candle['crossover'] == -1:  # Bearish crossover
            # Enhance with trend confirmation
            if last_candle['lower_highs']:  # Confirm downtrend with lower highs
                signal = self._generate_bearish_crossover_signal(data, last_candle)
                if signal:
                    signals.append(signal)

        # 2. Pullback Entry Signals (key addition from the plan)
        elif previous_candle is not None:
            # Bullish pullback to EMA in established uptrend
            if (last_candle['pullback_to_ema'] > 0 and
                    last_candle['higher_lows'] and  # Trending higher
                    data['trend_bias'].iloc[-5:].mean() > 0.8):  # Strong uptrend

                signal = self._generate_bullish_pullback_signal(data, last_candle)
                if signal:
                    signals.append(signal)

            # Bearish pullback to EMA in established downtrend
            elif (last_candle['pullback_to_ema'] < 0 and
                  last_candle['lower_highs'] and  # Trending lower
                  data['trend_bias'].iloc[-5:].mean() < -0.8):  # Strong downtrend

                signal = self._generate_bearish_pullback_signal(data, last_candle)
                if signal:
                    signals.append(signal)

        return signals

    def _generate_bullish_crossover_signal(self, data, last_candle):
        """Generate a bullish signal from EMA crossover."""
        # Find recent swing low for stop loss placement
        recent_swing_low = data['swing_low'].iloc[-6:-1].min()

        # Create BUY signal
        entry_price = last_candle['close']
        stop_loss = recent_swing_low

        # Ensure stop loss is valid and not too close
        if stop_loss >= entry_price * 0.998:  # Less than 0.2% away
            # Use ATR-based stop instead
            stop_loss = entry_price - (last_candle['atr'] * 1.5)

        # Calculate take profit levels using reward:risk ratio
        risk = entry_price - stop_loss
        take_profit_1r = entry_price + risk  # 1:1 reward:risk for first target
        take_profit_2r = entry_price + (risk * 2)  # 2:1 reward:risk for second target

        signal = self.create_signal(
            signal_type="BUY",
            price=entry_price,
            strength=0.8,  # Higher strength for crossover signals
            metadata={
                'fast_ema': last_candle['fast_ema'],
                'slow_ema': last_candle['slow_ema'],
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'atr': last_candle['atr'],
                'signal_type': 'crossover',
                'reason': 'Bullish EMA crossover with trend confirmation',
                'timeframe': self.timeframe,
                'swing_low': recent_swing_low,
                'higher_lows': bool(last_candle['higher_lows']),
                'higher_highs': bool(last_candle['higher_highs']),
            }
        )

        self.logger.info(
            f"Generated BUY signal for {self.symbol} at {entry_price}. "
            f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}, "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}, Target 2: {take_profit_2r:.2f}"
        )

        return signal

    def _generate_bearish_crossover_signal(self, data, last_candle):
        """Generate a bearish signal from EMA crossover."""
        # Find recent swing high for stop loss placement
        recent_swing_high = data['swing_high'].iloc[-6:-1].max()

        # Create SELL signal
        entry_price = last_candle['close']
        stop_loss = recent_swing_high

        # Ensure stop loss is valid and not too close
        if stop_loss <= entry_price * 1.002:  # Less than 0.2% away
            # Use ATR-based stop instead
            stop_loss = entry_price + (last_candle['atr'] * 1.5)

        # Calculate take profit levels using reward:risk ratio
        risk = stop_loss - entry_price
        take_profit_1r = entry_price - risk  # 1:1 reward:risk for first target
        take_profit_2r = entry_price - (risk * 2)  # 2:1 reward:risk for second target

        signal = self.create_signal(
            signal_type="SELL",
            price=entry_price,
            strength=0.8,  # Higher strength for crossover signals
            metadata={
                'fast_ema': last_candle['fast_ema'],
                'slow_ema': last_candle['slow_ema'],
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'atr': last_candle['atr'],
                'signal_type': 'crossover',
                'reason': 'Bearish EMA crossover with trend confirmation',
                'timeframe': self.timeframe,
                'swing_high': recent_swing_high,
                'lower_lows': bool(last_candle['lower_lows']),
                'lower_highs': bool(last_candle['lower_highs']),
            }
        )

        self.logger.info(
            f"Generated SELL signal for {self.symbol} at {entry_price}. "
            f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}, "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}, Target 2: {take_profit_2r:.2f}"
        )

        return signal

    def _generate_bullish_pullback_signal(self, data, last_candle):
        """Generate a bullish signal from pullback to EMA in uptrend.

        This implements the pullback entry method described in the plan:
        "if gold is in an uptrend (price making higher highs and holding above the 50 EMA),
        wait for price to dip near the 50 EMA or a known support and then buy"
        """
        # Find recent swing low for stop loss placement
        recent_swing_low = data['swing_low'].iloc[-6:-1].min()

        # Create BUY signal
        entry_price = last_candle['close']

        # For pullbacks, we use either the swing low or the slow EMA as stop
        # whichever is closer to price
        slow_ema_stop = last_candle['slow_ema'] * 0.997  # Just below slow EMA
        stop_loss = max(recent_swing_low, slow_ema_stop)

        # Ensure stop isn't too far (risk management)
        max_risk_pct = 0.01  # Maximum 1% risk from price
        if (entry_price - stop_loss) / entry_price > max_risk_pct:
            # Use ATR-based stop instead
            stop_loss = entry_price - (last_candle['atr'] * 1.5)

        # Calculate take profit levels using reward:risk ratio
        risk = entry_price - stop_loss
        take_profit_1r = entry_price + risk  # 1:1 reward:risk for first target
        take_profit_2r = entry_price + (risk * 2)  # 2:1 reward:risk for second target

        # Determine pullback type
        pullback_type = "Fast EMA"
        if last_candle['pullback_to_ema'] == 2:
            pullback_type = "Slow EMA"

        signal = self.create_signal(
            signal_type="BUY",
            price=entry_price,
            strength=0.7,  # Slightly lower strength for pullback entries
            metadata={
                'fast_ema': last_candle['fast_ema'],
                'slow_ema': last_candle['slow_ema'],
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'atr': last_candle['atr'],
                'signal_type': 'pullback',
                'pullback_type': pullback_type,
                'reason': f'Pullback to {pullback_type} in confirmed uptrend',
                'timeframe': self.timeframe,
                'swing_low': recent_swing_low,
                'higher_lows': bool(last_candle['higher_lows']),
                'higher_highs': bool(last_candle['higher_highs']),
            }
        )

        self.logger.info(
            f"Generated BUY signal for {self.symbol} at {entry_price} "
            f"(pullback to {pullback_type} in uptrend). "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}, Target 2: {take_profit_2r:.2f}"
        )

        return signal

    def _generate_bearish_pullback_signal(self, data, last_candle):
        """Generate a bearish signal from pullback to EMA in downtrend.

        This implements the pullback entry method described in the plan:
        "In a downtrend, wait for a rally toward the EMA or a resistance and then sell from a higher level"
        """
        # Find recent swing high for stop loss placement
        recent_swing_high = data['swing_high'].iloc[-6:-1].max()

        # Create SELL signal
        entry_price = last_candle['close']

        # For pullbacks, we use either the swing high or the slow EMA as stop
        # whichever is closer to price
        slow_ema_stop = last_candle['slow_ema'] * 1.003  # Just above slow EMA
        stop_loss = min(recent_swing_high, slow_ema_stop)

        # Ensure stop isn't too far (risk management)
        max_risk_pct = 0.01  # Maximum 1% risk from price
        if (stop_loss - entry_price) / entry_price > max_risk_pct:
            # Use ATR-based stop instead
            stop_loss = entry_price + (last_candle['atr'] * 1.5)

        # Calculate take profit levels using reward:risk ratio
        risk = stop_loss - entry_price
        take_profit_1r = entry_price - risk  # 1:1 reward:risk for first target
        take_profit_2r = entry_price - (risk * 2)  # 2:1 reward:risk for second target

        # Determine pullback type
        pullback_type = "Fast EMA"
        if last_candle['pullback_to_ema'] == -2:
            pullback_type = "Slow EMA"

        signal = self.create_signal(
            signal_type="SELL",
            price=entry_price,
            strength=0.7,  # Slightly lower strength for pullback entries
            metadata={
                'fast_ema': last_candle['fast_ema'],
                'slow_ema': last_candle['slow_ema'],
                'stop_loss': stop_loss,
                'take_profit_1r': take_profit_1r,
                'take_profit_2r': take_profit_2r,
                'atr': last_candle['atr'],
                'signal_type': 'pullback',
                'pullback_type': pullback_type,
                'reason': f'Pullback to {pullback_type} in confirmed downtrend',
                'timeframe': self.timeframe,
                'swing_high': recent_swing_high,
                'lower_lows': bool(last_candle['lower_lows']),
                'lower_highs': bool(last_candle['lower_highs']),
            }
        )

        self.logger.info(
            f"Generated SELL signal for {self.symbol} at {entry_price} "
            f"(pullback to {pullback_type} in downtrend). "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}, Target 2: {take_profit_2r:.2f}"
        )

        return signal