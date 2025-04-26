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
        """Initialize the Enhanced Moving Average strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H1".
            fast_period (int, optional): Fast EMA period. Defaults to 20.
            slow_period (int, optional): Slow EMA period. Defaults to 50.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
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
        """Calculate strategy indicators on OHLC data.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with indicators added
        """
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

        # Calculate pullback detection
        # A pullback is when price retraces toward the EMA in an established trend
        data['pullback_to_fast_ema'] = np.where(
            (data['trend_bias'] == 1) &  # Bullish trend
            (data['low'] < data['fast_ema']) &  # Price dipped to/below fast EMA
            (data['close'] > data['fast_ema']),  # But closed above it
            1,  # Bullish pullback to fast EMA
            np.where(
                (data['trend_bias'] == -1) &  # Bearish trend
                (data['high'] > data['fast_ema']) &  # Price rose to/above fast EMA
                (data['close'] < data['fast_ema']),  # But closed below it
                -1,  # Bearish pullback to fast EMA
                0  # No pullback
            )
        )

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
        if data.empty or 'crossover' not in data.columns:
            self.logger.warning("Insufficient data for analysis after calculations")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]
        previous_candle = data.iloc[-2] if len(data) > 1 else None

        # Check for signals
        # 1. Crossover on the last candle
        if last_candle['crossover'] == 1:  # Bullish crossover
            signal = self._generate_bullish_crossover_signal(data, last_candle)
            if signal:
                signals.append(signal)

        elif last_candle['crossover'] == -1:  # Bearish crossover
            signal = self._generate_bearish_crossover_signal(data, last_candle)
            if signal:
                signals.append(signal)

        # 2. Pullback entries in established trends
        elif previous_candle is not None:
            # Bullish pullback entry in uptrend
            if (last_candle['pullback_to_fast_ema'] == 1 and
                    data['trend_bias'].iloc[-3:].mean() > 0.5):  # Consistent uptrend

                signal = self._generate_bullish_pullback_signal(data, last_candle)
                if signal:
                    signals.append(signal)

            # Bearish pullback entry in downtrend
            elif (last_candle['pullback_to_fast_ema'] == -1 and
                  data['trend_bias'].iloc[-3:].mean() < -0.5):  # Consistent downtrend

                signal = self._generate_bearish_pullback_signal(data, last_candle)
                if signal:
                    signals.append(signal)

        return signals

    def _generate_bullish_crossover_signal(self, data, last_candle):
        """Generate a bullish signal from EMA crossover.

        Args:
            data (pandas.DataFrame): OHLC data with indicators
            last_candle (pandas.Series): Last complete candle

        Returns:
            StrategySignal or None: The generated signal or None
        """
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
                'reason': 'Bullish EMA crossover'
            }
        )

        self.logger.info(
            f"Generated BUY signal for {self.symbol} at {entry_price}. "
            f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}, "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}"
        )

        return signal

    def _generate_bearish_crossover_signal(self, data, last_candle):
        """Generate a bearish signal from EMA crossover.

        Args:
            data (pandas.DataFrame): OHLC data with indicators
            last_candle (pandas.Series): Last complete candle

        Returns:
            StrategySignal or None: The generated signal or None
        """
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
                'reason': 'Bearish EMA crossover'
            }
        )

        self.logger.info(
            f"Generated SELL signal for {self.symbol} at {entry_price}. "
            f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}, "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}"
        )

        return signal

    def _generate_bullish_pullback_signal(self, data, last_candle):
        """Generate a bullish signal from pullback to EMA in uptrend.

        Args:
            data (pandas.DataFrame): OHLC data with indicators
            last_candle (pandas.Series): Last complete candle

        Returns:
            StrategySignal or None: The generated signal or None
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
                'reason': 'Pullback to fast EMA in uptrend'
            }
        )

        self.logger.info(
            f"Generated BUY signal for {self.symbol} at {entry_price} "
            f"(pullback to fast EMA in uptrend). "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}"
        )

        return signal

    def _generate_bearish_pullback_signal(self, data, last_candle):
        """Generate a bearish signal from pullback to EMA in downtrend.

        Args:
            data (pandas.DataFrame): OHLC data with indicators
            last_candle (pandas.Series): Last complete candle

        Returns:
            StrategySignal or None: The generated signal or None
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
                'reason': 'Pullback to fast EMA in downtrend'
            }
        )

        self.logger.info(
            f"Generated SELL signal for {self.symbol} at {entry_price} "
            f"(pullback to fast EMA in downtrend). "
            f"Stop: {stop_loss:.2f}, Target 1: {take_profit_1r:.2f}"
        )

        return signal