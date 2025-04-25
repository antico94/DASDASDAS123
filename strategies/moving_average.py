# strategies/moving_average.py
import pandas as pd
import numpy as np
from datetime import datetime
from logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class MovingAverageStrategy(BaseStrategy):
    """Moving Average Trend-Following Strategy for XAU/USD."""

    def __init__(self, symbol="XAUUSD", timeframe="H1",
                 fast_period=20, slow_period=50, data_fetcher=None):
        """Initialize the Moving Average strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H1".
            fast_period (int, optional): Fast MA period. Defaults to 20.
            slow_period (int, optional): Slow MA period. Defaults to 50.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="MA_Trend", data_fetcher=data_fetcher)

        # Validate inputs
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")

        self.fast_period = fast_period
        self.slow_period = slow_period

        # Ensure we fetch enough data for calculations
        self.min_required_candles = slow_period + 10

        self.logger.info(
            f"Initialized Moving Average strategy: {symbol} {timeframe}, "
            f"Fast MA: {fast_period}, Slow MA: {slow_period}"
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

        # Identify swing highs and lows for stop placement
        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

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

        # Check for crossover on the last candle
        if last_candle['crossover'] == 1:  # Bullish crossover
            # Find recent swing low for stop loss placement
            recent_swing_low = data['swing_low'].iloc[-6:-1].min()

            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = recent_swing_low

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=0.7,  # Higher strength for crossover signals
                metadata={
                    'fast_ema': last_candle['fast_ema'],
                    'slow_ema': last_candle['slow_ema'],
                    'stop_loss': stop_loss,
                    'reason': 'Bullish EMA crossover'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}"
            )

        elif last_candle['crossover'] == -1:  # Bearish crossover
            # Find recent swing high for stop loss placement
            recent_swing_high = data['swing_high'].iloc[-6:-1].max()

            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = recent_swing_high

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=0.7,
                metadata={
                    'fast_ema': last_candle['fast_ema'],
                    'slow_ema': last_candle['slow_ema'],
                    'stop_loss': stop_loss,
                    'reason': 'Bearish EMA crossover'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Fast EMA: {last_candle['fast_ema']:.2f}, Slow EMA: {last_candle['slow_ema']:.2f}"
            )

        # Check for pullback entries (price approaching MA after established trend)
        # This is a more refined entry strategy as mentioned in the research
        elif previous_candle is not None:
            # Uptrend pullback entry (buy on pullback to fast EMA in uptrend)
            if (
                    last_candle['fast_ema'] > last_candle['slow_ema'] and  # Uptrend condition
                    last_candle['close'] > last_candle['slow_ema'] and  # Price above slow EMA confirms trend
                    previous_candle['low'] < previous_candle['fast_ema'] and  # Previous candle touched/crossed fast EMA
                    last_candle['close'] > last_candle['fast_ema']  # Current candle closed back above fast EMA
            ):
                # Find recent swing low for stop loss placement
                recent_swing_low = data['swing_low'].iloc[-6:-1].min()

                entry_price = last_candle['close']
                stop_loss = min(recent_swing_low, last_candle['slow_ema'])

                # Ensure stop loss is valid
                if stop_loss >= entry_price:
                    stop_loss = entry_price * 0.995  # Default 0.5% below entry

                signal = self.create_signal(
                    signal_type="BUY",
                    price=entry_price,
                    strength=0.6,  # Slightly lower strength for pullback entries
                    metadata={
                        'fast_ema': last_candle['fast_ema'],
                        'slow_ema': last_candle['slow_ema'],
                        'stop_loss': stop_loss,
                        'reason': 'Pullback to fast EMA in uptrend'
                    }
                )
                signals.append(signal)

                self.logger.info(
                    f"Generated BUY signal for {self.symbol} at {entry_price} "
                    f"(pullback to fast EMA in uptrend)"
                )

            # Downtrend pullback entry (sell on pullback to fast EMA in downtrend)
            elif (
                    last_candle['fast_ema'] < last_candle['slow_ema'] and  # Downtrend condition
                    last_candle['close'] < last_candle['slow_ema'] and  # Price below slow EMA confirms trend
                    previous_candle['high'] > previous_candle[
                        'fast_ema'] and  # Previous candle touched/crossed fast EMA
                    last_candle['close'] < last_candle['fast_ema']  # Current candle closed back below fast EMA
            ):
                # Find recent swing high for stop loss placement
                recent_swing_high = data['swing_high'].iloc[-6:-1].max()

                entry_price = last_candle['close']
                stop_loss = max(recent_swing_high, last_candle['slow_ema'])

                # Ensure stop loss is valid
                if stop_loss <= entry_price:
                    stop_loss = entry_price * 1.005  # Default 0.5% above entry

                signal = self.create_signal(
                    signal_type="SELL",
                    price=entry_price,
                    strength=0.6,
                    metadata={
                        'fast_ema': last_candle['fast_ema'],
                        'slow_ema': last_candle['slow_ema'],
                        'stop_loss': stop_loss,
                        'reason': 'Pullback to fast EMA in downtrend'
                    }
                )
                signals.append(signal)

                self.logger.info(
                    f"Generated SELL signal for {self.symbol} at {entry_price} "
                    f"(pullback to fast EMA in downtrend)"
                )

        return signals