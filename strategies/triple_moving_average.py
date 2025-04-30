# strategies/triple_moving_average.py
import numpy as np
import pandas as pd
from db_logger.db_logger import DBLogger
from strategies.base_strategy import BaseStrategy


class TripleMovingAverageStrategy(BaseStrategy):
    """Triple Moving Average Strategy for XAU/USD.

    This strategy uses three moving averages:
    - Fast EMA (10 period) for entry/exit signals
    - Medium SMA (50 period) for crossover signals
    - Slow SMA (200 period) as trend filter

    Signal generation logic:
    - Long entry: Fast EMA crosses above Medium SMA with Medium SMA above Slow SMA
    - Short entry: Fast EMA crosses below Medium SMA with Medium SMA below Slow SMA
    - Exit long: Fast EMA crosses below Medium SMA
    - Exit short: Fast EMA crosses above Medium SMA
    """

    def __init__(self, symbol="XAUUSD", timeframe="H4",
                 fast_period=10, medium_period=50, slow_period=200, data_fetcher=None):
        """Initialize the Triple Moving Average strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H4".
            fast_period (int, optional): Fast EMA period. Defaults to 10.
            medium_period (int, optional): Medium SMA period. Defaults to 50.
            slow_period (int, optional): Slow SMA period. Defaults to 200.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="TripleMA_Trend", data_fetcher=data_fetcher)

        # Validate inputs
        if fast_period >= medium_period:
            error_msg = f"Fast period ({fast_period}) must be less than medium period ({medium_period})"
            DBLogger.log_error("TripleMovingAverageStrategy", error_msg)
            raise ValueError(error_msg)

        if medium_period >= slow_period:
            error_msg = f"Medium period ({medium_period}) must be less than slow period ({slow_period})"
            DBLogger.log_error("TripleMovingAverageStrategy", error_msg)
            raise ValueError(error_msg)

        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period

        # Ensure we fetch enough data for calculations
        self.min_required_candles = slow_period + 10  # Need extra bars for calculations

        DBLogger.log_event("INFO",
                           f"Initialized Triple Moving Average strategy: {symbol} {timeframe}, "
                           f"Fast EMA: {fast_period}, Medium SMA: {medium_period}, Slow SMA: {slow_period}",
                           "TripleMovingAverageStrategy")

    def calculate_indicators(self, data):
        """Calculate strategy indicators on OHLC data.

        Args:
            data (pd.DataFrame): OHLC data with columns 'open', 'high', 'low', 'close', 'volume'

        Returns:
            pd.DataFrame: Data with added indicators
        """
        if len(data) < self.min_required_candles:
            DBLogger.log_event("WARNING",
                               f"Insufficient data for MA calculations. "
                               f"Need at least {self.min_required_candles} candles.",
                               "TripleMovingAverageStrategy")
            return data

        # Calculate Fast EMA (10 period)
        data['fast_ema'] = data['close'].ewm(span=self.fast_period, adjust=False).mean()

        # Calculate Medium SMA (50 period)
        data['medium_sma'] = data['close'].rolling(window=self.medium_period).mean()

        # Calculate Slow SMA (200 period)
        data['slow_sma'] = data['close'].rolling(window=self.slow_period).mean()

        # Determine trend filter based on Medium SMA vs Slow SMA
        data['trend_filter'] = np.where(
            data['medium_sma'] > data['slow_sma'],
            1,  # Uptrend
            np.where(
                data['medium_sma'] < data['slow_sma'],
                -1,  # Downtrend
                0  # Neutral (when exactly equal)
            )
        )

        # Calculate crossover signals between Fast EMA and Medium SMA
        data['ema_above_sma'] = data['fast_ema'] > data['medium_sma']

        # Detect crossovers (changes in ema_above_sma status)
        shifted_ema_above = data['ema_above_sma'].shift(1)

        # Create boolean series without using fillna directly
        # Replace NaN values with False using numpy where
        has_na = shifted_ema_above.isna()
        filled_shifted_ema_above = np.where(has_na, False, shifted_ema_above)

        data['cross_up'] = (data['ema_above_sma'] & ~filled_shifted_ema_above).astype(int)
        data['cross_down'] = (~data['ema_above_sma'] & filled_shifted_ema_above).astype(int)

        # Calculate trade signals based on crossovers and trend filter
        data['signal'] = 0  # Initialize with no signal

        # Identify buy signals: Cross up AND uptrend
        data.loc[(data['cross_up'] == 1) & (data['trend_filter'] == 1), 'signal'] = 1

        # Identify sell signals: Cross down AND downtrend
        data.loc[(data['cross_down'] == 1) & (data['trend_filter'] == -1), 'signal'] = -1

        # Add ATR for stop loss and take profit calculations
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        return data

    def analyze(self, data):
        """Analyze market data and generate trading signals.

        Args:
            data (pd.DataFrame): OHLC data with or without indicators

        Returns:
            list: Generated trading signals
        """
        # Calculate indicators if not already present
        if 'signal' not in data.columns:
            data = self.calculate_indicators(data)

        # Check if we have sufficient data after calculations
        if data.empty or 'signal' not in data.columns:
            DBLogger.log_event("WARNING",
                               "Insufficient data for analysis after calculations",
                               "TripleMovingAverageStrategy")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]
        previous_candle = data.iloc[-2] if len(data) > 1 else None

        # Check for buy signal (fast EMA crosses above medium SMA in uptrend)
        if last_candle['signal'] == 1:
            # Create BUY signal
            signal = self._generate_buy_signal(data, last_candle)
            if signal:
                signals.append(signal)
                DBLogger.log_event("INFO",
                                   f"Generated BUY signal for {self.symbol} at {last_candle['close']:.2f}. "
                                   f"Fast EMA: {last_candle['fast_ema']:.2f}, Medium SMA: {last_candle['medium_sma']:.2f}, "
                                   f"Slow SMA: {last_candle['slow_sma']:.2f}",
                                   "TripleMovingAverageStrategy")

        # Check for sell signal (fast EMA crosses below medium SMA in downtrend)
        elif last_candle['signal'] == -1:
            # Create SELL signal
            signal = self._generate_sell_signal(data, last_candle)
            if signal:
                signals.append(signal)
                DBLogger.log_event("INFO",
                                   f"Generated SELL signal for {self.symbol} at {last_candle['close']:.2f}. "
                                   f"Fast EMA: {last_candle['fast_ema']:.2f}, Medium SMA: {last_candle['medium_sma']:.2f}, "
                                   f"Slow SMA: {last_candle['slow_sma']:.2f}",
                                   "TripleMovingAverageStrategy")

        # Check for exit conditions
        if previous_candle is not None:
            # Exit long if fast EMA crosses below medium SMA (regardless of trend filter)
            if last_candle['cross_down'] == 1:
                signal = self._generate_exit_signal(data, last_candle, "CLOSE")
                if signal:
                    signals.append(signal)
                    DBLogger.log_event("INFO",
                                       f"Generated EXIT signal for {self.symbol} at {last_candle['close']:.2f}. "
                                       f"Fast EMA crossed below Medium SMA.",
                                       "TripleMovingAverageStrategy")

            # Exit short if fast EMA crosses above medium SMA (regardless of trend filter)
            elif last_candle['cross_up'] == 1:
                signal = self._generate_exit_signal(data, last_candle, "CLOSE")
                if signal:
                    signals.append(signal)
                    DBLogger.log_event("INFO",
                                       f"Generated EXIT signal for {self.symbol} at {last_candle['close']:.2f}. "
                                       f"Fast EMA crossed above Medium SMA.",
                                       "TripleMovingAverageStrategy")

        return signals

    def _generate_buy_signal(self, data, last_candle):
        """Generate a buy signal with appropriate stop loss and take profit.

        Args:
            data (pd.DataFrame): OHLC data with indicators
            last_candle (pd.Series): Last complete candle

        Returns:
            StrategySignal: The generated buy signal or None if error
        """
        try:
            entry_price = last_candle['close']

            # Calculate stop loss based on ATR
            atr = last_candle['atr']
            stop_loss = entry_price - (atr * 2)  # 2 ATR below entry

            # Calculate take profit using risk-reward ratio
            risk = entry_price - stop_loss
            take_profit_1r = entry_price + risk  # 1:1 ratio
            take_profit_2r = entry_price + (risk * 2)  # 2:1 ratio

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=0.8,  # High confidence due to trend filter
                metadata={
                    'fast_ema': float(last_candle['fast_ema']),
                    'medium_sma': float(last_candle['medium_sma']),
                    'slow_sma': float(last_candle['slow_sma']),
                    'trend_filter': int(last_candle['trend_filter']),
                    'stop_loss': float(stop_loss),
                    'take_profit_1r': float(take_profit_1r),
                    'take_profit_2r': float(take_profit_2r),
                    'atr': float(atr),
                    'reason': 'Fast EMA crossed above Medium SMA with uptrend filter (Medium SMA > Slow SMA)'
                }
            )

            return signal
        except Exception as e:
            DBLogger.log_error("TripleMovingAverageStrategy",
                               f"Error generating BUY signal: {str(e)}",
                               exception=e)
            return None

    def _generate_sell_signal(self, data, last_candle):
        """Generate a sell signal with appropriate stop loss and take profit.

        Args:
            data (pd.DataFrame): OHLC data with indicators
            last_candle (pd.Series): Last complete candle

        Returns:
            StrategySignal: The generated sell signal or None if error
        """
        try:
            entry_price = last_candle['close']

            # Calculate stop loss based on ATR
            atr = last_candle['atr']
            stop_loss = entry_price + (atr * 2)  # 2 ATR above entry

            # Calculate take profit using risk-reward ratio
            risk = stop_loss - entry_price
            take_profit_1r = entry_price - risk  # 1:1 ratio
            take_profit_2r = entry_price - (risk * 2)  # 2:1 ratio

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=0.8,  # High confidence due to trend filter
                metadata={
                    'fast_ema': float(last_candle['fast_ema']),
                    'medium_sma': float(last_candle['medium_sma']),
                    'slow_sma': float(last_candle['slow_sma']),
                    'trend_filter': int(last_candle['trend_filter']),
                    'stop_loss': float(stop_loss),
                    'take_profit_1r': float(take_profit_1r),
                    'take_profit_2r': float(take_profit_2r),
                    'atr': float(atr),
                    'reason': 'Fast EMA crossed below Medium SMA with downtrend filter (Medium SMA < Slow SMA)'
                }
            )

            return signal
        except Exception as e:
            DBLogger.log_error("TripleMovingAverageStrategy",
                               f"Error generating SELL signal: {str(e)}",
                               exception=e)
            return None

    def _generate_exit_signal(self, data, last_candle, exit_type="CLOSE"):
        """Generate an exit signal.

        Args:
            data (pd.DataFrame): OHLC data with indicators
            last_candle (pd.Series): Last complete candle
            exit_type (str): Type of exit signal

        Returns:
            StrategySignal: The generated exit signal or None if error
        """
        try:
            exit_price = last_candle['close']

            signal = self.create_signal(
                signal_type=exit_type,
                price=exit_price,
                strength=0.9,  # High confidence for exits
                metadata={
                    'fast_ema': float(last_candle['fast_ema']),
                    'medium_sma': float(last_candle['medium_sma']),
                    'slow_sma': float(last_candle['slow_sma']),
                    'trend_filter': int(last_candle['trend_filter']),
                    'reason': 'Exit signal based on Fast EMA and Medium SMA crossover'
                }
            )

            return signal
        except Exception as e:
            DBLogger.log_error("TripleMovingAverageStrategy",
                               f"Error generating EXIT signal: {str(e)}",
                               exception=e)
            return None