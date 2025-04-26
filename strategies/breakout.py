# strategies/breakout.py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from custom_logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class BreakoutStrategy(BaseStrategy):
    """Support/Resistance Breakout Strategy for XAU/USD."""

    def __init__(self, symbol="XAUUSD", timeframe="M15",
                 lookback_periods=48, min_range_bars=10,
                 volume_threshold=1.5, data_fetcher=None):
        """Initialize the Breakout strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "M15".
            lookback_periods (int, optional): Number of bars to look back for range. Defaults to 48.
            min_range_bars (int, optional): Minimum bars to confirm a range. Defaults to 10.
            volume_threshold (float, optional): Volume multiplier to confirm breakout. Defaults to 1.5.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Breakout", data_fetcher=data_fetcher)

        # Validate inputs
        if lookback_periods < min_range_bars:
            raise ValueError(f"lookback_periods ({lookback_periods}) must be >= min_range_bars ({min_range_bars})")

        self.lookback_periods = lookback_periods
        self.min_range_bars = min_range_bars
        self.volume_threshold = volume_threshold

        # Ensure we fetch enough data for calculations
        self.min_required_candles = lookback_periods + 20

        self.logger.info(
            f"Initialized Breakout strategy: {symbol} {timeframe}, "
            f"Lookback: {lookback_periods}, Min Range Bars: {min_range_bars}"
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
                f"Insufficient data for breakout calculations. "
                f"Need at least {self.min_required_candles} candles."
            )
            return data

        # Calculate Average True Range (ATR) for volatility
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Calculate volume moving average
        data['volume_ma'] = data['volume'].rolling(window=20).mean()

        # Calculate Bollinger Bands for volatility contraction signal
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        data['middle_band'] = typical_price.rolling(window=20).mean()
        price_std = typical_price.rolling(window=20).std()
        data['upper_band'] = data['middle_band'] + (price_std * 2)
        data['lower_band'] = data['middle_band'] - (price_std * 2)
        data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['middle_band']

        # Identify potential ranges
        data = self._identify_ranges(data)

        # Identify breakout signals
        data = self._identify_breakouts(data)

        return data

    def _identify_ranges(self, data):
        """Identify potential trading ranges in the data.

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with range information
        """
        # Initialize range columns
        data['in_range'] = False
        data['range_top'] = np.nan
        data['range_bottom'] = np.nan
        data['range_bars'] = 0

        # We'll use rolling windows to detect ranges
        for i in range(self.min_range_bars, len(data)):
            # Select the lookback window
            window = data.iloc[max(0, i - self.lookback_periods):i]

            # Criteria for a range:
            # 1. Price is bounded within a tight range relative to ATR
            # 2. No strong directional movement (high-low bounded)
            # 3. Minimum number of bars

            # Check if volatility is contracting (BB width narrowing)
            bb_width_now = data.iloc[i - 1]['bb_width']
            bb_width_past = data.iloc[max(0, i - 10)]['bb_width'] if i >= 10 else bb_width_now

            volatility_contracting = bb_width_now < bb_width_past

            # Calculate the high and low of the window
            window_high = window['high'].max()
            window_low = window['low'].min()

            # Range height as percentage
            range_height_pct = (window_high - window_low) / window_low

            # Range height compared to average ATR
            avg_atr = window['atr'].mean()
            range_atr_ratio = (window_high - window_low) / (avg_atr * 5)  # 5 ATRs is arbitrary threshold

            # Determine if we're in a range
            # A range is identified if:
            # - Range height is less than 2% of price (for gold)
            # - Range has lasted for minimum number of bars
            # - Volatility is contracting

            is_range = (range_height_pct < 0.02 or range_atr_ratio < 1.0) and volatility_contracting

            if is_range:
                data.loc[data.index[i], 'in_range'] = True
                data.loc[data.index[i], 'range_top'] = window_high
                data.loc[data.index[i], 'range_bottom'] = window_low

                # Count how many bars we've been in this range
                if i > 0 and data.iloc[i - 1]['in_range']:
                    prev_top = data.iloc[i - 1]['range_top']
                    prev_bottom = data.iloc[i - 1]['range_bottom']

                    # If the range boundaries are similar, increment the count
                    if (abs(window_high - prev_top) / prev_top < 0.003 and
                            abs(window_low - prev_bottom) / prev_bottom < 0.003):
                        data.loc[data.index[i], 'range_bars'] = data.iloc[i - 1]['range_bars'] + 1
                    else:
                        data.loc[data.index[i], 'range_bars'] = 1
                else:
                    data.loc[data.index[i], 'range_bars'] = 1

        return data

    def _identify_breakouts(self, data):
        """Identify potential breakout signals.

        Args:
            data (pandas.DataFrame): OHLC data with range information

        Returns:
            pandas.DataFrame: Data with breakout signals
        """
        # Initialize breakout columns
        data['breakout_signal'] = 0  # 0: No signal, 1: Bullish, -1: Bearish
        data['breakout_strength'] = 0.0
        data['breakout_stop_loss'] = np.nan

        for i in range(1, len(data)):
            # Skip if not enough prior data
            if i < 2:
                continue

            # We need to have been in a range in the previous bar
            if not data.iloc[i - 1]['in_range'] or data.iloc[i - 1]['range_bars'] < self.min_range_bars:
                continue

            # Get range boundaries
            range_top = data.iloc[i - 1]['range_top']
            range_bottom = data.iloc[i - 1]['range_bottom']

            # Current and previous candle info
            current_close = data.iloc[i]['close']
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            current_volume = data.iloc[i]['volume']
            prev_volume_ma = data.iloc[i - 1]['volume_ma']

            # Get ATR for volatility reference
            current_atr = data.iloc[i]['atr']

            # Check for breakout conditions
            # 1. Price closes beyond the range
            # 2. With increased volume (if available)
            # 3. With enough momentum (candle size)

            # Bullish breakout
            if (current_close > range_top and
                    current_volume > prev_volume_ma * self.volume_threshold and
                    (current_close - current_low) > current_atr * 0.5):

                # Calculate breakout strength (1.0 = very strong)
                strength = min(1.0, (current_close - range_top) / (range_top * 0.01))

                data.loc[data.index[i], 'breakout_signal'] = 1
                data.loc[data.index[i], 'breakout_strength'] = strength
                # Stop loss just below the breakout level (the range top)
                data.loc[data.index[i], 'breakout_stop_loss'] = range_top - (current_atr * 0.5)

            # Bearish breakout
            elif (current_close < range_bottom and
                  current_volume > prev_volume_ma * self.volume_threshold and
                  (current_high - current_close) > current_atr * 0.5):

                # Calculate breakout strength (1.0 = very strong)
                strength = min(1.0, (range_bottom - current_close) / (range_bottom * 0.01))

                data.loc[data.index[i], 'breakout_signal'] = -1
                data.loc[data.index[i], 'breakout_strength'] = strength
                # Stop loss just above the breakout level (the range bottom)
                data.loc[data.index[i], 'breakout_stop_loss'] = range_bottom + (current_atr * 0.5)

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
        if data.empty or 'breakout_signal' not in data.columns:
            self.logger.warning("Insufficient data for analysis after calculations")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]

        # Check for breakout signal on the last candle
        if last_candle['breakout_signal'] == 1:  # Bullish breakout
            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = last_candle['breakout_stop_loss']

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate potential take profit based on the range height
            range_height = last_candle['range_top'] - last_candle['range_bottom']
            take_profit_1r = entry_price + (entry_price - stop_loss)  # 1:1 risk:reward
            take_profit_extension = entry_price + range_height  # Project the range height

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['breakout_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1r': take_profit_1r,
                    'take_profit_extension': take_profit_extension,
                    'range_top': last_candle['range_top'],
                    'range_bottom': last_candle['range_bottom'],
                    'range_bars': last_candle['range_bars'],
                    'atr': last_candle['atr'],
                    'reason': 'Bullish breakout from consolidation range'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Breakout above {last_candle['range_top']:.2f} after "
                f"{last_candle['range_bars']} bars in range."
            )

        elif last_candle['breakout_signal'] == -1:  # Bearish breakout
            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['breakout_stop_loss']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate potential take profit based on the range height
            range_height = last_candle['range_top'] - last_candle['range_bottom']
            take_profit_1r = entry_price - (stop_loss - entry_price)  # 1:1 risk:reward
            take_profit_extension = entry_price - range_height  # Project the range height

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['breakout_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1r': take_profit_1r,
                    'take_profit_extension': take_profit_extension,
                    'range_top': last_candle['range_top'],
                    'range_bottom': last_candle['range_bottom'],
                    'range_bars': last_candle['range_bars'],
                    'atr': last_candle['atr'],
                    'reason': 'Bearish breakout from consolidation range'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Breakout below {last_candle['range_bottom']:.2f} after "
                f"{last_candle['range_bars']} bars in range."
            )

        return signals