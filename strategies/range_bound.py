# strategies/range_bound.py - Updated version

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from custom_logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class RangeBoundStrategy(BaseStrategy):
    """Range-Bound Mean Reversion Strategy for XAU/USD.

    This strategy identifies periods when price is trading within a horizontal range and
    takes mean-reversion trades at the edges of the range (buy at support, sell at resistance)
    with confirmation from oscillators (RSI) and trend filters (ADX).
    """

    def __init__(self, symbol="XAUUSD", timeframe="M15",
                 lookback_periods=48, min_range_bars=10,
                 rsi_period=14, rsi_overbought=70, rsi_oversold=30,
                 adx_period=14, adx_threshold=20, data_fetcher=None):
        """Initialize the Range-Bound Mean Reversion strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "M15".
            lookback_periods (int, optional): Number of bars to look back for range. Defaults to 48.
            min_range_bars (int, optional): Minimum bars to confirm a range. Defaults to 10.
            rsi_period (int, optional): Period for RSI calculation. Defaults to 14.
            rsi_overbought (int, optional): RSI threshold for overbought. Defaults to 70.
            rsi_oversold (int, optional): RSI threshold for oversold. Defaults to 30.
            adx_period (int, optional): Period for ADX calculation. Defaults to 14.
            adx_threshold (int, optional): ADX threshold for trending market. Defaults to 20.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Range_Mean_Reversion", data_fetcher=data_fetcher)

        # Validate inputs
        if lookback_periods < min_range_bars:
            raise ValueError(f"lookback_periods ({lookback_periods}) must be >= min_range_bars ({min_range_bars})")
        if rsi_overbought <= rsi_oversold:
            raise ValueError(f"rsi_overbought ({rsi_overbought}) must be > rsi_oversold ({rsi_oversold})")

        self.lookback_periods = lookback_periods
        self.min_range_bars = min_range_bars
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        # Ensure we fetch enough data for calculations
        self.min_required_candles = lookback_periods + max(rsi_period, adx_period) + 20

        self.logger.info(
            f"Initialized Range-Bound Mean Reversion strategy: {symbol} {timeframe}, "
            f"RSI: {rsi_period}/{rsi_oversold}/{rsi_overbought}, ADX threshold: {adx_threshold}"
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
                f"Insufficient data for range-bound calculations. "
                f"Need at least {self.min_required_candles} candles."
            )
            return data

        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Calculate RSI
        self.logger.debug("Calculating RSI...")
        # Calculate price differences
        delta = df['close'].diff()

        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        # Calculate relative strength
        rs = avg_gain / avg_loss

        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        self.logger.debug("Calculating Bollinger Bands...")
        # Calculate 20-period SMA as middle band
        df['middle_band'] = df['close'].rolling(window=20).mean()

        # Calculate standard deviation
        df['std_dev'] = df['close'].rolling(window=20).std()

        # Calculate upper and lower bands
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

        # Calculate BB width (normalized by middle band)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

        # Calculate ADX and directional indicators
        self.logger.debug("Calculating ADX...")
        df = self._calculate_adx(df)

        # Identify potential ranges in the data
        self.logger.debug("Identifying price ranges...")
        df = self._identify_ranges(df)

        # Identify entry signals at range extremes with oscillator confirmation
        self.logger.debug("Identifying entry signals...")
        df = self._identify_entry_signals(df)

        return df

    def analyze(self, data):
        """Analyze market data and generate trading signals.

        Args:
            data (pandas.DataFrame): OHLC data for analysis

        Returns:
            list: Generated trading signals
        """
        # Check if data is None or empty
        if data is None or len(data) == 0:
            self.logger.warning("No data provided for range-bound analysis")
            return []

        # Calculate indicators
        try:
            data = self.calculate_indicators(data)
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return []

        # Check if data is None or empty after calculations
        if data is None or (hasattr(data, 'empty') and data.empty):
            self.logger.warning("Insufficient data for range-bound analysis after calculations")
            return []

        # Check if required columns exist
        if 'signal' not in data.columns:
            self.logger.warning("Required column 'signal' not found in processed data")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]

        # Check for trading signal on the last candle
        if last_candle['signal'] == 1:  # Buy signal
            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit = last_candle['take_profit']

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                self.logger.warning(
                    f"Invalid stop loss ({stop_loss}) for BUY signal - must be below entry price ({entry_price})")
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate extended take profit (full range)
            range_top = last_candle['range_top']
            range_bottom = last_candle['range_bottom']
            take_profit_full = range_top * 0.997  # Just below the top

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_midpoint': take_profit,
                    'take_profit_full': take_profit_full,
                    'range_top': range_top,
                    'range_bottom': range_bottom,
                    'range_bars': last_candle['range_bars'],
                    'rsi': last_candle['rsi'],
                    'adx': last_candle['adx'],
                    'reason': 'Buy at support in range with oversold RSI'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Support around {range_bottom:.2f}, RSI: {last_candle['rsi']:.1f}, "
                f"ADX: {last_candle['adx']:.1f}."
            )

        elif last_candle['signal'] == -1:  # Sell signal
            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit = last_candle['take_profit']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                self.logger.warning(
                    f"Invalid stop loss ({stop_loss}) for SELL signal - must be above entry price ({entry_price})")
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate extended take profit (full range)
            range_top = last_candle['range_top']
            range_bottom = last_candle['range_bottom']
            take_profit_full = range_bottom * 1.003  # Just above the bottom

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_midpoint': take_profit,
                    'take_profit_full': take_profit_full,
                    'range_top': range_top,
                    'range_bottom': range_bottom,
                    'range_bars': last_candle['range_bars'],
                    'rsi': last_candle['rsi'],
                    'adx': last_candle['adx'],
                    'reason': 'Sell at resistance in range with overbought RSI'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Resistance around {range_top:.2f}, RSI: {last_candle['rsi']:.1f}, "
                f"ADX: {last_candle['adx']:.1f}."
            )

        return signals

    def _calculate_adx(self, data):
        """Calculate Average Directional Index (ADX).

        Args:
            data (pandas.DataFrame): OHLC data

        Returns:
            pandas.DataFrame: Data with ADX indicators
        """
        # Calculate +DI and -DI
        high_diff = data['high'].diff()
        low_diff = data['low'].diff().multiply(-1)

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        # Calculate True Range
        tr1 = abs(data['high'] - data['low'])
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['tr'] = tr

        # Smooth the TR and directional movement
        period = self.adx_period

        # Use exponential smoothing for TR and DM
        tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean()

        # Calculate directional indicators
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate directional index (DX)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Calculate ADX (smoothed DX)
        data['adx'] = dx.ewm(alpha=1 / period, adjust=False).mean()
        data['plus_di'] = plus_di
        data['minus_di'] = minus_di

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
        data['range_midpoint'] = np.nan

        # Define when we're in a range:
        # 1. ADX below threshold (non-trending)
        # 2. Bollinger Band width relatively narrow
        # 3. Price oscillating within defined boundaries

        for i in range(self.min_range_bars, len(data)):
            # Skip if ADX is not calculated yet or is NaN
            if i <= 0 or 'adx' not in data.columns or np.isnan(data.iloc[i - 1]['adx']):
                continue

            # Check if ADX indicates a non-trending market
            if data.iloc[i - 1]['adx'] < self.adx_threshold:
                # Select the lookback window
                start_idx = max(0, i - self.lookback_periods)
                window = data.iloc[start_idx:i]

                # Calculate potential range boundaries
                if len(window) > 0:
                    window_high = window['high'].max()
                    window_low = window['low'].min()

                    # Range height as percentage
                    if window_low > 0:  # Prevent division by zero
                        range_height_pct = (window_high - window_low) / window_low
                    else:
                        range_height_pct = float('inf')  # Set to a large value if window_low is zero or negative

                    # Check Bollinger Band width (narrowing bands suggest consolidation)
                    bb_width_now = data.iloc[i - 1]['bb_width'] if 'bb_width' in data.columns and i > 0 else np.nan
                    bb_width_past = data.iloc[max(0, i - 10)][
                        'bb_width'] if 'bb_width' in data.columns and i >= 10 else np.nan

                    # Default to true if we don't have BB width data for testing
                    volatility_contained = True
                    if not np.isnan(bb_width_now) and not np.isnan(bb_width_past) and bb_width_past > 0:
                        volatility_contained = bb_width_now <= bb_width_past * 1.2  # Not expanding rapidly

                    # Determine if we're in a range
                    # A range is identified if:
                    # - Range height is relatively small (less than 2% for gold)
                    # - ADX is below threshold (non-trending)
                    # - BB width is not expanding rapidly

                    is_range = (range_height_pct < 0.02) and volatility_contained

                    if is_range:
                        data.loc[data.index[i], 'in_range'] = True
                        data.loc[data.index[i], 'range_top'] = window_high
                        data.loc[data.index[i], 'range_bottom'] = window_low
                        data.loc[data.index[i], 'range_midpoint'] = (window_high + window_low) / 2

                        # Count how many bars we've been in this range
                        if i > 0 and data.iloc[i - 1]['in_range']:
                            prev_top = data.iloc[i - 1]['range_top']
                            prev_bottom = data.iloc[i - 1]['range_bottom']

                            # If the range boundaries are similar, increment the count
                            if (not np.isnan(prev_top) and not np.isnan(
                                    prev_bottom) and prev_top > 0 and prev_bottom > 0 and
                                    abs(window_high - prev_top) / prev_top < 0.003 and
                                    abs(window_low - prev_bottom) / prev_bottom < 0.003):
                                data.loc[data.index[i], 'range_bars'] = data.iloc[i - 1]['range_bars'] + 1
                            else:
                                data.loc[data.index[i], 'range_bars'] = 1
                        else:
                            data.loc[data.index[i], 'range_bars'] = 1

        # Add logging to help debug test failures
        range_count = data['in_range'].sum()
        self.logger.debug(f"Identified {range_count} bars in range conditions")

        return data

    def _identify_entry_signals(self, data):
        """Identify entry signals at range extremes with oscillator confirmation.

        Args:
            data (pandas.DataFrame): OHLC data with range information

        Returns:
            pandas.DataFrame: Data with signal information
        """
        # Initialize signal columns
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data['signal_strength'] = 0.0
        data['stop_loss'] = np.nan
        data['take_profit'] = np.nan

        for i in range(2, len(data)):
            # Skip if not enough prior data
            if i < 3:
                continue

            # We need to have been in a range for sufficient time
            if not data.iloc[i]['in_range'] or data.iloc[i]['range_bars'] < self.min_range_bars:
                continue

            # Get range boundaries
            range_top = data.iloc[i]['range_top']
            range_bottom = data.iloc[i]['range_bottom']
            range_midpoint = data.iloc[i]['range_midpoint']

            # Current price and indicator values
            current_close = data.iloc[i]['close']
            current_rsi = data.iloc[i]['rsi']
            current_adx = data.iloc[i]['adx']

            # Calculate range proximity - how close we are to boundaries
            # Initialize both proximity variables with default values
            proximity_to_bottom = 0.0
            proximity_to_top = 0.0

            if current_close < range_midpoint:
                # Lower half of range
                if not np.isnan(range_midpoint) and not np.isnan(range_bottom) and (range_midpoint - range_bottom) > 0:
                    proximity_to_bottom = (range_midpoint - current_close) / (range_midpoint - range_bottom)
            else:
                # Upper half of range
                if not np.isnan(range_midpoint) and not np.isnan(range_top) and (range_top - range_midpoint) > 0:
                    proximity_to_top = (current_close - range_midpoint) / (range_top - range_midpoint)

            # Buy Signal Conditions:
            # 1. Price is near the lower boundary of the range
            # 2. RSI is oversold
            # 3. ADX is below threshold (non-trending)
            if (current_close < range_midpoint and
                    proximity_to_bottom > 0.7 and  # Close to bottom
                    not np.isnan(current_rsi) and current_rsi <= self.rsi_oversold and
                    not np.isnan(current_adx) and current_adx < self.adx_threshold):

                # Calculate signal strength (1.0 = very strong)
                strength = min(1.0, proximity_to_bottom * (self.rsi_oversold - current_rsi) / 10)

                # Set a stop loss just below the range bottom
                stop_loss = range_bottom * 0.997  # 0.3% below range bottom

                # Set take profit at range midpoint or higher
                take_profit_midpoint = range_midpoint
                take_profit_full = range_top * 0.997  # Just below the top

                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit_midpoint  # Using midpoint for main TP

            # Sell Signal Conditions:
            # 1. Price is near the upper boundary of the range
            # 2. RSI is overbought
            # 3. ADX is below threshold (non-trending)
            elif (current_close > range_midpoint and
                  proximity_to_top > 0.7 and  # Close to top
                  not np.isnan(current_rsi) and current_rsi >= self.rsi_overbought and
                  not np.isnan(current_adx) and current_adx < self.adx_threshold):

                # Calculate signal strength (1.0 = very strong)
                strength = min(1.0, proximity_to_top * (current_rsi - self.rsi_overbought) / 10)

                # Set a stop loss just above the range top
                stop_loss = range_top * 1.003  # 0.3% above range top

                # Set take profit at range midpoint or lower
                take_profit_midpoint = range_midpoint
                take_profit_full = range_bottom * 1.003  # Just above the bottom

                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit_midpoint  # Using midpoint for main TP

        return data