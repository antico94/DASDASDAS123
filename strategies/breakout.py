# strategies/breakout.py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from custom_logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class BreakoutStrategy(BaseStrategy):
    """Support/Resistance Breakout Strategy for XAU/USD.

    This strategy seeks to profit from explosive moves when gold breaks out
    of a consolidation range or key support/resistance level, with volume
    confirmation and proper ATR-based risk management.
    """

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
            f"Lookback: {lookback_periods}, Min Range Bars: {min_range_bars}, "
            f"Volume Threshold: {volume_threshold}"
        )

    def calculate_indicators(self, data):
        """Calculate strategy indicators on OHLC data."""
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

        # Calculate volume moving average for volume confirmation
        data['volume_ma'] = data['volume'].rolling(window=20).mean()

        # Calculate volume ratio for confirmation
        data['volume_ratio'] = data['volume'] / data['volume_ma']

        # Calculate Bollinger Bands for volatility contraction signal (squeeze)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        data['middle_band'] = typical_price.rolling(window=20).mean()
        price_std = typical_price.rolling(window=20).std()
        data['upper_band'] = data['middle_band'] + (price_std * 2)
        data['lower_band'] = data['middle_band'] - (price_std * 2)
        data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['middle_band']

        # Calculate BB width change to detect squeeze
        data['bb_width_change'] = data['bb_width'] / data['bb_width'].shift(10) - 1

        # Detect BB squeeze (narrowing bands)
        data['bb_squeeze'] = (data['bb_width_change'] < -0.1)

        # Identify potential ranges
        data = self._identify_ranges(data)

        # Identify breakout signals with volume confirmation
        data = self._identify_breakouts(data)

        return data

    def _identify_ranges(self, data):
        """Identify potential trading ranges in the data."""
        # Initialize range columns
        data['in_range'] = False
        data['range_top'] = np.nan
        data['range_bottom'] = np.nan
        data['range_bars'] = 0

        # We'll use rolling windows to detect ranges
        for i in range(self.min_range_bars, len(data)):
            # Select the lookback window
            window = data.iloc[max(0, i - self.lookback_periods):i]

            # Calculate the high and low of the window
            window_high = window['high'].max()
            window_low = window['low'].min()

            # Range height as percentage
            range_height_pct = (window_high - window_low) / window_low

            # Check if volatility is contracting (BB width narrowing)
            bb_width_now = data.iloc[i - 1]['bb_width']
            bb_width_past = data.iloc[max(0, i - 10)]['bb_width'] if i >= 10 else bb_width_now

            volatility_contracting = bb_width_now < bb_width_past * 0.9  # 10% narrower than 10 bars ago

            # Determine if we're in a range
            # Per the plan: "a period of low volatility (narrow Bollinger Bands or low ATR) often precedes a breakout"
            is_range = (range_height_pct < 0.015 and  # Tight range for gold (1.5%)
                        volatility_contracting and  # Bollinger Band squeeze
                        data.iloc[i - 1]['bb_squeeze'])  # Confirmed squeeze condition

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
        """Identify potential breakout signals with volume confirmation."""
        # Initialize breakout columns
        data['breakout_signal'] = 0  # 0: No signal, 1: Bullish, -1: Bearish
        data['breakout_strength'] = 0.0
        data['breakout_stop_loss'] = np.nan
        data['breakout_volume_confirmed'] = False
        data['breakout_candle_strength'] = 0.0

        for i in range(1, len(data)):
            # Skip if not enough prior data
            if i < 3:
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
            current_volume_ratio = data.iloc[i]['volume_ratio']
            prev_volume_ma = data.iloc[i - 1]['volume_ma']

            # Get ATR for volatility reference
            current_atr = data.iloc[i]['atr']

            # Calculate candle size relative to ATR
            candle_size = abs(current_high - current_low)
            candle_strength = candle_size / current_atr

            # Volume confirmation - per the plan, we need "increased volume at the break"
            volume_confirmed = current_volume_ratio > self.volume_threshold

            # Check for breakout conditions
            # 1. Price closes beyond the range with enough momentum
            # 2. With increased volume (volume confirmation)
            # 3. With enough momentum (candle size)

            # Bullish breakout with volume confirmation
            if (current_close > range_top and
                    volume_confirmed and  # Volume confirmation
                    candle_strength > 1.0):  # Candle is larger than ATR

                # Calculate breakout strength (1.0 = very strong)
                # Based on how far price broke out and volume increase
                price_breakout = (current_close - range_top) / (range_top * 0.01)  # % beyond range
                strength = min(1.0, price_breakout * 0.5 + (current_volume_ratio - 1) * 0.5)

                data.loc[data.index[i], 'breakout_signal'] = 1
                data.loc[data.index[i], 'breakout_strength'] = strength
                data.loc[data.index[i], 'breakout_volume_confirmed'] = volume_confirmed
                data.loc[data.index[i], 'breakout_candle_strength'] = candle_strength

                # Critical fix: use ATR-based stop per the plan:
                # "stop loss to 1.5×ATR below entry for bullish breakouts"
                data.loc[data.index[i], 'breakout_stop_loss'] = current_close - (current_atr * 1.5)

            # Bearish breakout with volume confirmation
            elif (current_close < range_bottom and
                  volume_confirmed and  # Volume confirmation
                  candle_strength > 1.0):  # Candle is larger than ATR

                # Calculate breakout strength
                price_breakout = (range_bottom - current_close) / (range_bottom * 0.01)  # % beyond range
                strength = min(1.0, price_breakout * 0.5 + (current_volume_ratio - 1) * 0.5)

                data.loc[data.index[i], 'breakout_signal'] = -1
                data.loc[data.index[i], 'breakout_strength'] = strength
                data.loc[data.index[i], 'breakout_volume_confirmed'] = volume_confirmed
                data.loc[data.index[i], 'breakout_candle_strength'] = candle_strength

                # Critical fix: use ATR-based stop per the plan:
                # "stop loss to 1.5×ATR above entry for bearish breakouts"
                data.loc[data.index[i], 'breakout_stop_loss'] = current_close + (current_atr * 1.5)

        return data

    def analyze(self, data):
        """Analyze market data and generate trading signals."""
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

            # Use ATR-based stop loss as specified in the plan
            stop_loss = last_candle['breakout_stop_loss']

            # Calculate take profit targets based on ATR as specified in the plan
            # "take half off at 1×ATR profit and move stop to breakeven"
            atr = last_candle['atr']
            take_profit_1atr = entry_price + atr  # First target at 1×ATR

            # "take the rest off at 2×ATR or trail by 1.5×ATR"
            take_profit_2atr = entry_price + (atr * 2)  # Second target at 2×ATR

            # Alternative take profit based on range projection
            range_height = last_candle['range_top'] - last_candle['range_bottom']
            take_profit_projection = entry_price + range_height

            # Use the smaller of the two projections
            take_profit_2 = min(take_profit_2atr, take_profit_projection)

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['breakout_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1atr': take_profit_1atr,  # 1×ATR target
                    'take_profit_2atr': take_profit_2atr,  # 2×ATR target
                    'take_profit_projection': take_profit_projection,  # Range projection target
                    'take_profit_1r': take_profit_1atr,  # For compatibility with order manager
                    'take_profit_2r': take_profit_2,  # For compatibility with order manager
                    'range_top': last_candle['range_top'],
                    'range_bottom': last_candle['range_bottom'],
                    'range_bars': last_candle['range_bars'],
                    'atr': atr,
                    'volume_confirmed': bool(last_candle['breakout_volume_confirmed']),
                    'candle_strength': last_candle['breakout_candle_strength'],
                    'reason': 'Bullish breakout from consolidation range with volume confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Breakout above {last_candle['range_top']:.2f} after "
                f"{last_candle['range_bars']} bars in range. "
                f"Volume ratio: {last_candle['volume_ratio']:.2f}, ATR: {atr:.2f}"
            )

        elif last_candle['breakout_signal'] == -1:  # Bearish breakout
            # Create SELL signal
            entry_price = last_candle['close']

            # Use ATR-based stop loss as specified in the plan
            stop_loss = last_candle['breakout_stop_loss']

            # Calculate take profit targets based on ATR as specified in the plan
            atr = last_candle['atr']
            take_profit_1atr = entry_price - atr  # First target at 1×ATR
            take_profit_2atr = entry_price - (atr * 2)  # Second target at 2×ATR

            # Alternative take profit based on range projection
            range_height = last_candle['range_top'] - last_candle['range_bottom']
            take_profit_projection = entry_price - range_height

            # Use the larger of the two projections (as price is going down)
            take_profit_2 = max(take_profit_2atr, take_profit_projection)

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['breakout_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1atr': take_profit_1atr,  # 1×ATR target
                    'take_profit_2atr': take_profit_2atr,  # 2×ATR target
                    'take_profit_projection': take_profit_projection,  # Range projection target
                    'take_profit_1r': take_profit_1atr,  # For compatibility with order manager
                    'take_profit_2r': take_profit_2,  # For compatibility with order manager
                    'range_top': last_candle['range_top'],
                    'range_bottom': last_candle['range_bottom'],
                    'range_bars': last_candle['range_bars'],
                    'atr': atr,
                    'volume_confirmed': bool(last_candle['breakout_volume_confirmed']),
                    'candle_strength': last_candle['breakout_candle_strength'],
                    'reason': 'Bearish breakout from consolidation range with volume confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Breakout below {last_candle['range_bottom']:.2f} after "
                f"{last_candle['range_bars']} bars in range. "
                f"Volume ratio: {last_candle['volume_ratio']:.2f}, ATR: {atr:.2f}"
            )

        return signals