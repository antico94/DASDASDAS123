# strategies/momentum_scalping.py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class MomentumScalpingStrategy(BaseStrategy):
    """5-Minute Momentum Scalping Strategy for XAU/USD."""

    def __init__(self, symbol="XAUUSD", timeframe="M5",
                 ema_period=20, macd_fast=12, macd_slow=26, macd_signal=9,
                 data_fetcher=None):
        """Initialize the Momentum Scalping strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "M5".
            ema_period (int, optional): EMA period for trend. Defaults to 20.
            macd_fast (int, optional): MACD fast period. Defaults to 12.
            macd_slow (int, optional): MACD slow period. Defaults to 26.
            macd_signal (int, optional): MACD signal period. Defaults to 9.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Momentum_Scalping", data_fetcher=data_fetcher)

        # Validate inputs
        if macd_fast >= macd_slow:
            raise ValueError(f"macd_fast ({macd_fast}) must be < macd_slow ({macd_slow})")

        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        # Ensure we fetch enough data for calculations
        self.min_required_candles = max(ema_period, macd_slow + macd_signal) + 20

        self.logger.info(
            f"Initialized Momentum Scalping strategy: {symbol} {timeframe}, "
            f"EMA: {ema_period}, MACD: {macd_fast}/{macd_slow}/{macd_signal}"
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
                f"Insufficient data for momentum calculations. "
                f"Need at least {self.min_required_candles} candles."
            )
            return data

        # Calculate 20 EMA
        data['ema'] = data['close'].ewm(span=self.ema_period, adjust=False).mean()

        # Calculate MACD
        ema_fast = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        data['macd'] = ema_fast - ema_slow
        data['macd_signal'] = data['macd'].ewm(span=self.macd_signal, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Calculate Average True Range (ATR) for volatility
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Calculate recent swing highs and lows for stop loss placement
        data['swing_high'] = data['high'].rolling(window=5, center=True).max()
        data['swing_low'] = data['low'].rolling(window=5, center=True).min()

        # Identify signal conditions
        data = self._identify_signals(data)

        return data

    def _identify_signals(self, data):
        """Identify momentum scalping signals.

        Args:
            data (pandas.DataFrame): OHLC data with indicators

        Returns:
            pandas.DataFrame: Data with signal information
        """
        # Initialize signal columns
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data['prior_trend'] = 0  # 0: None, 1: Uptrend, -1: Downtrend
        data['signal_strength'] = 0.0
        data['stop_loss'] = np.nan
        data['take_profit'] = np.nan

        for i in range(5, len(data)):
            # Skip if not enough prior data
            if i < 5:
                continue

            # Current candle and previous candle data
            current_close = data.iloc[i]['close']
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            current_ema = data.iloc[i]['ema']
            current_macd_hist = data.iloc[i]['macd_histogram']

            prev_close = data.iloc[i - 1]['close']
            prev_ema = data.iloc[i - 1]['ema']
            prev_macd_hist = data.iloc[i - 1]['macd_histogram']

            # Looking back further for trend determination
            price_below_ema_before = data.iloc[i - 3:i - 1]['close'].lt(data.iloc[i - 3:i - 1]['ema']).all()
            price_above_ema_before = data.iloc[i - 3:i - 1]['close'].gt(data.iloc[i - 3:i - 1]['ema']).all()

            macd_hist_negative_before = data.iloc[i - 3:i - 1]['macd_histogram'].lt(0).all()
            macd_hist_positive_before = data.iloc[i - 3:i - 1]['macd_histogram'].gt(0).all()

            # Determine prior trend
            if price_below_ema_before and macd_hist_negative_before:
                data.loc[data.index[i], 'prior_trend'] = -1  # Downtrend
            elif price_above_ema_before and macd_hist_positive_before:
                data.loc[data.index[i], 'prior_trend'] = 1  # Uptrend

            # Buy Signal Conditions:
            # 1. Price was below 20 EMA and MACD histogram was negative (prior downtrend)
            # 2. Now price crosses above 20 EMA
            # 3. MACD histogram crosses from negative to positive or is crossing around the same time
            if (data.iloc[i]['prior_trend'] == -1 and  # Prior downtrend
                    current_close > current_ema and  # Price crosses above EMA
                    prev_close <= prev_ema and  # Confirmed cross
                    ((current_macd_hist > 0 and prev_macd_hist <= 0) or  # MACD histogram crosses positive
                     (current_macd_hist > prev_macd_hist and abs(current_macd_hist) < 0.3 * data.iloc[i][
                         'atr']))):  # Or momentum shifting

                # Find a recent swing low for stop placement
                recent_lows = data.iloc[i - 5:i]['low'].tolist()
                stop_loss = min(recent_lows)

                # Ensure stop loss isn't too far or too close
                max_stop_distance = data.iloc[i]['atr'] * 2  # Maximum 2 ATR
                if current_close - stop_loss > max_stop_distance:
                    stop_loss = current_close - max_stop_distance

                # Calculate take profit based on risk (1:1 reward to risk for first target)
                risk = current_close - stop_loss
                take_profit = current_close + risk

                # Calculate signal strength
                strength = min(1.0, (current_close - prev_ema) / prev_ema * 100)

                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit

            # Sell Signal Conditions:
            # 1. Price was above 20 EMA and MACD histogram was positive (prior uptrend)
            # 2. Now price crosses below 20 EMA
            # 3. MACD histogram crosses from positive to negative or is crossing around the same time
            elif (data.iloc[i]['prior_trend'] == 1 and  # Prior uptrend
                  current_close < current_ema and  # Price crosses below EMA
                  prev_close >= prev_ema and  # Confirmed cross
                  ((current_macd_hist < 0 and prev_macd_hist >= 0) or  # MACD histogram crosses negative
                   (current_macd_hist < prev_macd_hist and abs(current_macd_hist) < 0.3 * data.iloc[i][
                       'atr']))):  # Or momentum shifting

                # Find a recent swing high for stop placement
                recent_highs = data.iloc[i - 5:i]['high'].tolist()
                stop_loss = max(recent_highs)

                # Ensure stop loss isn't too far or too close
                max_stop_distance = data.iloc[i]['atr'] * 2  # Maximum 2 ATR
                if stop_loss - current_close > max_stop_distance:
                    stop_loss = current_close + max_stop_distance

                # Calculate take profit based on risk (1:1 reward to risk for first target)
                risk = stop_loss - current_close
                take_profit = current_close - risk

                # Calculate signal strength
                strength = min(1.0, (prev_ema - current_close) / prev_ema * 100)

                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit

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
        if data.empty or 'signal' not in data.columns:
            self.logger.warning("Insufficient data for momentum scalping analysis after calculations")
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
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate risk in dollars
            risk = entry_price - stop_loss

            # Calculate multiple take profit levels for scaling out
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
                    'ema': last_candle['ema'],
                    'macd_histogram': last_candle['macd_histogram'],
                    'atr': last_candle['atr'],
                    'reason': 'Bullish momentum with EMA and MACD confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                f"MACD histogram: {last_candle['macd_histogram']:.6f}"
            )

        elif last_candle['signal'] == -1:  # Sell signal
            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']
            take_profit = last_candle['take_profit']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate risk in dollars
            risk = stop_loss - entry_price

            # Calculate multiple take profit levels for scaling out
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
                    'ema': last_candle['ema'],
                    'macd_histogram': last_candle['macd_histogram'],
                    'atr': last_candle['atr'],
                    'reason': 'Bearish momentum with EMA and MACD confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Stop loss: {stop_loss:.2f}, Take profit (1R): {take_profit_1r:.2f}, "
                f"MACD histogram: {last_candle['macd_histogram']:.6f}"
            )

        return signals