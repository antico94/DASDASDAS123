# strategies/ichimoku.py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from custom_logging.logger import app_logger
from strategies.base_strategy import BaseStrategy
from data.models import StrategySignal


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud Strategy for XAU/USD."""

    def __init__(self, symbol="XAUUSD", timeframe="H1",
                 tenkan_period=9, kijun_period=26, senkou_b_period=52,
                 data_fetcher=None):
        """Initialize the Ichimoku Cloud strategy.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to "XAUUSD".
            timeframe (str, optional): Chart timeframe. Defaults to "H1".
            tenkan_period (int, optional): Tenkan-sen period. Defaults to 9.
            kijun_period (int, optional): Kijun-sen period. Defaults to 26.
            senkou_b_period (int, optional): Senkou Span B period. Defaults to 52.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        super().__init__(symbol, timeframe, name="Ichimoku_Cloud", data_fetcher=data_fetcher)

        # Validate inputs
        if tenkan_period >= kijun_period:
            raise ValueError(f"tenkan_period ({tenkan_period}) must be < kijun_period ({kijun_period})")
        if kijun_period >= senkou_b_period:
            raise ValueError(f"kijun_period ({kijun_period}) must be < senkou_b_period ({senkou_b_period})")

        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

        # Ensure we fetch enough data for calculations
        self.min_required_candles = kijun_period * 2 + senkou_b_period + 30  # Extra for displaced Senkou spans and Chikou Span

        self.logger.info(
            f"Initialized Ichimoku Cloud strategy: {symbol} {timeframe}, "
            f"Tenkan: {tenkan_period}, Kijun: {kijun_period}, Senkou B: {senkou_b_period}"
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
                f"Insufficient data for Ichimoku calculations. "
                f"Need at least {self.min_required_candles} candles."
            )
            return data

        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
        period_high = data['high'].rolling(window=self.tenkan_period).max()
        period_low = data['low'].rolling(window=self.tenkan_period).min()
        data['tenkan_sen'] = (period_high + period_low) / 2

        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
        period_high = data['high'].rolling(window=self.kijun_period).max()
        period_low = data['low'].rolling(window=self.kijun_period).min()
        data['kijun_sen'] = (period_high + period_low) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(self.kijun_period)

        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_b_period, plotted 26 periods ahead
        period_high = data['high'].rolling(window=self.senkou_b_period).max()
        period_low = data['low'].rolling(window=self.senkou_b_period).min()
        data['senkou_span_b'] = ((period_high + period_low) / 2).shift(self.kijun_period)

        # Calculate Chikou Span (Lagging Span): Current closing price, plotted 26 periods back
        data['chikou_span'] = data['close'].shift(-self.kijun_period)

        # Calculate average true range for stop loss placement
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()

        # Identify Kumo (Cloud) color
        data['cloud_bullish'] = data['senkou_span_a'] > data['senkou_span_b']

        # Identify TK cross signals
        data = self._identify_signals(data)

        return data

    def _identify_signals(self, data):
        """Identify Ichimoku signals.

        Args:
            data (pandas.DataFrame): OHLC data with Ichimoku indicators

        Returns:
            pandas.DataFrame: Data with signal information
        """
        # Initialize signal columns
        data['signal'] = 0  # 0: No signal, 1: Buy, -1: Sell
        data['signal_strength'] = 0.0
        data['stop_loss'] = np.nan
        data['take_profit'] = np.nan

        for i in range(self.kijun_period + 10, len(data)):
            # Skip if not enough prior data
            if i < self.kijun_period + 10:
                continue

            # Get current values (current completed candle)
            current_close = data.iloc[i]['close']
            current_tenkan = data.iloc[i]['tenkan_sen']
            current_kijun = data.iloc[i]['kijun_sen']
            current_span_a = data.iloc[i]['senkou_span_a']
            current_span_b = data.iloc[i]['senkou_span_b']

            # Get previous values
            prev_tenkan = data.iloc[i - 1]['tenkan_sen']
            prev_kijun = data.iloc[i - 1]['kijun_sen']

            # Calculate cloud top and bottom at current position
            cloud_top = max(current_span_a, current_span_b)
            cloud_bottom = min(current_span_a, current_span_b)

            # Determine if Chikou Span is above/below price 26 periods ago
            # Note: We need to look at current price vs. price that was 26 periods before current
            price_26_periods_ago = data.iloc[i - self.kijun_period]['close']
            chikou_above_price = current_close > price_26_periods_ago

            # Determine if we have a TK cross
            # Tenkan crosses above Kijun
            tk_bullish_cross = current_tenkan > current_kijun and prev_tenkan <= prev_kijun

            # Tenkan crosses below Kijun
            tk_bearish_cross = current_tenkan < current_kijun and prev_tenkan >= prev_kijun

            # Bullish Signal Conditions:
            # 1. TK Cross bullish (Tenkan crosses above Kijun)
            # 2. Price is above the Cloud
            # 3. Cloud ahead is bullish (Senkou A > Senkou B)
            # 4. Chikou Span is above price from 26 periods ago
            if (tk_bullish_cross and
                    current_close > cloud_top and
                    data.iloc[i]['cloud_bullish'] and
                    chikou_above_price):

                # Find a good stop loss level (use Kijun-sen or cloud top)
                # Ichimoku suggests Kijun-sen as a logical support in an uptrend
                stop_loss = current_kijun

                # If Kijun is too far, use a tighter stop based on ATR
                kijun_distance = current_close - current_kijun
                if kijun_distance > data.iloc[i]['atr'] * 3:
                    stop_loss = current_close - data.iloc[i]['atr'] * 2

                # Calculate take profit levels (1.5 and 3 times the risk)
                risk = current_close - stop_loss
                take_profit_1 = current_close + (risk * 1.5)
                take_profit_2 = current_close + (risk * 3)

                # Calculate signal strength (1.0 = very strong)
                # Based on cloud thickness, TK cross decisiveness, and Chikou span position
                cloud_thickness = (cloud_top - cloud_bottom) / cloud_bottom
                tk_decisiveness = (current_tenkan - current_kijun) / current_kijun
                chikou_strength = (current_close - price_26_periods_ago) / price_26_periods_ago

                strength = min(1.0, (0.4 * cloud_thickness + 0.3 * abs(tk_decisiveness) + 0.3 * abs(chikou_strength)))

                data.loc[data.index[i], 'signal'] = 1  # Buy signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit_1  # Store primary target

            # Bearish Signal Conditions:
            # 1. TK Cross bearish (Tenkan crosses below Kijun)
            # 2. Price is below the Cloud
            # 3. Cloud ahead is bearish (Senkou A < Senkou B)
            # 4. Chikou Span is below price from 26 periods ago
            elif (tk_bearish_cross and
                  current_close < cloud_bottom and
                  not data.iloc[i]['cloud_bullish'] and
                  not chikou_above_price):

                # Find a good stop loss level (use Kijun-sen or cloud bottom)
                # Ichimoku suggests Kijun-sen as a logical resistance in a downtrend
                stop_loss = current_kijun

                # If Kijun is too far, use a tighter stop based on ATR
                kijun_distance = current_kijun - current_close
                if kijun_distance > data.iloc[i]['atr'] * 3:
                    stop_loss = current_close + data.iloc[i]['atr'] * 2

                # Calculate take profit levels (1.5 and 3 times the risk)
                risk = stop_loss - current_close
                take_profit_1 = current_close - (risk * 1.5)
                take_profit_2 = current_close - (risk * 3)

                # Calculate signal strength (1.0 = very strong)
                cloud_thickness = (cloud_top - cloud_bottom) / cloud_bottom
                tk_decisiveness = (current_kijun - current_tenkan) / current_kijun
                chikou_strength = (price_26_periods_ago - current_close) / price_26_periods_ago

                strength = min(1.0, (0.4 * cloud_thickness + 0.3 * abs(tk_decisiveness) + 0.3 * abs(chikou_strength)))

                data.loc[data.index[i], 'signal'] = -1  # Sell signal
                data.loc[data.index[i], 'signal_strength'] = strength
                data.loc[data.index[i], 'stop_loss'] = stop_loss
                data.loc[data.index[i], 'take_profit'] = take_profit_1  # Store primary target

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
            self.logger.warning("Insufficient data for Ichimoku analysis after calculations")
            return []

        signals = []

        # Get the last complete candle
        last_candle = data.iloc[-1]

        # Check for trading signal on the last candle
        if last_candle['signal'] == 1:  # Buy signal
            # Create BUY signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']

            # Ensure stop loss is valid
            if stop_loss >= entry_price:
                stop_loss = entry_price * 0.995  # Default 0.5% below entry

            # Calculate risk in dollars
            risk = entry_price - stop_loss

            # Calculate multiple take profit levels
            take_profit_1 = entry_price + (risk * 1.5)  # 1.5:1 reward-to-risk
            take_profit_2 = entry_price + (risk * 3)  # 3:1 reward-to-risk

            signal = self.create_signal(
                signal_type="BUY",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'tenkan_sen': last_candle['tenkan_sen'],
                    'kijun_sen': last_candle['kijun_sen'],
                    'senkou_span_a': last_candle['senkou_span_a'],
                    'senkou_span_b': last_candle['senkou_span_b'],
                    'cloud_bullish': bool(last_candle['cloud_bullish']),
                    'reason': 'Bullish TK cross above cloud with Chikou confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated BUY signal for {self.symbol} at {entry_price}. "
                f"Tenkan: {last_candle['tenkan_sen']:.2f}, Kijun: {last_candle['kijun_sen']:.2f}, "
                f"Cloud: {'Bullish' if last_candle['cloud_bullish'] else 'Bearish'}"
            )

        elif last_candle['signal'] == -1:  # Sell signal
            # Create SELL signal
            entry_price = last_candle['close']
            stop_loss = last_candle['stop_loss']

            # Ensure stop loss is valid
            if stop_loss <= entry_price:
                stop_loss = entry_price * 1.005  # Default 0.5% above entry

            # Calculate risk in dollars
            risk = stop_loss - entry_price

            # Calculate multiple take profit levels
            take_profit_1 = entry_price - (risk * 1.5)  # 1.5:1 reward-to-risk
            take_profit_2 = entry_price - (risk * 3)  # 3:1 reward-to-risk

            signal = self.create_signal(
                signal_type="SELL",
                price=entry_price,
                strength=last_candle['signal_strength'],
                metadata={
                    'stop_loss': stop_loss,
                    'take_profit_1': take_profit_1,
                    'take_profit_2': take_profit_2,
                    'tenkan_sen': last_candle['tenkan_sen'],
                    'kijun_sen': last_candle['kijun_sen'],
                    'senkou_span_a': last_candle['senkou_span_a'],
                    'senkou_span_b': last_candle['senkou_span_b'],
                    'cloud_bullish': bool(last_candle['cloud_bullish']),
                    'reason': 'Bearish TK cross below cloud with Chikou confirmation'
                }
            )
            signals.append(signal)

            self.logger.info(
                f"Generated SELL signal for {self.symbol} at {entry_price}. "
                f"Tenkan: {last_candle['tenkan_sen']:.2f}, Kijun: {last_candle['kijun_sen']:.2f}, "
                f"Cloud: {'Bullish' if last_candle['cloud_bullish'] else 'Bearish'}"
            )

        return signals