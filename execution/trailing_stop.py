import json
import pandas as pd
from db_logger.db_logger import DBLogger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository
from mt5_connector.data_fetcher import MT5DataFetcher


class EnhancedTrailingStopManager:
    """Enhanced class to manage trailing stops with strategy-specific techniques."""

    def __init__(self, connector=None, trade_repository=None, data_fetcher=None):
        """Initialize the enhanced trailing stop manager."""
        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.data_fetcher = data_fetcher or MT5DataFetcher(connector=self.connector)

    def update_trailing_stops(self):
        """Update trailing stops for all open positions with strategy-based trailing."""
        # Get all open positions
        positions = self.connector.get_positions()

        if not positions:
            DBLogger.log_event("DEBUG", "No open positions to update trailing stops", "TrailingStopManager")
            return 0

        DBLogger.log_event("INFO", f"Updating trailing stops for {len(positions)} positions", "TrailingStopManager")

        updated_count = 0
        for position in positions:
            try:
                # Extract strategy and signal info from comment
                strategy_name = self._extract_strategy_name(position['comment'])
                signal_id = self._extract_signal_id(position['comment'])

                # Check if this is a partial position (second half)
                is_partial = "_Part" in position['comment']
                is_second_half = "Part2" in position['comment']

                # Get the trade from database
                trade = self._find_trade_by_signal_and_comment(signal_id, position['comment'])
                if not trade or trade.close_time is not None:
                    continue

                # Get current stop loss
                current_sl = position['stop_loss']
                current_price = position['current_price']
                position_type = position['type']  # 0=BUY, 1=SELL
                entry_price = position['open_price']

                # For second-half positions that have reached breakeven,
                # apply strategy-specific trailing stop logic
                if is_second_half:
                    # Calculate new stop loss based on strategy
                    new_sl = self._calculate_trailing_stop(
                        position=position,
                        strategy_name=strategy_name,
                        trade=trade,
                        is_partial=is_partial
                    )

                    # Only update if new SL is better than current
                    if new_sl and self._is_better_stop_loss(position_type, current_sl, new_sl, current_price):
                        # Update the stop loss
                        self.connector.modify_position(
                            ticket=position['ticket'],
                            stop_loss=new_sl
                        )

                        # Update in database
                        trade.stop_loss = new_sl
                        self.trade_repository.update(trade)

                        DBLogger.log_position(
                            position_type="MODIFIED",
                            symbol=position['symbol'],
                            ticket=position['ticket'],
                            volume=position['volume'],
                            stop_loss=new_sl,
                            message=f"Updated trailing stop for position {position['ticket']} "
                                    f"({position['symbol']}): {current_sl} -> {new_sl}"
                        )
                        updated_count += 1

                # For first-half positions, check if we should move to breakeven
                elif "Part1" in position['comment']:
                    # Check if first target has been reached
                    # Parse metadata to get the first target
                    metadata = self._get_signal_metadata(trade)

                    # Calculate first target based on risk multiple
                    if position_type == 0:  # BUY
                        # First target is 1:1 for most strategies
                        risk = entry_price - current_sl
                        first_target = entry_price + risk
                        target_reached = current_price >= first_target
                    else:  # SELL
                        risk = current_sl - entry_price
                        first_target = entry_price - risk
                        target_reached = current_price <= first_target

                    # Move second half to breakeven when first target is reached
                    if target_reached:
                        # Find the second half position
                        second_half_positions = [
                            p for p in positions
                            if p['comment'].replace("Part1", "Part2") == position['comment'].replace("Part1", "Part2")
                        ]

                        if second_half_positions:
                            second_half = second_half_positions[0]

                            # Set stop to breakeven
                            new_sl = second_half['open_price']

                            # Only update if different from current
                            if abs(new_sl - second_half['stop_loss']) > 0.0001:
                                # Update the stop loss to breakeven
                                self.connector.modify_position(
                                    ticket=second_half['ticket'],
                                    stop_loss=new_sl
                                )

                                DBLogger.log_position(
                                    position_type="MODIFIED",
                                    symbol=second_half['symbol'],
                                    ticket=second_half['ticket'],
                                    volume=second_half['volume'],
                                    stop_loss=new_sl,
                                    message=f"First target reached - moved second half stop to breakeven: "
                                            f"position {second_half['ticket']}, SL: {new_sl}"
                                )
                                updated_count += 1

            except Exception as e:
                DBLogger.log_error("TrailingStopManager",
                                   f"Error updating trailing stop for position {position['ticket']}", exception=e)

        DBLogger.log_event("INFO", f"Updated trailing stops for {updated_count} positions", "TrailingStopManager")
        return updated_count

    def _calculate_trailing_stop(self, position, strategy_name, trade, is_partial):
        """Calculate new trailing stop based on strategy and position type."""
        # Get position details
        symbol = position['symbol']
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Extract metadata from trade
        metadata = self._get_signal_metadata(trade)

        # Calculate strategy-specific trailing stop
        if strategy_name == "EnhancedMA_Trend":
            return self._ma_trend_trailing_stop(position, metadata)
        elif strategy_name == "Breakout":
            return self._breakout_trailing_stop(position, metadata)
        elif strategy_name == "Range_Mean_Reversion":
            return self._range_trailing_stop(position, metadata)
        elif strategy_name == "Momentum_Scalping":
            return self._momentum_trailing_stop(position, metadata)
        elif strategy_name == "Ichimoku_Cloud":
            return self._ichimoku_trailing_stop(position, metadata)
        else:
            # Default ATR-based trailing if strategy not recognized
            return self._atr_trailing_stop(position)

    def _ma_trend_trailing_stop(self, position, metadata):
        """Calculate trailing stop for MA Trend strategy using ATR, Price-Swing, or MA methods."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']
        entry_price = position['open_price']
        highest_price = current_price  # Initialize with current price

        # Get OHLC data to calculate trailing stop methods
        timeframe = metadata.get('timeframe', 'H1')
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 100)

        if df.empty:
            return None

        # Calculate ATR for volatility-based trail
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        atr = df['atr'].iloc[-1]

        # Calculate Slow EMA (50-period) for MA-based trail
        df['slow_ema'] = df['close'].ewm(span=50, adjust=False).mean()
        slow_ema = df['slow_ema'].iloc[-1]

        # Find recent swing points for price-swing trail
        # For longs, we want the highest high and the recent highest low
        if position_type == 0:  # BUY
            # Find highest price since entry
            # Get recent highest high
            highest_bars = df.iloc[-20:]['high']
            if not highest_bars.empty:
                highest_price = highest_bars.max()

            # Find most recent swing low (higher low in uptrend)
            # A simple way is to look for local minima in the last few bars
            recent_lows = []
            for i in range(2, min(20, len(df) - 2)):
                if df['low'].iloc[-i] < df['low'].iloc[-i + 1] and df['low'].iloc[-i] < df['low'].iloc[-i - 1]:
                    recent_lows.append(df['low'].iloc[-i])
                    if len(recent_lows) >= 3:  # Get up to 3 recent swing lows
                        break

            # Get the highest of the recent swing lows (most recent higher low)
            recent_swing_low = None
            if recent_lows:
                recent_swing_low = max(recent_lows)

            # Calculate potential stop loss values using all three methods:

            # 1. ATR Chandelier Exit (2.5-3x ATR below highest price)
            atr_multiplier = 2.5  # Documented recommendation for gold
            atr_stop = highest_price - (atr * atr_multiplier)

            # 2. Price Swing Trail (below recent swing low)
            swing_stop = None
            if recent_swing_low is not None and recent_swing_low > current_sl:
                swing_stop = recent_swing_low

            # 3. Moving Average Trail (below slow EMA with buffer)
            ema_buffer = 0.003  # 0.3% below slow EMA
            ema_stop = slow_ema * (1 - ema_buffer)

            # Choose the tightest valid stop loss from the three methods
            # Start with ATR as baseline
            new_sl = atr_stop

            # Check if swing stop is tighter but still valid
            if swing_stop is not None and swing_stop > new_sl:
                new_sl = swing_stop

            # Check if EMA stop is tighter but still valid
            if ema_stop > new_sl:
                new_sl = ema_stop

            # Only return if the new stop is better than current
            if new_sl > current_sl:
                return new_sl

        else:  # SELL
            # Find lowest price since entry
            lowest_bars = df.iloc[-20:]['low']
            if not lowest_bars.empty:
                lowest_price = lowest_bars.min()
            else:
                lowest_price = current_price

            # Find most recent swing high (lower high in downtrend)
            recent_highs = []
            for i in range(2, min(20, len(df) - 2)):
                if df['high'].iloc[-i] > df['high'].iloc[-i + 1] and df['high'].iloc[-i] > df['high'].iloc[-i - 1]:
                    recent_highs.append(df['high'].iloc[-i])
                    if len(recent_highs) >= 3:  # Get up to 3 recent swing highs
                        break

            # Get the lowest of the recent swing highs (most recent lower high)
            recent_swing_high = None
            if recent_highs:
                recent_swing_high = min(recent_highs)

            # Calculate potential stop loss values:

            # 1. ATR Chandelier Exit
            atr_multiplier = 2.5  # Documented recommendation for gold
            atr_stop = lowest_price + (atr * atr_multiplier)

            # 2. Price Swing Trail
            swing_stop = None
            if recent_swing_high is not None and recent_swing_high < current_sl:
                swing_stop = recent_swing_high

            # 3. Moving Average Trail
            ema_buffer = 0.003  # 0.3% above slow EMA
            ema_stop = slow_ema * (1 + ema_buffer)

            # Choose the tightest valid stop loss
            new_sl = atr_stop

            if swing_stop is not None and swing_stop < new_sl:
                new_sl = swing_stop

            if ema_stop < new_sl:
                new_sl = ema_stop

            # Only return if the new stop is better than current
            if new_sl < current_sl:
                return new_sl

        return None

    def _breakout_trailing_stop(self, position, metadata):
        """Calculate trailing stop for Breakout strategy using ATR."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']
        entry_price = position['open_price']

        # Get current ATR value
        timeframe = metadata.get('timeframe', 'M15')
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 30)

        if df.empty:
            return None

        # Calculate ATR (14-period)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        atr = df['atr'].iloc[-1]

        # Per the document: "trail the stop by 1.5×ATR once price exceeds 2×ATR from entry"
        if position_type == 0:  # BUY
            # Check if price has moved 2×ATR from entry
            if current_price >= entry_price + (atr * 2):
                # Find highest price since entry for Chandelier Exit style
                highest_bars = df.iloc[-20:]['high']
                highest_price = highest_bars.max() if not highest_bars.empty else current_price

                # Trail by 1.5×ATR below highest price (Chandelier Exit)
                new_sl = highest_price - (atr * 1.5)

                if new_sl > current_sl:
                    return new_sl
        else:  # SELL
            # Check if price has moved 2×ATR from entry
            if current_price <= entry_price - (atr * 2):
                # Find lowest price since entry
                lowest_bars = df.iloc[-20:]['low']
                lowest_price = lowest_bars.min() if not lowest_bars.empty else current_price

                # Trail by 1.5×ATR above lowest price
                new_sl = lowest_price + (atr * 1.5)

                if new_sl < current_sl:
                    return new_sl

        return None

    def _range_trailing_stop(self, position, metadata):
        """Calculate trailing stop for Range strategy using midpoint/SMA."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']
        entry_price = position['open_price']

        # Get OHLC data to calculate SMA and range midpoint
        timeframe = metadata.get('timeframe', 'M15')
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 30)

        if df.empty:
            return None

        # Calculate 20-period SMA (midpoint proxy for range)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        sma_20 = df['sma_20'].iloc[-1]

        # Get actual range midpoint from metadata if available
        range_midpoint = metadata.get('range_midpoint', 0)
        range_top = metadata.get('range_top', 0)
        range_bottom = metadata.get('range_bottom', 0)

        # If we have a valid range, use the midpoint
        if range_top > 0 and range_bottom > 0:
            range_midpoint = (range_top + range_bottom) / 2

        # For range trades, once profit reaches a threshold, we trail
        # to protect profit in case price fails to reach the opposite boundary

        if position_type == 0:  # BUY at support, trail once we move up significantly
            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price

            # If we've moved significantly (halfway to target or 0.5%), trail using SMA or midpoint
            if (profit_pct > 0.005 or (range_midpoint > 0 and current_price > range_midpoint)):
                # Use SMA-20 as a trailing reference with a small buffer
                sma_buffer = 0.0015  # 0.15% below SMA
                sma_stop = sma_20 * (1 - sma_buffer)

                # If we have range midpoint, consider that too
                midpoint_stop = None
                if range_midpoint > 0:
                    midpoint_stop = range_midpoint * 0.997  # Just below midpoint

                # Choose the higher of SMA-based or midpoint-based stop
                new_sl = sma_stop
                if midpoint_stop and midpoint_stop > new_sl:
                    new_sl = midpoint_stop

                # Only update if it's better than current stop
                if new_sl > current_sl:
                    return new_sl

        else:  # SELL at resistance, trail once we move down significantly
            # Calculate profit percentage (for shorts, profit is positive when price falls)
            profit_pct = (entry_price - current_price) / entry_price

            # If we've moved significantly, trail using SMA or midpoint
            if (profit_pct > 0.005 or (range_midpoint > 0 and current_price < range_midpoint)):
                # Use SMA-20 as a trailing reference with a small buffer
                sma_buffer = 0.0015  # 0.15% above SMA
                sma_stop = sma_20 * (1 + sma_buffer)

                # If we have range midpoint, consider that too
                midpoint_stop = None
                if range_midpoint > 0:
                    midpoint_stop = range_midpoint * 1.003  # Just above midpoint

                # Choose the lower of SMA-based or midpoint-based stop
                new_sl = sma_stop
                if midpoint_stop and midpoint_stop < new_sl:
                    new_sl = midpoint_stop

                # Only update if it's better than current stop
                if new_sl < current_sl:
                    return new_sl

        return None

    def _momentum_trailing_stop(self, position, metadata):
        """Calculate trailing stop for Momentum Scalping strategy based on EMA and one-bar trail."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']
        entry_price = position['open_price']

        # Get OHLC data to calculate 20 EMA
        timeframe = "M5"  # Momentum strategy uses M5
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 30)

        if df.empty:
            return None

        # Calculate 20 EMA
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        ema_20 = df['ema_20'].iloc[-1]

        # For momentum scalping, we have two methods:
        # 1. EMA-based trailing with buffer
        # 2. One-bar trailing stop

        # Let's implement the EMA-based trailing first, which is mentioned in the document:
        # "trail the stop by either breakeven or [20-period EMA minus 15 pips], whichever is higher"

        # For gold, 15 pips is approximately $1.50 if pip = $0.10
        pip_value = 0.10  # Can be adjusted based on broker's pip definition
        pip_offset = 15 * pip_value  # $1.50 for gold

        if position_type == 0:  # BUY
            # EMA-based stop (EMA minus buffer)
            ema_stop = ema_20 - pip_offset

            # Also consider one-bar trailing (for very aggressive trailing)
            one_bar_stop = None
            if len(df) >= 2:
                # Use the low of the previous bar
                one_bar_stop = df['low'].iloc[-2]

                # Only valid if it's higher than entry and the bar wasn't extremely long
                bar_range = df['high'].iloc[-2] - df['low'].iloc[-2]
                if one_bar_stop <= entry_price or bar_range > pip_offset * 2:
                    one_bar_stop = None

            # Choose the better of the two methods
            if one_bar_stop and one_bar_stop > ema_stop:
                new_sl = one_bar_stop
            else:
                new_sl = ema_stop

            # Only update if better than current stop and breakeven
            breakeven_stop = entry_price
            new_sl = max(new_sl, breakeven_stop)

            if new_sl > current_sl:
                return new_sl

        else:  # SELL
            # EMA-based stop (EMA plus buffer)
            ema_stop = ema_20 + pip_offset

            # Also consider one-bar trailing
            one_bar_stop = None
            if len(df) >= 2:
                # Use the high of the previous bar
                one_bar_stop = df['high'].iloc[-2]

                # Only valid if it's lower than entry and the bar wasn't extremely long
                bar_range = df['high'].iloc[-2] - df['low'].iloc[-2]
                if one_bar_stop >= entry_price or bar_range > pip_offset * 2:
                    one_bar_stop = None

            # Choose the better of the two methods
            if one_bar_stop and one_bar_stop < ema_stop:
                new_sl = one_bar_stop
            else:
                new_sl = ema_stop

            # Only update if better than current stop and breakeven
            breakeven_stop = entry_price
            new_sl = min(new_sl, breakeven_stop)

            if new_sl < current_sl:
                return new_sl

        return None

    def _ichimoku_trailing_stop(self, position, metadata):
        """Calculate trailing stop using Kijun-sen for Ichimoku strategy."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data
        timeframe = metadata.get('timeframe', 'H1')
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 80)  # Need extended data for Ichimoku

        if df.empty:
            return None

        # Calculate Kijun-sen (26-period)
        period_high = df['high'].rolling(window=26).max()
        period_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (period_high + period_low) / 2

        # Also calculate the Cloud for alternative trailing method
        period_high_senkou_b = df['high'].rolling(window=52).max()
        period_low_senkou_b = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = (period_high_senkou_b + period_low_senkou_b) / 2

        tenkan_high = df['high'].rolling(window=9).max()
        tenkan_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = df['senkou_span_b'].shift(26)

        # Get current values
        kijun = df['kijun_sen'].iloc[-1]

        # For cloud values, we need to account for the 26-period shift
        # We want current cloud values, not future projection
        senkou_a_current = None
        senkou_b_current = None

        if len(df) > 26:
            # Get current cloud values (actual cloud at current price position)
            cloud_idx = -26
            if 0 <= cloud_idx < len(df):
                senkou_a_current = df['senkou_span_a'].iloc[cloud_idx]
                senkou_b_current = df['senkou_span_b'].iloc[cloud_idx]

        # Determine cloud boundaries
        cloud_top = None
        cloud_bottom = None
        if senkou_a_current is not None and senkou_b_current is not None:
            cloud_top = max(senkou_a_current, senkou_b_current)
            cloud_bottom = min(senkou_a_current, senkou_b_current)

        # Per the document: "use the Kijun-sen as a trailing stop"
        # With option to also use cloud boundary
        if position_type == 0:  # BUY
            # Primary: Kijun-sen trailing
            kijun_stop = kijun * 0.998  # 0.2% below Kijun

            # Alternative: Cloud bottom trailing (if price is above cloud)
            cloud_stop = None
            if cloud_bottom is not None and current_price > cloud_top:
                cloud_stop = cloud_bottom

            # Choose the better stop (higher is better for long positions)
            new_sl = kijun_stop
            if cloud_stop and cloud_stop > new_sl:
                new_sl = cloud_stop

            # Only update if better than current
            if new_sl > current_sl:
                return new_sl

        else:  # SELL
            # Primary: Kijun-sen trailing
            kijun_stop = kijun * 1.002  # 0.2% above Kijun

            # Alternative: Cloud top trailing (if price is below cloud)
            cloud_stop = None
            if cloud_top is not None and current_price < cloud_bottom:
                cloud_stop = cloud_top

            # Choose the better stop (lower is better for short positions)
            new_sl = kijun_stop
            if cloud_stop and cloud_stop < new_sl:
                new_sl = cloud_stop

            # Only update if better than current
            if new_sl < current_sl:
                return new_sl

        return None

    def _atr_trailing_stop(self, position):
        """Default ATR-based trailing stop for any strategy."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data and calculate ATR
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, "H1", 30)

        if df.empty:
            return None

        # Calculate ATR (14-period)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        atr = df['atr'].iloc[-1]

        # Use Chandelier Exit style ATR trailing
        if position_type == 0:  # BUY
            # Find highest price in recent bars
            highest_high = df['high'].rolling(window=10).max().iloc[-1]
            new_sl = highest_high - (atr * 2.5)  # 2.5x ATR is recommended for gold

            if new_sl > current_sl:
                return new_sl
        else:  # SELL
            # Find lowest price in recent bars
            lowest_low = df['low'].rolling(window=10).min().iloc[-1]
            new_sl = lowest_low + (atr * 2.5)

            if new_sl < current_sl:
                return new_sl

        return None

    def _get_signal_metadata(self, trade):
        """Extract and parse metadata from a trade's signal."""
        metadata = {}
        if trade and trade.signal_id:
            try:
                # Get signal for the trade
                session = self.trade_repository._session()
                signal = session.query("StrategySignal").filter_by(id=trade.signal_id).first()

                if signal and signal.signal_data:
                    metadata = json.loads(signal.signal_data)

                session.close()
            except Exception as e:
                DBLogger.log_error("TrailingStopManager", "Could not get signal metadata", exception=e)

        return metadata

    def _extract_strategy_name(self, comment):
        """Extract strategy name from position comment."""
        parts = comment.split('_')
        if len(parts) >= 3 and parts[0] == "Signal":
            # Format is "Signal_{signal_id}_{strategy_name}"
            # or "Signal_{signal_id}_{strategy_name}_Part{N}"
            if len(parts) == 3:
                return parts[2]
            elif len(parts) > 3 and parts[3].startswith("Part"):
                return parts[2]
        return None

    def _extract_signal_id(self, comment):
        """Extract signal ID from position comment."""
        parts = comment.split('_')
        if len(parts) >= 2 and parts[0] == "Signal":
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    def _find_trade_by_signal_and_comment(self, signal_id, comment):
        """Find a trade by signal ID and comment."""
        if not signal_id:
            return None

        trades = self.trade_repository.get_all()
        for trade in trades:
            if trade.signal_id == signal_id and trade.comment == comment and trade.close_time is None:
                return trade

        return None

    def _is_better_stop_loss(self, position_type, current_sl, new_sl, current_price):
        """Check if the new stop loss is better than the current one."""
        # For buy positions, a higher stop loss is better
        if position_type == 0:
            # Ensure SL doesn't get too close to current price
            min_distance = current_price * 0.002  # 0.2% minimum distance
            if new_sl < current_price - min_distance and new_sl > current_sl:
                return True
        # For sell positions, a lower stop loss is better
        else:
            # Ensure SL doesn't get too close to current price
            min_distance = current_price * 0.002  # 0.2% minimum distance
            if new_sl > current_price + min_distance and new_sl < current_sl:
                return True

        return False