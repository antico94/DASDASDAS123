# execution/trailing_stop.py - Updated with strategy-specific mechanics
import json
from datetime import datetime
from custom_logging.logger import app_logger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository
from mt5_connector.data_fetcher import MT5DataFetcher


class EnhancedTrailingStopManager:
    """Enhanced class to manage trailing stops with strategy-specific trailing."""

    def __init__(self, connector=None, trade_repository=None, data_fetcher=None):
        """Initialize the enhanced trailing stop manager."""
        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.data_fetcher = data_fetcher or MT5DataFetcher(connector=self.connector)
        self.logger = app_logger

    def update_trailing_stops(self):
        """Update trailing stops for all open positions with strategy-based trailing."""
        # Get all open positions
        positions = self.connector.get_positions()

        if not positions:
            self.logger.debug("No open positions to update trailing stops")
            return 0

        self.logger.info(f"Updating trailing stops for {len(positions)} positions")

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

                        self.logger.info(
                            f"Updated trailing stop for position {position['ticket']} "
                            f"({position['symbol']}): {current_sl} -> {new_sl}"
                        )
                        updated_count += 1

                # For first-half positions, check if we should move to breakeven
                elif "Part1" in position['comment']:
                    # Check if first target has been reached
                    entry_price = position['open_price']
                    target_reached = False

                    # Parse metadata to get the first target
                    metadata = self._get_signal_metadata(trade)

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

                                self.logger.info(
                                    f"First target reached - moved second half stop to breakeven: "
                                    f"position {second_half['ticket']}, SL: {new_sl}"
                                )
                                updated_count += 1

            except Exception as e:
                self.logger.error(f"Error updating trailing stop for position {position['ticket']}: {str(e)}")

        self.logger.info(f"Updated trailing stops for {updated_count} positions")
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
        """Calculate trailing stop for MA Trend strategy based on EMAs."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data with EMAs
        timeframe = metadata.get('timeframe', 'H1')
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 50)

        if df.empty:
            return None

        # Calculate EMAs - fast EMA (20) for trailing
        fast_period = metadata.get('fast_period', 20)
        df['fast_ema'] = df['close'].ewm(span=fast_period, adjust=False).mean()

        # Get the fast EMA value
        fast_ema = df['fast_ema'].iloc[-1]

        # Per the plan: "trail the stop-loss under the 20 EMA by a fixed gap"
        # Fixed gap is typically a small percentage for gold
        if position_type == 0:  # BUY
            buffer_factor = 0.997  # 0.3% below EMA for gold
            new_sl = fast_ema * buffer_factor

            # Only update if better than current stop
            if new_sl > current_sl:
                return new_sl

        else:  # SELL
            buffer_factor = 1.003  # 0.3% above EMA for gold
            new_sl = fast_ema * buffer_factor

            # Only update if better than current stop
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
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 20)

        if df.empty:
            return None

        # Calculate ATR (14-period)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        atr = df['atr'].iloc[-1]

        # Per the plan: "trail the stop by 1.5×ATR once price exceeds 2×ATR from entry"
        if position_type == 0:  # BUY
            # Check if price has moved 2×ATR from entry
            if current_price >= entry_price + (atr * 2):
                # Trail by 1.5×ATR
                new_sl = current_price - (atr * 1.5)
                if new_sl > current_sl:
                    return new_sl

        else:  # SELL
            # Check if price has moved 2×ATR from entry
            if current_price <= entry_price - (atr * 2):
                # Trail by 1.5×ATR
                new_sl = current_price + (atr * 1.5)
                if new_sl < current_sl:
                    return new_sl

        # If price hasn't moved enough, no update
        return None

    def _range_trailing_stop(self, position, metadata):
        """Calculate trailing stop for Range strategy."""
        # For range strategy, we generally don't use trailing stops
        # Just move to breakeven when price reaches midpoint
        midpoint = metadata.get('range_midpoint', 0)
        position_type = position['type']
        entry_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        if midpoint > 0:
            if position_type == 0 and current_price > midpoint:  # BUY crossed midpoint
                return max(entry_price, current_sl)
            elif position_type == 1 and current_price < midpoint:  # SELL crossed midpoint
                return min(entry_price, current_sl)

        return None

    def _momentum_trailing_stop(self, position, metadata):
        """Calculate trailing stop for Momentum Scalping strategy."""
        symbol = position['symbol']
        position_type = position['type']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data to calculate 20 EMA
        timeframe = "M5"  # Momentum strategy uses M5
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 30)

        if df.empty:
            return None

        # Calculate 20 EMA
        df['ema'] = df['close'].ewm(span=20, adjust=False).mean()

        # Get current EMA value
        current_ema = df['ema'].iloc[-1]

        # Per the plan: "trail the stop by either breakeven or [20-period EMA minus 15 pips], whichever is higher"
        # For gold, 15 pips is $1.50 if pip = $0.10
        pip_value = 0.10  # Can be adjusted based on broker's pip definition
        pip_offset = 15 * pip_value

        if position_type == 0:  # BUY
            # Trail stop at (EMA - 15 pips) as specified in plan
            ema_based_stop = current_ema - pip_offset

            # Return the better of current stop or EMA-based stop
            if ema_based_stop > current_sl:
                return ema_based_stop

        else:  # SELL
            # Trail stop at (EMA + 15 pips)
            ema_based_stop = current_ema + pip_offset

            # Return the better of current stop or EMA-based stop
            if ema_based_stop < current_sl:
                return ema_based_stop

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

        # Get current Kijun value
        kijun = df['kijun_sen'].iloc[-1]

        # Per the plan: "use the Kijun-sen as a trailing stop"
        if position_type == 0:  # BUY
            # Trail slightly below Kijun
            kijun_stop = kijun * 0.998  # 0.2% below Kijun

            if kijun_stop > current_sl:
                return kijun_stop

        else:  # SELL
            # Trail slightly above Kijun
            kijun_stop = kijun * 1.002  # 0.2% above Kijun

            if kijun_stop < current_sl:
                return kijun_stop

        return None

    def _atr_trailing_stop(self, position):
        """Default ATR-based trailing stop for any strategy."""
        symbol = position['symbol']
        position_type = position['type']
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data and calculate ATR
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, "H1", 20)

        if df.empty:
            return None

        # Calculate ATR (14-period)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        atr = df['atr'].iloc[-1]

        # Generic ATR-based trailing
        if position_type == 0:  # BUY
            new_sl = current_price - (atr * 1.5)
            if new_sl > current_sl:
                return new_sl
        else:  # SELL
            new_sl = current_price + (atr * 1.5)
            if new_sl < current_sl:
                return new_sl

        return None

    def _get_signal_metadata(self, trade):
        """Extract and parse metadata from a trade's signal."""
        metadata = {}
        if trade and trade.signal_id:
            try:
                # Get signal for the trade
                session = self.trade_repository._session.query("StrategySignal")
                signal = session.get_by_id(trade.signal_id)

                if signal and signal.signal_data:
                    metadata = json.loads(signal.signal_data)
            except Exception as e:
                self.logger.warning(f"Could not get signal metadata: {str(e)}")

        return metadata

    # Existing helper methods...
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