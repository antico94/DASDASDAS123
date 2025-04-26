# execution/enhanced_trailing_stop.py
import json
import MetaTrader5 as mt5
from custom_logging.logger import app_logger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository
from mt5_connector.data_fetcher import MT5DataFetcher


class EnhancedTrailingStopManager:
    """Enhanced class to manage trailing stops with EMA-based trailing and multi-timeframe reference."""

    def __init__(self, connector=None, trade_repository=None, data_fetcher=None):
        """Initialize the enhanced trailing stop manager.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None.
            trade_repository (TradeRepository, optional): Trade repository. Defaults to None.
            data_fetcher (MT5DataFetcher, optional): Data fetcher. Defaults to None.
        """
        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.data_fetcher = data_fetcher or MT5DataFetcher(connector=self.connector)

        self.logger = app_logger

    def update_trailing_stops(self):
        """Update trailing stops for all open positions with strategy-based trailing.

        Returns:
            int: Number of positions updated
        """
        # Get all open positions
        positions = self.connector.get_positions()

        if not positions:
            self.logger.debug("No open positions to update trailing stops")
            return 0

        self.logger.info(f"Updating trailing stops for {len(positions)} positions")

        updated_count = 0
        for position in positions:
            try:
                # Get position comment to identify strategy
                comment = position['comment']
                if not comment:
                    continue

                # Extract strategy and signal info from comment
                strategy_name = self._extract_strategy_name(comment)
                signal_id = self._extract_signal_id(comment)

                if not strategy_name:
                    continue

                # Get the trade from database
                trade = self._find_trade_by_signal_and_comment(signal_id, comment)
                if not trade or trade.close_time is not None:
                    continue

                # Get current stop loss
                current_sl = position['stop_loss']
                current_price = position['current_price']
                position_type = position['type']  # 0=BUY, 1=SELL

                # Check if it's a partial position
                is_partial = "_Part" in comment

                # Calculate new stop loss based on strategy and position
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

            except Exception as e:
                self.logger.error(f"Error updating trailing stop for position {position['ticket']}: {str(e)}")

        self.logger.info(f"Updated trailing stops for {updated_count} positions")
        return updated_count

    def _extract_strategy_name(self, comment):
        """Extract strategy name from position comment.

        Args:
            comment (str): Position comment

        Returns:
            str: Strategy name or None
        """
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
        """Extract signal ID from position comment.

        Args:
            comment (str): Position comment

        Returns:
            int: Signal ID or None
        """
        parts = comment.split('_')
        if len(parts) >= 2 and parts[0] == "Signal":
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    def _find_trade_by_signal_and_comment(self, signal_id, comment):
        """Find a trade by signal ID and comment.

        Args:
            signal_id (int): Signal ID
            comment (str): Position comment

        Returns:
            Trade: Trade object or None
        """
        if not signal_id:
            return None

        # Get all trades for this signal
        trades = self.trade_repository.get_all()
        for trade in trades:
            if trade.signal_id == signal_id and trade.comment == comment and trade.close_time is None:
                return trade

        return None

    # execution/trailing_stop.py (excerpt - update metadata references only)

    def _calculate_trailing_stop(self, position, strategy_name, trade, is_partial):
        """Calculate new trailing stop based on strategy and position type.

        Args:
            position (dict): Position information
            strategy_name (str): Strategy name
            trade (Trade): Trade object from database
            is_partial (bool): Whether this is a partial position

        Returns:
            float: New stop loss price or None if no update needed
        """
        # Get position details
        symbol = position['symbol']
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Extract metadata from trade if available
        metadata = {}
        if trade and trade.signal_id:
            # Get the signal for more info
            try:
                signal_repo = self.trade_repository._session.query("StrategySignal")  # Ideally this would be passed in
                signal = signal_repo.get_by_id(trade.signal_id)
                if signal and signal.metadata:  # Use 'metadata' instead of 'signal_metadata'
                    metadata = json.loads(signal.metadata)
            except Exception as e:
                self.logger.warning(f"Could not get signal metadata: {str(e)}")

        # Apply different trailing stop logic based on strategy
        if strategy_name == "EnhancedMA_Trend":
            return self._ma_trend_trailing_stop(position, trade, metadata)
        elif strategy_name == "Breakout":
            # For breakout strategy, trail behind the breakout level
            if position_type == 0:  # BUY
                breakout_level = metadata.get('range_top', 0)
                if breakout_level > 0 and current_price > breakout_level * 1.005:
                    return max(breakout_level, current_sl)
            else:  # SELL
                breakout_level = metadata.get('range_bottom', 0)
                if breakout_level > 0 and current_price < breakout_level * 0.995:
                    return min(breakout_level, current_sl)
        elif strategy_name == "Range_Mean_Reversion":
            # For range-bound strategy, move to breakeven once at midpoint
            midpoint = metadata.get('range_midpoint', 0)
            if midpoint > 0:
                if position_type == 0 and current_price > midpoint:  # BUY crossed midpoint
                    return max(open_price, current_sl)
                elif position_type == 1 and current_price < midpoint:  # SELL crossed midpoint
                    return min(open_price, current_sl)
        elif strategy_name == "Momentum_Scalping":
            # For momentum scalping, use tight trailing once in profit
            risk = abs(open_price - current_sl) if current_sl > 0 else 0
            if risk > 0:
                if position_type == 0 and current_price > open_price + risk:  # BUY and in 1R profit
                    # Trail 50% of ATR behind price
                    atr = metadata.get('atr', 0)
                    if atr > 0:
                        return max(current_price - (atr * 0.5), current_sl)
                elif position_type == 1 and current_price < open_price - risk:  # SELL and in 1R profit
                    # Trail 50% of ATR behind price
                    atr = metadata.get('atr', 0)
                    if atr > 0:
                        return min(current_price + (atr * 0.5), current_sl)
        elif strategy_name == "Ichimoku_Cloud":
            # For Ichimoku, trail behind the Kijun-sen
            kijun = metadata.get('kijun_sen', 0)
            if kijun > 0:
                if position_type == 0 and current_price > kijun:  # BUY
                    return max(kijun * 0.998, current_sl)  # Trail slightly below Kijun
                elif position_type == 1 and current_price < kijun:  # SELL
                    return min(kijun * 1.002, current_sl)  # Trail slightly above Kijun
        else:
            # Default trailing stop using ATR
            return self._atr_trailing_stop(position)

        return None  # No update needed

    def _ma_trend_trailing_stop(self, position, trade, metadata):
        """Calculate trailing stop for MA Trend strategy based on EMAs.

        Args:
            position (dict): Position information
            trade (Trade): Trade object
            metadata (dict): Signal metadata

        Returns:
            float: New stop loss price or None if no update needed
        """
        symbol = position['symbol']
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # First check if we should move to breakeven
        # Once price has moved the initial risk distance in our favor
        risk_distance = 0
        if current_sl > 0:
            risk_distance = abs(open_price - current_sl)

            # If we're more than 1x risk in profit, at least move to breakeven
            if position_type == 0:  # BUY
                if current_price > open_price + risk_distance and current_sl < open_price:
                    return open_price  # Move to breakeven
            else:  # SELL
                if current_price < open_price - risk_distance and current_sl > open_price:
                    return open_price  # Move to breakeven

        # Get OHLC data with EMAs
        timeframe = metadata.get('timeframe', 'H1')  # Default to H1 if not specified
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, timeframe, 50)

        if df.empty:
            return None

        # Calculate EMAs
        fast_period = metadata.get('fast_period', 20)
        df['fast_ema'] = df['close'].ewm(span=fast_period, adjust=False).mean()

        # Trail behind the fast EMA
        fast_ema = df['fast_ema'].iloc[-1]

        # For long positions, trail below the fast EMA with offset
        if position_type == 0:  # BUY
            # Trail below the fast EMA
            buffer_factor = 0.997  # 0.3% below EMA for gold
            new_sl = fast_ema * buffer_factor

            # Ensure the new SL is better than current
            if new_sl > current_sl:
                return new_sl

        # For short positions, trail above the fast EMA with offset
        elif position_type == 1:  # SELL
            # Trail above the fast EMA
            buffer_factor = 1.003  # 0.3% above EMA
            new_sl = fast_ema * buffer_factor

            # Ensure the new SL is better than current
            if new_sl < current_sl:
                return new_sl

        return None

    def _atr_trailing_stop(self, position):
        """Calculate ATR-based trailing stop.

        Args:
            position (dict): Position information

        Returns:
            float: New stop loss price or None if no update needed
        """
        symbol = position['symbol']
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data and calculate ATR
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, "H1", 20)

        if df.empty:
            return None

        # Calculate ATR (14-period)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        atr = df['atr'].iloc[-1]

        # Calculate initial risk
        if position_type == 0:  # BUY
            initial_risk = open_price - current_sl if current_sl > 0 else atr
        else:  # SELL
            initial_risk = current_sl - open_price if current_sl > 0 else atr

        # Check if price has moved favorably
        if position_type == 0:  # BUY
            price_move = current_price - open_price
            if price_move >= atr:
                # Move to breakeven initially
                if current_sl < open_price:
                    return open_price

                # Then trail with ATR
                if price_move >= 2 * atr:
                    new_sl = current_price - atr
                    if new_sl > current_sl:
                        return new_sl
        else:  # SELL
            price_move = open_price - current_price
            if price_move >= atr:
                # Move to breakeven initially
                if current_sl > open_price:
                    return open_price

                # Then trail with ATR
                if price_move >= 2 * atr:
                    new_sl = current_price + atr
                    if new_sl < current_sl:
                        return new_sl

        return None

    def _is_better_stop_loss(self, position_type, current_sl, new_sl, current_price):
        """Check if the new stop loss is better than the current one.

        Args:
            position_type (int): Position type (0=BUY, 1=SELL)
            current_sl (float): Current stop loss
            new_sl (float): New stop loss
            current_price (float): Current price

        Returns:
            bool: True if new stop loss is better, False otherwise
        """
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