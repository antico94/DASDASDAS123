# execution/trailing_stop.py
import json
import MetaTrader5 as mt5
from logging.logger import app_logger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository
from mt5_connector.data_fetcher import MT5DataFetcher


class TrailingStopManager:
    """Class to manage trailing stops for open positions."""

    def __init__(self, connector=None, trade_repository=None, data_fetcher=None):
        """Initialize the trailing stop manager.

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
        """Update trailing stops for all open positions.

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
                # Get current SL and TP
                current_sl = position['stop_loss']
                current_price = position['current_price']
                position_type = position['type']  # 0=BUY, 1=SELL

                # Get the trade from our database to check the strategy
                trade_comment = position['comment']

                # Calculate new stop loss based on trailing logic
                new_sl = self._calculate_trailing_stop(
                    position=position,
                    comment=trade_comment
                )

                # Only update if new SL is better than current
                if new_sl and self._is_better_stop_loss(position_type, current_sl, new_sl, current_price):
                    # Update the stop loss
                    self.connector.modify_position(
                        ticket=position['ticket'],
                        stop_loss=new_sl
                    )

                    self.logger.info(
                        f"Updated trailing stop for position {position['ticket']} "
                        f"({position['symbol']}): {current_sl} -> {new_sl}"
                    )

                    updated_count += 1

            except Exception as e:
                self.logger.error(f"Error updating trailing stop for position {position['ticket']}: {str(e)}")

        self.logger.info(f"Updated trailing stops for {updated_count} positions")
        return updated_count

    def _calculate_trailing_stop(self, position, comment):
        """Calculate the new trailing stop for a position.

        This method uses different trailing strategies based on the trade strategy.

        Args:
            position (dict): Position information
            comment (str): Position comment (contains strategy info)

        Returns:
            float: New stop loss price or None if no update needed
        """
        # Default trailing offset in points
        trailing_offset = 300  # 30 pips for 5-digit broker

        # Extract strategy name from comment
        strategy_name = None
        if comment:
            parts = comment.split('_')
            if len(parts) >= 3:
                strategy_name = parts[2]

        # Get position details
        position_type = position['type']  # 0=BUY, 1=SELL
        symbol = position['symbol']
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Check if we're already in profit
        is_profitable = (position_type == 0 and current_price > open_price) or \
                        (position_type == 1 and current_price < open_price)

        if not is_profitable:
            return None  # Don't trail losing positions

        # Different trailing stop strategies by strategy name
        if strategy_name == "MA_Trend":
            # For MA Trend strategy, use the EMA as trailing stop reference
            return self._ma_trend_trailing_stop(position, strategy_name)
        elif strategy_name and "Momentum" in strategy_name:
            # For Momentum Scalping, move to breakeven quickly, then trail
            return self._momentum_trailing_stop(position)
        else:
            # Default ATR-based trailing for other strategies
            return self._atr_trailing_stop(position)

    def _ma_trend_trailing_stop(self, position, strategy_name):
        """Calculate trailing stop for MA Trend strategy based on EMAs.

        Args:
            position (dict): Position information
            strategy_name (str): Strategy name

        Returns:
            float: New stop loss price or None if no update needed
        """
        symbol = position['symbol']
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # Get OHLC data with EMAs
        df = self.data_fetcher.get_latest_data_to_dataframe(symbol, "H1", 100)

        if df.empty:
            return None

        # Calculate EMAs
        df['fast_ema'] = df['close'].ewm(span=20, adjust=False).mean()
        df['slow_ema'] = df['close'].ewm(span=50, adjust=False).mean()

        # Get latest EMA values
        fast_ema = df['fast_ema'].iloc[-1]
        slow_ema = df['slow_ema'].iloc[-1]

        # For long positions, trail behind the fast EMA with offset
        if position_type == 0:  # BUY
            # First check if we should move to breakeven
            if current_price > open_price + (open_price - current_sl) and current_sl < open_price:
                return open_price  # Move to breakeven

            # Trail below the fast EMA
            new_sl = fast_ema * 0.997  # 0.3% below EMA for buffer

            # Ensure the new SL is better than current
            if new_sl > current_sl:
                return new_sl

        # For short positions, trail above the fast EMA with offset
        elif position_type == 1:  # SELL
            # First check if we should move to breakeven
            if current_price < open_price - (current_sl - open_price) and current_sl > open_price:
                return open_price  # Move to breakeven

            # Trail above the fast EMA
            new_sl = fast_ema * 1.003  # 0.3% above EMA for buffer

            # Ensure the new SL is better than current
            if new_sl < current_sl:
                return new_sl

        return None

    def _momentum_trailing_stop(self, position):
        """Calculate trailing stop for Momentum Scalping strategy.

        Args:
            position (dict): Position information

        Returns:
            float: New stop loss price or None if no update needed
        """
        position_type = position['type']  # 0=BUY, 1=SELL
        open_price = position['open_price']
        current_price = position['current_price']
        current_sl = position['stop_loss']

        # execution/trailing_stop.py (continued)
        # Calculate reward-to-risk ratio
        if position_type == 0:  # BUY
            initial_risk = open_price - current_sl if current_sl > 0 else 0
            current_reward = current_price - open_price
        else:  # SELL
            initial_risk = current_sl - open_price if current_sl > 0 else 0
            current_reward = open_price - current_price

        rr_ratio = current_reward / initial_risk if initial_risk > 0 else 0

        # Move to breakeven once we have 1:1 reward-to-risk
        if rr_ratio >= 1.0 and ((position_type == 0 and current_sl < open_price) or
                                (position_type == 1 and current_sl > open_price)):
            return open_price  # Move to breakeven

        # Trail with a 1.5 x ATR distance once we reach 2:1 reward-to-risk
        if rr_ratio >= 2.0:
            symbol = position['symbol']

            # Get OHLC data and calculate ATR
            df = self.data_fetcher.get_latest_data_to_dataframe(symbol, "M5", 20)

            if df.empty:
                return None

            # Calculate ATR (14-period)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()

            atr = df['atr'].iloc[-1]

            # Set trail distance to 1.5 x ATR
            trail_distance = 1.5 * atr

            if position_type == 0:  # BUY
                new_sl = current_price - trail_distance
                if new_sl > current_sl:
                    return new_sl
            else:  # SELL
                new_sl = current_price + trail_distance
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
        df['atr'] = df['tr'].rolling(14).mean()

        atr = df['atr'].iloc[-1]

        # Calculate initial risk
        if position_type == 0:  # BUY
            initial_risk = open_price - current_sl if current_sl > 0 else atr
        else:  # SELL
            initial_risk = current_sl - open_price if current_sl > 0 else atr

        # Check if price has moved 1 ATR in our favor
        atr_profit_threshold = atr

        if position_type == 0:  # BUY
            price_move = current_price - open_price
            if price_move >= atr_profit_threshold:
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
            if price_move >= atr_profit_threshold:
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