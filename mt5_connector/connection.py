# mt5_connector/connection.py
import os
import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from logging.logger import app_logger
from config import Config


class MT5Connector:
    """Connector to the MetaTrader 5 terminal."""

    _instance = None
    _is_connected = False

    def __new__(cls):
        """Singleton pattern to ensure only one connection to MT5."""
        if cls._instance is None:
            cls._instance = super(MT5Connector, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the MT5 connector."""
        # Only initialize once due to singleton pattern
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._initialize_mt5()

    def _initialize_mt5(self):
        """Initialize the connection to the MT5 terminal."""
        app_logger.info("Initializing connection to MetaTrader 5...")

        if not os.path.exists(Config.MT5_TERMINAL_PATH):
            error_msg = f"MT5 terminal path does not exist: {Config.MT5_TERMINAL_PATH}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize MT5
        if not mt5.initialize(
                path=Config.MT5_TERMINAL_PATH,
                login=Config.MT5_LOGIN,
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER
        ):
            error_code = mt5.last_error()
            error_msg = f"Failed to initialize MT5: Error code {error_code}"
            app_logger.error(error_msg)
            raise ConnectionError(error_msg)

        app_logger.info("MetaTrader 5 connection initialized successfully")
        self._is_connected = True

    def ensure_connection(self):
        """Ensure the connection to MT5 is active, reconnect if necessary."""
        if not self._is_connected or not mt5.terminal_info():
            app_logger.warning("MT5 connection lost, attempting to reconnect...")
            mt5.shutdown()
            time.sleep(1)
            self._initialize_mt5()

    def disconnect(self):
        """Disconnect from the MT5 terminal."""
        if self._is_connected:
            mt5.shutdown()
            self._is_connected = False
            app_logger.info("Disconnected from MetaTrader 5")

    def get_account_info(self):
        """Get account information from MT5.

        Returns:
            dict: Account information
        """
        self.ensure_connection()

        account_info = mt5.account_info()
        if account_info is None:
            error_code = mt5.last_error()
            error_msg = f"Failed to get account info: Error code {error_code}"
            app_logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Convert to a regular dictionary
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'leverage': account_info.leverage,
            'currency': account_info.currency
        }

    def get_symbol_info(self, symbol):
        """Get symbol information from MT5.

        Args:
            symbol (str): Symbol name (e.g., 'XAUUSD')

        Returns:
            dict: Symbol information
        """
        self.ensure_connection()

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            error_code = mt5.last_error()
            error_msg = f"Failed to get symbol info for {symbol}: Error code {error_code}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Convert to a regular dictionary
        return {
            'name': symbol_info.name,
            'bid': symbol_info.bid,
            'ask': symbol_info.ask,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'min_lot': symbol_info.volume_min,
            'max_lot': symbol_info.volume_max,
            'lot_step': symbol_info.volume_step,
            'trade_mode': symbol_info.trade_mode
        }

    def get_positions(self, symbol=None):
        """Get open positions from MT5.

        Args:
            symbol (str, optional): Filter by symbol. Defaults to None (all symbols).

        Returns:
            list: Open positions
        """
        self.ensure_connection()

        positions = []
        if symbol:
            # Get positions for a specific symbol
            positions_data = mt5.positions_get(symbol=symbol)
        else:
            # Get all positions
            positions_data = mt5.positions_get()

        if positions_data is None:
            error_code = mt5.last_error()
            if error_code == 0:  # No error, just no positions
                return []
            error_msg = f"Failed to get positions: Error code {error_code}"
            app_logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Convert position_data to list of dictionaries
        for position in positions_data:
            positions.append({
                'ticket': position.ticket,
                'symbol': position.symbol,
                'type': position.type,  # 0 for buy, 1 for sell
                'volume': position.volume,
                'open_price': position.price_open,
                'open_time': position.time,
                'current_price': position.price_current,
                'stop_loss': position.sl,
                'take_profit': position.tp,
                'profit': position.profit,
                'comment': position.comment
            })

        return positions

    def place_order(self, order_type, symbol, volume, price=0.0,
                    stop_loss=0.0, take_profit=0.0, comment=""):
        """Place a new order in MT5.

        Args:
            order_type (int): Order type (0 for buy, 1 for sell)
            symbol (str): Symbol name
            volume (float): Order volume (lot size)
            price (float, optional): Price for pending orders. Defaults to 0.0 (market price).
            stop_loss (float, optional): Stop loss price. Defaults to 0.0 (none).
            take_profit (float, optional): Take profit price. Defaults to 0.0 (none).
            comment (str, optional): Order comment. Defaults to "".

        Returns:
            dict: Order result information
        """
        self.ensure_connection()

        # Validate inputs
        symbol_info = self.get_symbol_info(symbol)

        if volume < symbol_info['min_lot']:
            error_msg = f"Volume {volume} is below minimum lot size {symbol_info['min_lot']}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate volume step
        volume_steps = round(volume / symbol_info['lot_step'])
        adjusted_volume = volume_steps * symbol_info['lot_step']
        if adjusted_volume != volume:
            app_logger.warning(f"Volume adjusted from {volume} to {adjusted_volume} to match lot step")
            volume = adjusted_volume

        # Create order request
        action_map = {0: mt5.ORDER_TYPE_BUY, 1: mt5.ORDER_TYPE_SELL}
        action = action_map.get(order_type)

        if action is None:
            error_msg = f"Invalid order type: {order_type}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Get current price for market orders
        if price == 0.0:
            price = symbol_info['ask'] if order_type == 0 else symbol_info['bid']

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": action,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 10,  # Maximum price deviation in points
            "magic": 12345,  # Magic number for identification
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
            "type_filling": mt5.ORDER_FILLING_FOK,  # Fill or kill
        }

        # Send order
        app_logger.info(f"Sending {('BUY' if order_type == 0 else 'SELL')} order for {volume} lots of {symbol}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.retcode}, {result.comment}"
            app_logger.error(error_msg)
            raise RuntimeError(error_msg)

        app_logger.info(f"Order successfully placed with ticket: {result.order}")

        return {
            'ticket': result.order,
            'volume': volume,
            'price': result.price,
            'symbol': symbol,
            'type': order_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'comment': comment
        }

    def modify_position(self, ticket, stop_loss=None, take_profit=None):
        """Modify an existing position's stop loss and take profit.

        Args:
            ticket (int): Position ticket number
            stop_loss (float, optional): New stop loss price. Defaults to None (no change).
            take_profit (float, optional): New take profit price. Defaults to None (no change).

        Returns:
            bool: True if successful, False otherwise
        """
        self.ensure_connection()

        # Get the position to modify
        position = mt5.positions_get(ticket=ticket)
        if not position:
            error_msg = f"Position with ticket {ticket} not found"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        position = position[0]  # Extract position from the tuple

        # Only modify what's provided
        sl = stop_loss if stop_loss is not None else position.sl
        tp = take_profit if take_profit is not None else position.tp

        # No changes needed
        if sl == position.sl and tp == position.tp:
            app_logger.debug(f"No changes to position {ticket}, skipping modification")
            return True

        # Create modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }

        app_logger.info(f"Modifying position {ticket}: SL={sl}, TP={tp}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Position modification failed: {result.retcode}, {result.comment}"
            app_logger.error(error_msg)
            raise RuntimeError(error_msg)

        app_logger.info(f"Position {ticket} successfully modified")
        return True

    def close_position(self, ticket, volume=None):
        """Close an open position.

        Args:
            ticket (int): Position ticket number
            volume (float, optional): Volume to close. Defaults to None (close all).

        Returns:
            dict: Result of the close operation
        """
        self.ensure_connection()

        # Get the position to close
        position = mt5.positions_get(ticket=ticket)
        if not position:
            error_msg = f"Position with ticket {ticket} not found"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        position = position[0]  # Extract position from the tuple

        # If no volume specified, close the entire position
        close_volume = volume if volume is not None else position.volume

        # Validate volume
        if close_volume > position.volume:
            error_msg = f"Close volume {close_volume} exceeds position volume {position.volume}"
            app_logger.error(error_msg)
            raise ValueError(error_msg)

        # Create close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,  # Opposite of position type
            "position": ticket,
            "price": 0,  # Market price
            "deviation": 10,
            "magic": 12345,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        app_logger.info(f"Closing position {ticket}: volume={close_volume}")
        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Position close failed: {result.retcode}, {result.comment}"
            app_logger.error(error_msg)
            raise RuntimeError(error_msg)

        app_logger.info(f"Position {ticket} successfully closed: volume={close_volume}")

        return {
            'ticket': ticket,
            'close_volume': close_volume,
            'close_price': result.price,
            'profit': result.profit
        }