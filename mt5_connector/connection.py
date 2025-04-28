# mt5_connector/connection.py
import os
import re
import time
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from db_logger.db_logger import DBLogger
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
        DBLogger.log_event("INFO", "Connecting to MetaTrader 5...", "MT5Connector")

        if not os.path.exists(Config.MT5_TERMINAL_PATH):
            error_msg = f"MT5 terminal path not found: {Config.MT5_TERMINAL_PATH}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise ValueError(error_msg)

        # Initialize MT5
        if not mt5.initialize(
                path=Config.MT5_TERMINAL_PATH,
                login=Config.MT5_LOGIN,
                password=Config.MT5_PASSWORD,
                server=Config.MT5_SERVER
        ):
            error_code = mt5.last_error()
            error_msg = f"Failed to connect to MT5: Error code {error_code}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise ConnectionError(error_msg)

        DBLogger.log_event("INFO", "MetaTrader 5 connected successfully", "MT5Connector")
        self._is_connected = True

    def ensure_connection(self):
        """Ensure the connection to MT5 is active, reconnect if necessary."""
        if not self._is_connected or not mt5.terminal_info():
            DBLogger.log_event("WARNING", "MT5 connection lost, reconnecting...", "MT5Connector")
            mt5.shutdown()
            time.sleep(1)
            self._initialize_mt5()

    def disconnect(self):
        """Disconnect from the MT5 terminal."""
        if self._is_connected:
            mt5.shutdown()
            self._is_connected = False
            DBLogger.log_event("INFO", "Disconnected from MetaTrader 5", "MT5Connector")

    def get_account_info(self):
        """Get account information from MT5.

        Returns:
            dict: Account information
        """
        try:
            self.ensure_connection()

            account_info = mt5.account_info()
            if account_info is None:
                error_code = mt5.last_error()
                error_msg = f"Failed to get account info: Error code {error_code}"
                DBLogger.log_error("MT5Connector", error_msg)
                # Return a default dictionary with zeros instead of raising an exception
                return {
                    'balance': 0.0,
                    'equity': 0.0,
                    'margin': 0.0,
                    'free_margin': 0.0,
                    'margin_level': 0.0,
                    'leverage': 1,
                    'currency': 'USD'
                }

            # Convert to a regular dictionary with validation
            result = {
                'balance': account_info.balance if hasattr(account_info, 'balance') else 0.0,
                'equity': account_info.equity if hasattr(account_info, 'equity') else 0.0,
                'margin': account_info.margin if hasattr(account_info, 'margin') else 0.0,
                'free_margin': account_info.margin_free if hasattr(account_info, 'margin_free') else 0.0,
                'margin_level': account_info.margin_level if hasattr(account_info, 'margin_level') else 0.0,
                'leverage': account_info.leverage if hasattr(account_info, 'leverage') else 1,
                'currency': account_info.currency if hasattr(account_info, 'currency') else 'USD'
            }

            return result
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            DBLogger.log_error("MT5Connector", f"Error getting account info: {str(e)}", exception=e)
            # Return a default dictionary with zeros
            return {
                'balance': 0.0,
                'equity': 0.0,
                'margin': 0.0,
                'free_margin': 0.0,
                'margin_level': 0.0,
                'leverage': 1,
                'currency': 'USD'
            }

    def get_symbol_info(self, symbol):
        """Get symbol information from MT5.

        Args:
            symbol (str): Symbol name (e.g., 'XAUUSD')

        Returns:
            dict: Symbol information
        """
        self.ensure_connection()

        if symbol is None or not isinstance(symbol, str):
            error_msg = f"Invalid symbol: {symbol}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise ValueError(error_msg)

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                error_code = mt5.last_error()
                error_msg = f"Failed to get info for {symbol}: Error code {error_code}"
                DBLogger.log_error("MT5Connector", error_msg)
                raise ValueError(error_msg)

            # Convert to a regular dictionary with validation
            result = {
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

            # Validate critical values
            for key in ['min_lot', 'max_lot', 'lot_step']:
                if result[key] is None:
                    error_msg = f"Symbol info for {symbol} has None value for {key}"
                    DBLogger.log_error("MT5Connector", error_msg)
                    # Use default values instead of None
                    if key == 'min_lot':
                        result[key] = 0.01
                    elif key == 'max_lot':
                        result[key] = 10.0
                    elif key == 'lot_step':
                        result[key] = 0.01

            return result
        except Exception as e:
            DBLogger.log_error("MT5Connector", f"Error getting symbol info: {str(e)}", exception=e)
            # Return safe default values instead of raising an exception
            return {
                'name': symbol,
                'bid': 0.0,
                'ask': 0.0,
                'point': 0.01,
                'digits': 2,
                'min_lot': 0.01,
                'max_lot': 10.0,
                'lot_step': 0.01,
                'trade_mode': 0
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
            DBLogger.log_error("MT5Connector", error_msg)
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
        """Place a new order in MT5."""
        self.ensure_connection()
        comment = self._sanitize_comment(comment)
        # Validate inputs
        symbol_info = self.get_symbol_info(symbol)

        if volume < symbol_info['min_lot']:
            error_msg = f"Volume {volume} is below minimum lot size {symbol_info['min_lot']}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise ValueError(error_msg)

        # Validate volume step
        volume_steps = round(volume / symbol_info['lot_step'])
        adjusted_volume = volume_steps * symbol_info['lot_step']
        if abs(adjusted_volume - volume) > 1e-10:  # Using epsilon for float comparison
            DBLogger.log_event("WARNING", f"Volume adjusted from {volume} to {adjusted_volume} to match lot step",
                               "MT5Connector")
            volume = adjusted_volume

        # Create order request
        action_map = {0: mt5.ORDER_TYPE_BUY, 1: mt5.ORDER_TYPE_SELL}
        action = action_map.get(order_type)

        if action is None:
            error_msg = f"Invalid order type: {order_type}"
            DBLogger.log_error("MT5Connector", error_msg)
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

        # Log order request
        order_desc = 'BUY' if order_type == 0 else 'SELL'
        DBLogger.log_order_request(
            order_type=order_desc,
            symbol=symbol,
            volume=volume,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            message=f"Placing {order_desc} order: {volume} lots of {symbol} at ${price:.2f}"
        )

        try:
            result = mt5.order_send(request)

            if result is None:
                error_code = mt5.last_error()
                error_msg = f"Order failed: MT5 returned None, error code: {error_code}"
                DBLogger.log_error("MT5Connector", error_msg)
                raise RuntimeError(error_msg)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.retcode}, {result.comment}"
                DBLogger.log_error("MT5Connector", error_msg)
                raise RuntimeError(error_msg)

            # Log successful order execution
            DBLogger.log_order_execution(
                execution_type="OPENED",
                symbol=symbol,
                volume=volume,
                price=result.price,
                ticket=result.order,
                message=f"Order placed successfully: Ticket #{result.order}"
            )

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
        except Exception as e:
            DBLogger.log_error("MT5Connector", f"Exception placing order: {str(e)}", exception=e)
            raise  # Re-raise to handle in calling function

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
            DBLogger.log_error("MT5Connector", error_msg)
            raise ValueError(error_msg)

        position = position[0]  # Extract position from the tuple

        # Only modify what's provided
        sl = stop_loss if stop_loss is not None else position.sl
        tp = take_profit if take_profit is not None else position.tp

        # No changes needed
        if sl == position.sl and tp == position.tp:
            DBLogger.log_event("DEBUG", f"No changes needed for position {ticket}", "MT5Connector")
            return True

        # Create modification request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp
        }

        # Log modification request
        DBLogger.log_order_request(
            order_type="MODIFY",
            symbol=position.symbol,
            ticket=ticket,
            stop_loss=sl,
            take_profit=tp,
            message=f"Modifying position {ticket}: SL=${sl:.2f}, TP=${tp:.2f}"
        )

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Position modification failed: {result.retcode}, {result.comment}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise RuntimeError(error_msg)

        # Log successful modification
        DBLogger.log_position(
            position_type="MODIFIED",
            symbol=position.symbol,
            ticket=ticket,
            volume=position.volume,
            stop_loss=sl,
            take_profit=tp,
            message=f"Position {ticket} modified: SL=${sl:.2f}, TP=${tp:.2f}"
        )

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
            DBLogger.log_error("MT5Connector", error_msg)
            raise ValueError(error_msg)

        position = position[0]  # Extract position from the tuple

        # If no volume specified, close the entire position
        close_volume = volume if volume is not None else position.volume

        # Validate volume
        if close_volume > position.volume:
            error_msg = f"Close volume {close_volume} exceeds position volume {position.volume}"
            DBLogger.log_error("MT5Connector", error_msg)
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

        # Log close request
        DBLogger.log_order_request(
            order_type="CLOSE",
            symbol=position.symbol,
            volume=close_volume,
            ticket=ticket,
            message=f"Closing position {ticket}: volume={close_volume}"
        )

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Position close failed: {result.retcode}, {result.comment}"
            DBLogger.log_error("MT5Connector", error_msg)
            raise RuntimeError(error_msg)

        # Log close operation success
        is_partial = close_volume < position.volume
        position_type = "PARTIAL_CLOSE" if is_partial else "CLOSED"

        DBLogger.log_position(
            position_type=position_type,
            symbol=position.symbol,
            ticket=ticket,
            volume=close_volume,
            current_price=result.price,
            profit=result.profit,
            message=f"Position {ticket} closed: price=${result.price:.2f}, profit=${result.profit:.2f}"
        )

        return {
            'ticket': ticket,
            'close_volume': close_volume,
            'close_price': result.price,
            'profit': result.profit
        }

    @staticmethod
    def _sanitize_comment(comment):
        """
        Sanitize the comment for MT5 trade requests based on common strict broker rules.
        Allows ONLY letters (a-z, A-Z) and digits (0-9).
        Truncates to a max length of 63.
        Provides a default if sanitization results in an empty string.
        """
        # Ensure the input is treated as a string
        try:
            comment_str = str(comment)
        except Exception:
            # Handle cases where str() might fail for very unusual inputs
            comment_str = ""
            DBLogger.log_event("WARNING", f"Could not convert comment to string: {comment}. Using empty string.",
                               "MT5Connector")

        # --- MODIFIED REGEX ---
        # Remove any characters NOT in the allowed set: letters, digits
        # This removes underscores, hash, brackets, spaces, punctuation etc.
        sanitized = re.sub(r'[^a-zA-Z0-9]', '', comment_str)
        # ----------------------

        # Truncate to the maximum allowed length (63 characters for MT5 comments)
        max_length = 63
        sanitized = sanitized[:max_length]

        # If sanitization results in an empty string, use a safe default comment
        # This prevents sending a blank comment if the original was all disallowed characters
        if not sanitized:
            sanitized = "BotOrder"  # Or use a more specific default like "Signal"

        # Add a debug log here to see the final comment being used
        DBLogger.log_event("DEBUG", f"Original comment: '{comment_str}' -> Sanitized comment: '{sanitized}'",
                           "MT5Connector")

        return sanitized