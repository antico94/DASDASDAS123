# risk_management/position_sizing.py
from custom_logging.logger import app_logger
from mt5_connector.connection import MT5Connector


class PositionSizer:
    """Class to calculate position sizes based on risk parameters."""

    def __init__(self, connector=None, max_risk_percent=None):
        """Initialize the position sizer.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None (creates new).
            max_risk_percent (float, optional): Maximum risk per trade. Defaults to None (from config).
        """
        from config import Config

        self.connector = connector or MT5Connector()
        self.max_risk_percent = max_risk_percent or Config.MAX_RISK_PER_TRADE_PERCENT
        self.logger = app_logger

    def calculate_position_size(self, symbol, entry_price, stop_loss_price):
        """Calculate position size based on risk parameters.

        Args:
            symbol (str): Symbol to trade
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price

        Returns:
            float: Calculated position size in lots
        """
        from config import Config

        # Get account info
        account_info = self.connector.get_account_info()
        account_balance = account_info['balance']

        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)

        # Calculate risk amount in account currency
        risk_amount = account_balance * (self.max_risk_percent / 100)

        # Calculate price difference
        if entry_price <= 0 or stop_loss_price <= 0:
            error_msg = f"Invalid prices: entry={entry_price}, stop_loss={stop_loss_price}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        price_difference = abs(entry_price - stop_loss_price)
        if price_difference == 0:
            error_msg = "Entry price and stop loss price cannot be equal"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Special handling for XAU/USD
        if symbol == "XAUUSD":
            # Convert pips to currency for gold
            # In gold, 1 lot (100 oz) with 1 pip ($0.01) move = $1 P&L
            risk_per_pip = risk_amount / (price_difference * 100)  # Convert dollars to pips
            position_size = risk_per_pip  # 1 lot = $1 per pip
        else:
            # For other instruments (not fully implemented)
            # Generic calculation - would need to be adapted per instrument
            risk_per_pip = risk_amount / price_difference
            position_size = risk_per_pip / 10  # Simplified

        # Adjust to symbol's lot limitations
        min_lot = symbol_info['min_lot']
        max_lot = symbol_info['max_lot']
        lot_step = symbol_info['lot_step']

        # Round to nearest lot step
        position_size = round(position_size / lot_step) * lot_step

        # Ensure within limits
        position_size = max(min_lot, min(position_size, max_lot))

        self.logger.info(
            f"Calculated position size: {position_size} lots for {symbol} "
            f"(risk: {self.max_risk_percent}%, amount: {risk_amount}, "
            f"price diff: {price_difference})"
        )

        return position_size

    def validate_position_size(self, symbol, position_size):
        """Validate that a position size is within acceptable limits.

        Args:
            symbol (str): Symbol to trade
            position_size (float): Position size to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)

        min_lot = symbol_info['min_lot']
        max_lot = symbol_info['max_lot']
        lot_step = symbol_info['lot_step']

        # Check minimum and maximum
        if position_size < min_lot:
            self.logger.warning(f"Position size {position_size} is below minimum {min_lot} for {symbol}")
            return False

        if position_size > max_lot:
            self.logger.warning(f"Position size {position_size} exceeds maximum {max_lot} for {symbol}")
            return False

        # Check lot step
        remainder = position_size % lot_step
        if remainder > 1e-10:  # Using small epsilon for float comparison
            self.logger.warning(f"Position size {position_size} is not a multiple of lot step {lot_step} for {symbol}")
            return False

        return True