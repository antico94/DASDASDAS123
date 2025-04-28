# risk_management/risk_validator.py
from datetime import datetime, timedelta
from db_logger.db_logger import DBLogger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository, AccountSnapshotRepository


class RiskValidator:
    """Class to validate trading decisions against risk parameters."""

    def __init__(self, connector=None, trade_repository=None, account_repository=None):
        """Initialize the risk validator.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None (creates new).
            trade_repository (TradeRepository, optional): Trade repository. Defaults to None (creates new).
            account_repository (AccountSnapshotRepository, optional): Account repository. Defaults to None (creates new).
        """
        from config import Config

        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.account_repository = account_repository or AccountSnapshotRepository()

        self.max_positions = Config.MAX_POSITIONS
        self.max_daily_risk_percent = Config.MAX_DAILY_RISK_PERCENT
        self.max_drawdown_percent = Config.MAX_DRAWDOWN_PERCENT

    def can_open_new_position(self, symbol=None):
        """Check if a new position can be opened based on risk rules.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to None.

        Returns:
            tuple: (bool, str) - (can_open, reason)
        """
        try:
            # Check maximum positions
            current_positions = self.connector.get_positions(symbol)

            # Validate current_positions is a list and not None
            if current_positions is None:
                DBLogger.log_event("WARNING", "get_positions returned None instead of a list", "RiskValidator")
                current_positions = []

            if len(current_positions) >= self.max_positions:
                reason = f"Maximum number of positions ({self.max_positions}) already open"
                DBLogger.log_event("WARNING", reason, "RiskValidator")
                return False, reason

            # Check daily risk
            daily_loss = self._calculate_daily_loss()

            # Validate daily_loss is a number and not None
            if daily_loss is None:
                DBLogger.log_event("WARNING", "_calculate_daily_loss returned None, using 0 instead", "RiskValidator")
                daily_loss = 0

            account_info = self.connector.get_account_info()

            # Validate account_info is a dict and not None
            if account_info is None or not isinstance(account_info, dict):
                reason = "Failed to get account info"
                DBLogger.log_event("WARNING", reason, "RiskValidator")
                return False, reason

            # Validate account_info contains 'balance'
            if 'balance' not in account_info or account_info['balance'] is None:
                reason = "Account info does not contain valid balance"
                DBLogger.log_event("WARNING", reason, "RiskValidator")
                return False, reason

            daily_risk_threshold = account_info['balance'] * (self.max_daily_risk_percent / 100)

            if daily_loss >= daily_risk_threshold:
                reason = f"Daily loss ({daily_loss}) exceeds threshold ({daily_risk_threshold})"
                DBLogger.log_event("WARNING", reason, "RiskValidator")
                return False, reason

            # Check drawdown
            max_drawdown = self._calculate_drawdown()

            # Validate max_drawdown is a number and not None
            if max_drawdown is None:
                DBLogger.log_event("WARNING", "_calculate_drawdown returned None, using 0 instead", "RiskValidator")
                max_drawdown = 0

            drawdown_threshold = self.max_drawdown_percent / 100

            if max_drawdown >= drawdown_threshold:
                reason = f"Current drawdown ({max_drawdown:.2%}) exceeds threshold ({drawdown_threshold:.2%})"
                DBLogger.log_event("WARNING", reason, "RiskValidator")
                return False, reason

            # If all checks pass
            DBLogger.log_event("DEBUG", f"Risk validation passed for new position", "RiskValidator")
            return True, "All risk checks passed"

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            DBLogger.log_error("RiskValidator", f"Error in can_open_new_position", exception=e)
            return False, f"Error in risk validation: {str(e)}"

    def _calculate_daily_loss(self):
        """Calculate total loss for the current trading day.

        Returns:
            float: Total loss (positive number)
        """
        try:
            # Get today's date (server time)
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())

            # Get closed trades for today
            today_trades = self.trade_repository.get_trades_by_date_range(from_date=today_start)

            # Validate today_trades is a list
            if today_trades is None:
                DBLogger.log_event("WARNING", "get_trades_by_date_range returned None instead of a list", "RiskValidator")
                return 0

            # Sum up losses (ensure profit values are valid)
            daily_loss = 0
            for trade in today_trades:
                if trade.profit is not None and trade.profit < 0:
                    daily_loss += abs(trade.profit)

            DBLogger.log_event("DEBUG", f"Calculated daily loss: {daily_loss}", "RiskValidator")
            return daily_loss

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            DBLogger.log_error("RiskValidator", f"Error calculating daily loss", exception=e)
            return 0  # Return a safe default value

    def _calculate_drawdown(self):
        """Calculate current drawdown from peak equity.

        Returns:
            float: Current drawdown as a decimal (0.1 = 10%)
        """
        try:
            # Get account snapshots for the last 30 days
            snapshots = self.account_repository.get_daily_snapshots(days=30)

            # Validate snapshots is a list
            if snapshots is None:
                DBLogger.log_event("WARNING", "get_daily_snapshots returned None instead of a list", "RiskValidator")
                return 0

            if not snapshots:
                DBLogger.log_event("WARNING", "No account snapshots available for drawdown calculation", "RiskValidator")
                return 0

            # Find peak equity (ensure equity values are valid)
            peak_equity = 0
            for snapshot in snapshots:
                if snapshot.equity is not None and snapshot.equity > peak_equity:
                    peak_equity = snapshot.equity

            # Get current equity
            account_info = self.connector.get_account_info()

            # Validate account_info is a dict and contains 'equity'
            if account_info is None or not isinstance(account_info, dict) or 'equity' not in account_info or \
                    account_info['equity'] is None:
                DBLogger.log_event("WARNING", "Failed to get valid account equity", "RiskValidator")
                return 0

            current_equity = account_info['equity']

            # Calculate drawdown
            if peak_equity <= 0:
                return 0

            drawdown = (peak_equity - current_equity) / peak_equity

            DBLogger.log_event("DEBUG",
                f"Calculated drawdown: {drawdown:.2%} (peak: {peak_equity}, current: {current_equity})",
                "RiskValidator")
            return max(0, drawdown)  # Ensure non-negative

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            DBLogger.log_error("RiskValidator", f"Error calculating drawdown", exception=e)
            return 0  # Return a safe default value

    def validate_stop_loss(self, symbol, order_type, entry_price, stop_loss_price):
        """Validate that a stop loss is properly set and reasonable.

        Args:
            symbol (str): Symbol to trade
            order_type (int): Order type (0 for buy, 1 for sell)
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price

        Returns:
            bool: True if valid, False otherwise
        """
        # Enhanced input validation
        if symbol is None or not isinstance(symbol, str):
            DBLogger.log_event("WARNING", f"Invalid symbol: {symbol}", "RiskValidator")
            return False

        if order_type is None or not isinstance(order_type, int) or order_type not in [0, 1]:
            DBLogger.log_event("WARNING", f"Invalid order type: {order_type}", "RiskValidator")
            return False

        if entry_price is None or not isinstance(entry_price, (int, float)) or entry_price <= 0:
            DBLogger.log_event("WARNING", f"Invalid entry price: {entry_price}", "RiskValidator")
            return False

        if stop_loss_price is None or not isinstance(stop_loss_price, (int, float)) or stop_loss_price <= 0:
            DBLogger.log_event("WARNING", f"Invalid stop loss price: {stop_loss_price}", "RiskValidator")
            return False

        try:
            # Get symbol info
            symbol_info = self.connector.get_symbol_info(symbol)
            if symbol_info is None:
                DBLogger.log_event("WARNING", f"Failed to get symbol info for {symbol}", "RiskValidator")
                return False

            # Check that stop loss exists
            if stop_loss_price <= 0:
                DBLogger.log_event("WARNING", f"No stop loss set for {symbol} trade", "RiskValidator")
                return False

            # Check stop loss direction
            if order_type == 0:  # Buy
                if stop_loss_price >= entry_price:
                    DBLogger.log_event("WARNING",
                        f"Stop loss ({stop_loss_price}) is above entry price ({entry_price}) for BUY order",
                        "RiskValidator")
                    return False
            else:  # Sell
                if stop_loss_price <= entry_price:
                    DBLogger.log_event("WARNING",
                        f"Stop loss ({stop_loss_price}) is below entry price ({entry_price}) for SELL order",
                        "RiskValidator")
                    return False

            # Calculate stop loss distance
            stop_distance = abs(entry_price - stop_loss_price)
            price_point = symbol_info.get('point', 0.01)  # Default to 0.01 if not found

            # For gold, check if stop is too tight
            if symbol == "XAUUSD":
                # Minimum stop distance for gold (e.g., $3)
                min_distance = 3.0
                if stop_distance < min_distance:
                    DBLogger.log_event("WARNING", f"Stop loss distance ({stop_distance}) is too small for {symbol}", "RiskValidator")
                    return False

                # Maximum stop distance for gold (e.g., $50)
                max_distance = 50.0
                if stop_distance > max_distance:
                    DBLogger.log_event("WARNING", f"Stop loss distance ({stop_distance}) is too large for {symbol}", "RiskValidator")
                    return False

            DBLogger.log_event("DEBUG",
                f"Stop loss validation passed: {order_type}, entry={entry_price}, stop={stop_loss_price}",
                "RiskValidator")
            return True

        except Exception as e:
            DBLogger.log_error("RiskValidator", f"Error validating stop loss", exception=e)
            return False