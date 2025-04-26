# risk_management/risk_validator.py
from datetime import datetime, timedelta
from custom_logging.logger import app_logger
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

        self.logger = app_logger

    def can_open_new_position(self, symbol=None):
        """Check if a new position can be opened based on risk rules.

        Args:
            symbol (str, optional): Symbol to trade. Defaults to None.

        Returns:
            tuple: (bool, str) - (can_open, reason)
        """
        # Check maximum positions
        current_positions = self.connector.get_positions(symbol)
        if len(current_positions) >= self.max_positions:
            reason = f"Maximum number of positions ({self.max_positions}) already open"
            self.logger.warning(reason)
            return False, reason

        # Check daily risk
        daily_loss = self._calculate_daily_loss()
        account_info = self.connector.get_account_info()
        daily_risk_threshold = account_info['balance'] * (self.max_daily_risk_percent / 100)

        if daily_loss >= daily_risk_threshold:
            reason = f"Daily loss ({daily_loss}) exceeds threshold ({daily_risk_threshold})"
            self.logger.warning(reason)
            return False, reason

        # Check drawdown
        max_drawdown = self._calculate_drawdown()
        drawdown_threshold = self.max_drawdown_percent / 100

        if max_drawdown >= drawdown_threshold:
            reason = f"Current drawdown ({max_drawdown:.2%}) exceeds threshold ({drawdown_threshold:.2%})"
            self.logger.warning(reason)
            return False, reason

        # If all checks pass
        self.logger.debug(f"Risk validation passed for new position")
        return True, "All risk checks passed"

    def _calculate_daily_loss(self):
        """Calculate total loss for the current trading day.

        Returns:
            float: Total loss (positive number)
        """
        # Get today's date (server time)
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())

        # Get closed trades for today
        today_trades = self.trade_repository.get_trades_by_date_range(from_date=today_start)

        # Sum up losses
        daily_loss = sum(abs(trade.profit) for trade in today_trades if trade.profit < 0)

        self.logger.debug(f"Calculated daily loss: {daily_loss}")
        return daily_loss

    def _calculate_drawdown(self):
        """Calculate current drawdown from peak equity.

        Returns:
            float: Current drawdown as a decimal (0.1 = 10%)
        """
        # Get account snapshots for the last 30 days
        snapshots = self.account_repository.get_daily_snapshots(days=30)

        if not snapshots:
            self.logger.warning("No account snapshots available for drawdown calculation")
            return 0

        # Find peak equity
        peak_equity = max(snapshot.equity for snapshot in snapshots)

        # Get current equity
        account_info = self.connector.get_account_info()
        current_equity = account_info['equity']

        # Calculate drawdown
        if peak_equity <= 0:
            return 0

        drawdown = (peak_equity - current_equity) / peak_equity

        self.logger.debug(f"Calculated drawdown: {drawdown:.2%} (peak: {peak_equity}, current: {current_equity})")
        return max(0, drawdown)  # Ensure non-negative

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
        # Get symbol info
        symbol_info = self.connector.get_symbol_info(symbol)

        # Check that stop loss exists
        if stop_loss_price <= 0:
            self.logger.warning(f"No stop loss set for {symbol} trade")
            return False

        # Check stop loss direction
        if order_type == 0:  # Buy
            if stop_loss_price >= entry_price:
                self.logger.warning(f"Stop loss ({stop_loss_price}) is above entry price ({entry_price}) for BUY order")
                return False
        else:  # Sell
            if stop_loss_price <= entry_price:
                self.logger.warning(
                    f"Stop loss ({stop_loss_price}) is below entry price ({entry_price}) for SELL order")
                return False

        # Calculate stop loss distance
        stop_distance = abs(entry_price - stop_loss_price)
        price_point = symbol_info['point']

        # For gold, check if stop is too tight
        if symbol == "XAUUSD":
            # Minimum stop distance for gold (e.g., $3)
            min_distance = 3.0
            if stop_distance < min_distance:
                self.logger.warning(f"Stop loss distance ({stop_distance}) is too small for {symbol}")
                return False

            # Maximum stop distance for gold (e.g., $50)
            max_distance = 50.0
            if stop_distance > max_distance:
                self.logger.warning(f"Stop loss distance ({stop_distance}) is too large for {symbol}")
                return False

        self.logger.debug(f"Stop loss validation passed: {order_type}, entry={entry_price}, stop={stop_loss_price}")
        return True