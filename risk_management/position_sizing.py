# risk_management/position_sizing.py - Updated with ATR-based sizing
import numpy as np
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

    def calculate_position_size(self, symbol, entry_price, stop_loss_price, atr_value=None):
        """Calculate position size based on risk parameters.

        Args:
            symbol (str): Symbol to trade
            entry_price (float): Entry price
            stop_loss_price (float): Stop loss price
            atr_value (float, optional): Current ATR value for ATR-based sizing

        Returns:
            float: Calculated position size in lots
        """
        # Validate inputs
        if entry_price is None or stop_loss_price is None:
            error_msg = f"Invalid prices: entry={entry_price}, stop_loss={stop_loss_price}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Get account info
            account_info = self.connector.get_account_info()
            if account_info is None:
                error_msg = "Failed to get account info"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            account_balance = account_info.get('balance')

            # Calculate risk amount in account currency
            risk_amount = account_balance * (self.max_risk_percent / 100)
            self.logger.debug(f"Risk amount: {risk_amount} ({self.max_risk_percent}% of {account_balance})")

            # Calculate price difference for stop loss
            price_difference = abs(entry_price - stop_loss_price)

            # If ATR value is provided, we can validate the stop distance
            if atr_value is not None:
                # Plan recommends stop at 1.5 × ATR
                recommended_stop_distance = atr_value * 1.5

                # Check if stop is too tight compared to ATR recommendation
                if price_difference < recommended_stop_distance * 0.5:
                    self.logger.warning(
                        f"Stop loss may be too tight: {price_difference} < {recommended_stop_distance / 2} "
                        f"(half of recommended 1.5×ATR={recommended_stop_distance})"
                    )

                # Check if stop is too wide compared to ATR recommendation
                elif price_difference > recommended_stop_distance * 2:
                    self.logger.warning(
                        f"Stop loss may be too wide: {price_difference} > {recommended_stop_distance * 2} "
                        f"(double of recommended 1.5×ATR={recommended_stop_distance})"
                    )

            # Calculate position size
            # Special handling for XAU/USD
            if symbol == "XAUUSD":
                # Convert risk amount to position size for gold
                # For gold, 1 lot (100 oz) with price_difference move = $price_difference * 100 P&L
                position_size = risk_amount / (price_difference * 100)
            else:
                # For other instruments (not fully implemented)
                # Would need to be adapted per instrument
                position_size = risk_amount / (price_difference * 10)

            # Get symbol info for position size constraints
            symbol_info = self.connector.get_symbol_info(symbol)
            min_lot = symbol_info['min_lot']
            max_lot = symbol_info['max_lot']
            lot_step = symbol_info['lot_step']

            # Round to nearest lot step with proper precision
            position_size = round(position_size / lot_step) * lot_step

            # Fix: Ensure position_size is properly rounded to avoid floating point errors
            position_size = round(position_size, 5)  # This helps with floating point precision issues

            # Ensure within limits
            position_size = max(min_lot, min(position_size, max_lot))

            # Final validation to ensure it's a valid lot size
            if (position_size % lot_step) > 1e-5:  # Use larger epsilon for final check
                position_size = round(position_size / lot_step) * lot_step
                position_size = round(position_size, 5)  # Re-round to ensure precision

            self.logger.info(
                f"Calculated position size: {position_size} lots for {symbol} "
                f"(risk: {self.max_risk_percent}%, amount: {risk_amount}, "
                f"price diff: {price_difference}, "
                f"ATR reference: {atr_value if atr_value else 'N/A'})"
            )

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Minimum lot size as fallback

    def calculate_atr_based_stop(self, current_price, atr_value, order_type, multiplier=1.5):
        """Calculate stop loss price based on ATR value.

        Args:
            current_price (float): Current price
            atr_value (float): ATR value
            order_type (int): Order type (0=BUY, 1=SELL)
            multiplier (float): ATR multiplier (default 1.5 per the plan)

        Returns:
            float: Calculated stop loss price
        """
        if atr_value <= 0:
            self.logger.warning(f"Invalid ATR value: {atr_value}")
            # Default to 1% as emergency fallback
            atr_value = current_price * 0.01

        stop_distance = atr_value * multiplier

        if order_type == 0:  # BUY
            stop_loss = current_price - stop_distance
        else:  # SELL
            stop_loss = current_price + stop_distance

        self.logger.info(
            f"Calculated ATR-based stop: {stop_loss} "
            f"({multiplier}×ATR={stop_distance} from price {current_price})"
        )

        return stop_loss

    def calculate_reward_targets(self, entry_price, stop_loss, order_type, ratios=[1.0, 2.0]):
        """Calculate take profit targets based on risk-reward ratios.

        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            order_type (int): Order type (0=BUY, 1=SELL)
            ratios (list): List of risk-reward ratios for targets

        Returns:
            list: Calculated take profit prices
        """
        risk = abs(entry_price - stop_loss)
        targets = []

        for ratio in ratios:
            if order_type == 0:  # BUY
                target = entry_price + (risk * ratio)
            else:  # SELL
                target = entry_price - (risk * ratio)

            targets.append(target)

        self.logger.info(
            f"Calculated reward targets: {targets} "
            f"(based on {ratios} × risk={risk})"
        )

        return targets

    # risk_management/position_sizing.py - Add the missing method

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

        if not symbol_info:
            self.logger.warning(f"Could not get symbol info for {symbol}")
            return False

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
        # remainder = position_size % lot_step
        # if remainder > 1e-10:  # Using small epsilon for float comparison
        #     self.logger.warning(f"Position size {position_size} is not a multiple of lot step {lot_step} for {symbol}")
        #     return False

        return True