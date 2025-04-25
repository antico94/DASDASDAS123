# execution/order_manager.py
import json
import time
from datetime import datetime
from logging.logger import app_logger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository, StrategySignalRepository
from data.models import Trade
from risk_management.position_sizing import PositionSizer
from risk_management.risk_validator import RiskValidator


class OrderManager:
    """Class to manage order execution and trade lifecycle."""

    def __init__(self, connector=None, trade_repository=None,
                 signal_repository=None, position_sizer=None, risk_validator=None):
        """Initialize the order manager.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None.
            trade_repository (TradeRepository, optional): Trade repository. Defaults to None.
            signal_repository (StrategySignalRepository, optional): Signal repository. Defaults to None.
            position_sizer (PositionSizer, optional): Position sizer. Defaults to None.
            risk_validator (RiskValidator, optional): Risk validator. Defaults to None.
        """
        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.signal_repository = signal_repository or StrategySignalRepository()
        self.position_sizer = position_sizer or PositionSizer(connector=self.connector)
        self.risk_validator = risk_validator or RiskValidator(connector=self.connector)

        self.logger = app_logger

    def process_pending_signals(self):
        """Process all pending trade signals.

        Returns:
            int: Number of signals processed
        """
        # Get pending signals
        pending_signals = self.signal_repository.get_pending_signals()

        if not pending_signals:
            self.logger.debug("No pending signals to process")
            return 0

        self.logger.info(f"Processing {len(pending_signals)} pending signals")

        processed_count = 0
        for signal in pending_signals:
            try:
                # Process the signal
                success = self._execute_signal(signal)

                if success:
                    # Mark signal as executed
                    self.signal_repository.mark_as_executed(signal.id)
                    processed_count += 1
            except Exception as e:
                self.logger.error(f"Error processing signal {signal.id}: {str(e)}")

        self.logger.info(f"Processed {processed_count} signals")
        return processed_count

    def _execute_signal(self, signal):
        """Execute a trading signal.

        Args:
            signal (StrategySignal): The signal to execute

        Returns:
            bool: True if executed successfully, False otherwise
        """
        self.logger.info(f"Executing signal {signal.id}: {signal.signal_type} for {signal.symbol}")

        try:
            # Parse metadata
            metadata = json.loads(signal.metadata or "{}")

            if signal.signal_type in ["BUY", "SELL"]:
                # Entry signal
                return self._execute_entry_signal(signal, metadata)
            elif signal.signal_type == "CLOSE":
                # Close signal
                return self._execute_close_signal(signal, metadata)
            else:
                self.logger.warning(f"Unknown signal type: {signal.signal_type}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing signal {signal.id}: {str(e)}")
            return False

    def _execute_entry_signal(self, signal, metadata):
        """Execute an entry (BUY/SELL) signal.

        Args:
            signal (StrategySignal): The entry signal
            metadata (dict): Signal metadata

        Returns:
            bool: True if executed successfully, False otherwise
        """
        # Convert signal type to MT5 order type
        order_type = 0 if signal.signal_type == "BUY" else 1  # 0=BUY, 1=SELL

        # Get stop loss from metadata
        stop_loss = metadata.get('stop_loss', 0.0)

        # Validate stop loss
        if not self.risk_validator.validate_stop_loss(
                symbol=signal.symbol,
                order_type=order_type,
                entry_price=signal.price,
                stop_loss_price=stop_loss
        ):
            self.logger.warning(f"Invalid stop loss for signal {signal.id}")
            return False

        # Check if we can open a new position based on risk rules
        can_open, reason = self.risk_validator.can_open_new_position(signal.symbol)
        if not can_open:
            self.logger.warning(f"Risk validation failed: {reason}")
            return False

        # Calculate position size
        try:
            position_size = self.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.price,
                stop_loss_price=stop_loss
            )
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return False

        # Validate position size
        if not self.position_sizer.validate_position_size(signal.symbol, position_size):
            self.logger.warning(f"Invalid position size calculated: {position_size}")
            return False

        # Calculate take profit (if applicable)
        # Here we implement the 1:1 R:R for the first target
        take_profit = 0.0
        if stop_loss > 0:
            if order_type == 0:  # BUY
                risk_distance = signal.price - stop_loss
                take_profit = signal.price + risk_distance  # 1:1 risk:reward
            else:  # SELL
                risk_distance = stop_loss - signal.price
                take_profit = signal.price - risk_distance  # 1:1 risk:reward

        # Place the order
        try:
            # Add signal info to order comment
            comment = f"Signal_{signal.id}_{signal.strategy_name}"

            # Execute the order
            order_result = self.connector.place_order(
                order_type=order_type,
                symbol=signal.symbol,
                volume=position_size,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )

            # Record the trade in the database
            trade = Trade(
                strategy_name=signal.strategy_name,
                signal_id=signal.id,
                symbol=signal.symbol,
                order_type=signal.signal_type,
                volume=position_size,
                open_price=order_result['price'],
                open_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment
            )

            self.trade_repository.add(trade)

            self.logger.info(
                f"Successfully executed {signal.signal_type} order for {signal.symbol}: "
                f"price={order_result['price']}, volume={position_size}, "
                f"SL={stop_loss}, TP={take_profit}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return False

    def _execute_close_signal(self, signal, metadata):
        """Execute a close signal.

        Args:
            signal (StrategySignal): The close signal
            metadata (dict): Signal metadata

        Returns:
            bool: True if executed successfully, False otherwise
        """
        # Get positions for the symbol
        positions = self.connector.get_positions(signal.symbol)

        if not positions:
            self.logger.info(f"No open positions for {signal.symbol} to close")
            return True  # No action needed, consider it "executed"

        # Filter positions by strategy if strategy is specified in metadata
        target_strategy = metadata.get('strategy_name')
        if target_strategy:
            positions = [p for p in positions if
                         p['comment'].startswith(f"Signal_") and target_strategy in p['comment']]

        # Get position side to close (if specified)
        position_type = metadata.get('position_type')  # 'BUY' or 'SELL'
        if position_type:
            position_type_map = {'BUY': 0, 'SELL': 1}
            position_type_value = position_type_map.get(position_type)
            if position_type_value is not None:
                positions = [p for p in positions if p['type'] == position_type_value]

        # Check if we have positions to close
        if not positions:
            self.logger.info(
                f"No matching positions found for {signal.symbol} "
                f"(strategy: {target_strategy}, type: {position_type})"
            )
            return True  # No action needed, consider it "executed"

        # Close all matching positions
        success = True
        for position in positions:
            try:
                # Get the trade from our database
                trade_comment = position['comment']
                trade = None
                if trade_comment:
                    # Extract signal ID from comment
                    parts = trade_comment.split('_')
                    if len(parts) >= 2 and parts[0] == "Signal":
                        try:
                            signal_id = int(parts[1])
                            trades = self.trade_repository.get_all()
                            for t in trades:
                                if t.signal_id == signal_id and t.close_time is None:
                                    trade = t
                                    break
                        except:
                            pass

                # Close the position
                close_result = self.connector.close_position(position['ticket'])

                # Update the trade in our database
                if trade:
                    trade.close_price = close_result['close_price']
                    trade.close_time = datetime.utcnow()
                    trade.profit = close_result['profit']
                    self.trade_repository.update(trade)

                self.logger.info(
                    f"Closed position {position['ticket']} for {signal.symbol}: "
                    f"price={close_result['close_price']}, profit={close_result['profit']}"
                )

            except Exception as e:
                self.logger.error(f"Error closing position {position['ticket']}: {str(e)}")
                success = False

        return success