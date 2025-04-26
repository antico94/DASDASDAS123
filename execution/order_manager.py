# execution/enhanced_order_manager.py
import json
import time
from datetime import datetime
from custom_logging.logger import app_logger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository, StrategySignalRepository
from data.models import Trade
from risk_management.position_sizing import PositionSizer
from risk_management.risk_validator import RiskValidator


class EnhancedOrderManager:
    """Enhanced class to manage order execution with partial profit-taking and trailing stops."""

    def __init__(self, connector=None, trade_repository=None,
                 signal_repository=None, position_sizer=None, risk_validator=None):
        """Initialize the enhanced order manager.

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
        """Process all pending trade signals with enhanced trade management.

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

    # execution/order_manager.py - _execute_signal method
    # execution/enhanced_order_manager.py (excerpt - update metadata references only)

    def _execute_signal(self, signal):
        """Execute a trading signal with enhanced trade management.

        Args:
            signal (StrategySignal): The signal to execute

        Returns:
            bool: True if executed successfully, False otherwise
        """
        self.logger.info(f"Executing signal {signal.id}: {signal.signal_type} for {signal.symbol}")

        try:
            # Parse metadata - use 'metadata' instead of 'signal_metadata'
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
        """Execute an entry (BUY/SELL) signal with enhanced position management.

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

        # Check if the signal suggests partial profit-taking
        take_profit_1 = metadata.get('take_profit_1r', 0.0)
        take_profit_2 = metadata.get('take_profit_2r', 0.0)

        if take_profit_1 > 0 and take_profit_2 > 0:
            # Use multi-position profit-taking strategy
            return self._execute_with_partial_profit(signal, metadata, order_type, stop_loss,
                                                     take_profit_1, take_profit_2)
        else:
            # Use standard single position strategy
            return self._execute_standard_entry(signal, metadata, order_type, stop_loss)

    def _execute_with_partial_profit(self, signal, metadata, order_type, stop_loss,
                                     take_profit_1, take_profit_2):
        """Execute an entry with partial profit-taking plan by opening two separate positions.

        Args:
            signal (StrategySignal): The entry signal
            metadata (dict): Signal metadata
            order_type (int): Order type (0=BUY, 1=SELL)
            stop_loss (float): Initial stop loss price
            take_profit_1 (float): First target for partial profit-taking
            take_profit_2 (float): Second target for the remainder

        Returns:
            bool: True if executed successfully, False otherwise
        """
        self.logger.info(f"Executing with partial profit strategy for signal {signal.id}")

        try:
            # Calculate total position size
            total_position_size = self.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.price,
                stop_loss_price=stop_loss
            )

            # Validate position size
            if not self.position_sizer.validate_position_size(signal.symbol, total_position_size):
                self.logger.warning(f"Invalid position size calculated: {total_position_size}")
                return False

            # Split into two equal parts
            position_size_1 = total_position_size * 0.5
            position_size_2 = total_position_size * 0.5

            # Round to allowed lot sizes if needed
            symbol_info = self.connector.get_symbol_info(signal.symbol)
            lot_step = symbol_info['lot_step']

            position_size_1 = round(position_size_1 / lot_step) * lot_step
            position_size_2 = total_position_size - position_size_1

            # Ensure minimum lot sizes
            min_lot = symbol_info['min_lot']
            if position_size_1 < min_lot or position_size_2 < min_lot:
                # Can't split, use single position approach
                self.logger.warning(f"Position size too small to split: {total_position_size}. Using single position.")
                return self._execute_standard_entry(signal, metadata, order_type, stop_loss)

            # Create comments for the two positions
            comment_1 = f"Signal_{signal.id}_{signal.strategy_name}_Part1"
            comment_2 = f"Signal_{signal.id}_{signal.strategy_name}_Part2"

            # Place first position with the first target
            order_result_1 = self.connector.place_order(
                order_type=order_type,
                symbol=signal.symbol,
                volume=position_size_1,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_1,
                comment=comment_1
            )

            # Record the first trade
            trade_1 = Trade(
                strategy_name=signal.strategy_name,
                signal_id=signal.id,
                symbol=signal.symbol,
                order_type=signal.signal_type,
                volume=position_size_1,
                open_price=order_result_1['price'],
                open_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit_1,
                comment=comment_1
            )
            self.trade_repository.add(trade_1)

            # Place second position with further target
            order_result_2 = self.connector.place_order(
                order_type=order_type,
                symbol=signal.symbol,
                volume=position_size_2,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_2,
                comment=comment_2
            )

            # Record the second trade
            trade_2 = Trade(
                strategy_name=signal.strategy_name,
                signal_id=signal.id,
                symbol=signal.symbol,
                order_type=signal.signal_type,
                volume=position_size_2,
                open_price=order_result_2['price'],
                open_time=datetime.utcnow(),
                stop_loss=stop_loss,
                take_profit=take_profit_2,
                comment=comment_2
            )
            self.trade_repository.add(trade_2)

            self.logger.info(
                f"Successfully executed {signal.signal_type} with partial profit strategy: "
                f"position 1: price={order_result_1['price']}, volume={position_size_1}, "
                f"SL={stop_loss}, TP1={take_profit_1}, "
                f"position 2: price={order_result_2['price']}, volume={position_size_2}, "
                f"SL={stop_loss}, TP2={take_profit_2}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error executing partial profit strategy: {str(e)}")
            return False

    def _execute_standard_entry(self, signal, metadata, order_type, stop_loss):
        """Execute a standard entry (single position).

        Args:
            signal (StrategySignal): The entry signal
            metadata (dict): Signal metadata
            order_type (int): Order type (0=BUY, 1=SELL)
            stop_loss (float): Stop loss price

        Returns:
            bool: True if executed successfully, False otherwise
        """
        try:
            # Calculate position size
            position_size = self.position_sizer.calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.price,
                stop_loss_price=stop_loss
            )

            # Validate position size
            if not self.position_sizer.validate_position_size(signal.symbol, position_size):
                self.logger.warning(f"Invalid position size calculated: {position_size}")
                return False

            # Calculate take profit if applicable
            take_profit = metadata.get('take_profit_1r', 0.0)  # Use first target if available
            if not take_profit:
                take_profit = self._calculate_default_take_profit(signal.price, stop_loss, order_type)

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
        """Execute a close signal to exit an existing position.

        Args:
            signal (StrategySignal): The close signal
            metadata (dict): Signal metadata

        Returns:
            bool: True if executed successfully, False otherwise
        """
        self.logger.info(f"Executing CLOSE signal for {signal.symbol}")

        try:
            # Get the positions to close
            positions = self.connector.get_positions(signal.symbol)

            if not positions:
                self.logger.warning(f"No open positions found for {signal.symbol} to close")
                return False

            # Extract position filter criteria from metadata if provided
            strategy_filter = metadata.get('strategy', None)
            ticket_filter = metadata.get('ticket', None)

            closed_count = 0

            for position in positions:
                # Apply filters if specified
                if ticket_filter and position['ticket'] != ticket_filter:
                    continue

                if strategy_filter and strategy_filter not in position['comment']:
                    continue

                # Close the position
                close_result = self.connector.close_position(position['ticket'])

                # Record the trade update in database
                if close_result:
                    # Find the trade record and update it
                    trades = self.trade_repository.get_open_trades(symbol=signal.symbol)
                    for trade in trades:
                        # Match by ticket number if included in comment
                        if str(position['ticket']) in trade.comment:
                            trade.close_price = close_result['close_price']
                            trade.close_time = datetime.utcnow()
                            trade.profit = close_result['profit']
                            self.trade_repository.update(trade)
                            break

                    self.logger.info(
                        f"Closed position {position['ticket']} for {signal.symbol} at {close_result['close_price']}, "
                        f"profit: {close_result['profit']}"
                    )
                    closed_count += 1

            if closed_count > 0:
                return True
            else:
                self.logger.warning(f"No matching positions were closed for {signal.symbol}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing close signal: {str(e)}")
            return False

    def _calculate_default_take_profit(self, entry_price, stop_loss, order_type):
        """Calculate a default take profit price based on risk-to-reward ratio.

        Args:
            entry_price (float): Position entry price
            stop_loss (float): Stop loss price
            order_type (int): Order type (0=BUY, 1=SELL)

        Returns:
            float: Calculated take profit price
        """
        # Default risk-to-reward ratio of 1:1.5
        risk_reward_ratio = 1.5

        # Calculate risk (distance from entry to stop)
        if order_type == 0:  # BUY
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:  # SELL
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * risk_reward_ratio)

        # Validate take profit (ensure it's reasonable)
        if risk <= 0:
            self.logger.warning(f"Invalid risk calculation: entry={entry_price}, stop={stop_loss}")
            # Use a default 1% move if risk calculation failed
            if order_type == 0:  # BUY
                take_profit = entry_price * 1.01
            else:  # SELL
                take_profit = entry_price * 0.99

        # Log the calculation
        self.logger.debug(
            f"Calculated default take profit for {order_type}: "
            f"entry={entry_price}, stop={stop_loss}, take_profit={take_profit}"
        )

        return take_profit