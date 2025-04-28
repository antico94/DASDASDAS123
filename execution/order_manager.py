# execution/order_manager.py
import json
import time
from datetime import datetime
from db_logger.db_logger import DBLogger
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
                # Skip signals marked as simulation only
                if signal.comment == "SIMULATION_MODE":
                    self.logger.info(f"Skipping signal {signal.id} marked as simulation")
                    self.signal_repository.mark_as_executed(signal.id)
                    processed_count += 1
                    continue

                # Process the signal
                success = self._execute_signal(signal)

                if success:
                    # Mark signal as executed
                    self.signal_repository.mark_as_executed(signal.id)
                    processed_count += 1
            except Exception as e:
                self.logger.error(f"Error processing signal {signal.id}: {str(e)}")

        if processed_count > 0:
            self.logger.info(f"Processed {processed_count} signals")
        return processed_count

    def _execute_signal(self, signal):
        """Execute a trading signal with enhanced trade management.

        Args:
            signal (StrategySignal): The signal to execute

        Returns:
            bool: True if executed successfully, False otherwise
        """
        self.logger.info(f"Executing signal {signal.id}: {signal.signal_type} for {signal.symbol}")

        try:
            # Parse metadata
            metadata = json.loads(signal.signal_data or "{}")

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
        # Input validation
        if signal is None:
            self.logger.error("Signal is None")
            return False

        if signal.signal_type not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid signal type: {signal.signal_type}")
            return False

        if signal.price is None or not isinstance(signal.price, (int, float)) or signal.price <= 0:
            self.logger.error(f"Invalid signal price: {signal.price}")
            return False

        if signal.symbol is None or not isinstance(signal.symbol, str):
            self.logger.error(f"Invalid symbol: {signal.symbol}")
            return False

        # Convert signal type to MT5 order type
        order_type = 0 if signal.signal_type == "BUY" else 1  # 0=BUY, 1=SELL

        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
            self.logger.warning("Metadata is None, using empty dictionary")

        # Calculate stop loss based on strategy
        stop_loss = self._calculate_strategy_stop_loss(signal, metadata, order_type)

        # Log the stop loss calculation
        self.logger.debug(f"Calculated stop loss: {stop_loss} for {signal.signal_type} signal")

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

        # Calculate position size based on risk
        atr_value = metadata.get('atr', None)
        position_size = self.position_sizer.calculate_position_size(
            symbol=signal.symbol,
            entry_price=signal.price,
            stop_loss_price=stop_loss,
            atr_value=atr_value
        )

        # Validate position size
        if position_size is None or not self.position_sizer.validate_position_size(signal.symbol, position_size):
            self.logger.warning(f"Invalid position size calculated: {position_size}")
            return False

        # Calculate take-profit levels based on risk-reward ratios and strategy
        take_profit_levels = self._calculate_strategy_take_profits(
            signal, metadata, stop_loss, order_type
        )

        # Use two-part trade execution as specified in the plan
        return self._execute_with_partial_profit(
            signal, metadata, order_type, stop_loss, position_size, take_profit_levels
        )

    def _calculate_strategy_stop_loss(self, signal, metadata, order_type):
        """Calculate appropriate stop loss based on strategy type.

        Args:
            signal (StrategySignal): The strategy signal
            metadata (dict): Signal metadata
            order_type (int): Order type (0=BUY, 1=SELL)

        Returns:
            float: Calculated stop loss price
        """
        strategy_name = signal.strategy_name

        # Get stop loss from metadata if available
        stop_loss = metadata.get('stop_loss')

        # If no stop_loss in metadata or invalid, calculate based on strategy
        if stop_loss is None or not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
            # Get ATR if available for ATR-based stops
            atr_value = metadata.get('atr')
            current_price = signal.price

            if strategy_name == "Breakout":
                # For Breakout strategy: stop at 1.5×ATR from entry (per plan)
                if atr_value:
                    if order_type == 0:  # BUY
                        stop_loss = current_price - (atr_value * 1.5)
                    else:  # SELL
                        stop_loss = current_price + (atr_value * 1.5)
                else:
                    # Fallback: just outside broken level
                    if order_type == 0:  # BUY
                        level = metadata.get('range_bottom', 0)
                        stop_loss = level * 0.997 if level > 0 else current_price * 0.995
                    else:  # SELL
                        level = metadata.get('range_top', 0)
                        stop_loss = level * 1.003 if level > 0 else current_price * 1.005

            elif strategy_name == "Momentum_Scalping":
                # For Momentum: below recent swing low or EMA-based
                swing_low = metadata.get('swing_low', 0)
                swing_high = metadata.get('swing_high', 0)
                ema = metadata.get('ema', 0)

                if order_type == 0:  # BUY
                    if swing_low > 0:
                        stop_loss = swing_low
                    elif ema > 0:
                        # 15-20 pips below EMA (approximately $1.5-2.0 for gold)
                        stop_loss = ema - 2.0
                    else:
                        # Fallback
                        stop_loss = current_price * 0.995
                else:  # SELL
                    if swing_high > 0:
                        stop_loss = swing_high
                    elif ema > 0:
                        stop_loss = ema + 2.0
                    else:
                        stop_loss = current_price * 1.005

            elif strategy_name == "Ichimoku_Cloud":
                # For Ichimoku: below cloud or Kijun-sen
                cloud_bottom = metadata.get('senkou_span_b', 0)
                kijun = metadata.get('kijun_sen', 0)

                if order_type == 0:  # BUY
                    if cloud_bottom > 0 and cloud_bottom < current_price:
                        stop_loss = cloud_bottom * 0.997
                    elif kijun > 0:
                        stop_loss = kijun * 0.997
                    else:
                        stop_loss = current_price * 0.99
                else:  # SELL
                    cloud_top = metadata.get('senkou_span_a', 0)
                    if cloud_top > 0 and cloud_top > current_price:
                        stop_loss = cloud_top * 1.003
                    elif kijun > 0:
                        stop_loss = kijun * 1.003
                    else:
                        stop_loss = current_price * 1.01

            elif strategy_name == "Range_Mean_Reversion":
                # For Range strategy: just outside the range boundary
                if order_type == 0:  # BUY at support
                    range_bottom = metadata.get('range_bottom', 0)
                    if range_bottom > 0:
                        stop_loss = range_bottom * 0.997  # Just below support
                    else:
                        stop_loss = current_price * 0.995
                else:  # SELL at resistance
                    range_top = metadata.get('range_top', 0)
                    if range_top > 0:
                        stop_loss = range_top * 1.003  # Just above resistance
                    else:
                        stop_loss = current_price * 1.005

            else:  # Moving Average or default
                # For MA strategy: below recent swing low/high
                if order_type == 0:  # BUY
                    swing_low = metadata.get('swing_low', 0)
                    if swing_low > 0:
                        stop_loss = swing_low
                    elif atr_value:
                        stop_loss = current_price - (atr_value * 1.5)
                    else:
                        stop_loss = current_price * 0.995
                else:  # SELL
                    swing_high = metadata.get('swing_high', 0)
                    if swing_high > 0:
                        stop_loss = swing_high
                    elif atr_value:
                        stop_loss = current_price + (atr_value * 1.5)
                    else:
                        stop_loss = current_price * 1.005

        # Final validation
        if stop_loss <= 0:
            self.logger.warning(f"Invalid calculated stop loss: {stop_loss}, using fallback")
            # Fallback: 1% from price
            stop_loss = signal.price * 0.99 if order_type == 0 else signal.price * 1.01

        # Log the strategy-specific stop calculation
        self.logger.debug(
            f"Calculated {strategy_name} stop loss: {stop_loss} "
            f"for {signal.signal_type} at {signal.price}"
        )

        return stop_loss

    def _calculate_strategy_take_profits(self, signal, metadata, stop_loss, order_type):
        """Calculate take-profit levels based on strategy requirements.

        Args:
            signal (StrategySignal): The strategy signal
            metadata (dict): Signal metadata
            stop_loss (float): Stop loss price
            order_type (int): Order type (0=BUY, 1=SELL)

        Returns:
            list: List of take-profit prices [target1, target2]
        """
        strategy_name = signal.strategy_name
        entry_price = signal.price

        # Calculate the risk (distance from entry to stop)
        risk = abs(entry_price - stop_loss)

        # Default risk-reward ratios
        r1 = 1.0  # First target at 1:1 (per plan)
        r2 = 2.0  # Second target at 2:1

        # Strategy-specific adjustments
        if strategy_name == "Breakout":
            # Breakout strategy per plan:
            # "take half off at 1×ATR profit and move stop to breakeven"
            r1 = 1.0
            r2 = 2.0  # Can also use projection based on range height

            # Range height projection as possible second target
            range_top = metadata.get('range_top', 0)
            range_bottom = metadata.get('range_bottom', 0)
            if range_top > 0 and range_bottom > 0 and (range_top > range_bottom):
                range_height = range_top - range_bottom

                if order_type == 0:  # BUY
                    projected_target = entry_price + range_height
                    r2 = min(3.0, (projected_target - entry_price) / risk)
                else:  # SELL
                    projected_target = entry_price - range_height
                    r2 = min(3.0, (entry_price - projected_target) / risk)

        elif strategy_name == "Momentum_Scalping":
            # Momentum strategy: "Take half profit at +1 R (i.e., a reward equal to that risk)"
            r1 = 1.0
            r2 = 2.0  # For the second half, can be larger if momentum continues

        elif strategy_name == "Ichimoku_Cloud":
            # Ichimoku strategy: "1.5:1 reward-to-risk for first target, 3:1 for remainder"
            r1 = 1.5
            r2 = 3.0

        elif strategy_name == "Range_Mean_Reversion":
            # Range strategy: midpoint and opposite side of range
            range_top = metadata.get('range_top', 0)
            range_bottom = metadata.get('range_bottom', 0)
            range_midpoint = metadata.get('range_midpoint', 0)

            # Custom calculation for range strategy
            if range_top > 0 and range_bottom > 0 and range_midpoint > 0:
                if order_type == 0:  # BUY at support, targets are midpoint and resistance
                    tp1 = range_midpoint
                    tp2 = range_top * 0.997  # Just below resistance

                    # Calculate equivalent reward ratios for logging
                    r1 = (tp1 - entry_price) / risk if risk > 0 else 1.0
                    r2 = (tp2 - entry_price) / risk if risk > 0 else 2.0

                    # Return absolute values rather than ratios
                    return [tp1, tp2]

                else:  # SELL at resistance, targets are midpoint and support
                    tp1 = range_midpoint
                    tp2 = range_bottom * 1.003  # Just above support

                    # Calculate equivalent reward ratios for logging
                    r1 = (entry_price - tp1) / risk if risk > 0 else 1.0
                    r2 = (entry_price - tp2) / risk if risk > 0 else 2.0

                    # Return absolute values rather than ratios
                    return [tp1, tp2]

        # For other strategies or fallback, calculate targets based on ratios
        if order_type == 0:  # BUY
            tp1 = entry_price + (risk * r1)
            tp2 = entry_price + (risk * r2)
        else:  # SELL
            tp1 = entry_price - (risk * r1)
            tp2 = entry_price - (risk * r2)

        self.logger.debug(
            f"Calculated {strategy_name} take-profits: {[tp1, tp2]} "
            f"(based on risk={risk} with R:R={[r1, r2]})"
        )

        return [tp1, tp2]

    def _execute_with_partial_profit(self, signal, metadata, order_type, stop_loss, position_size, take_profit_levels):
        """Execute an entry with the two-part profit strategy specified in the plan.

        Args:
            signal (StrategySignal): The entry signal
            metadata (dict): Signal metadata
            order_type (int): Order type (0=BUY, 1=SELL)
            stop_loss (float): Stop loss price
            position_size (float): Total position size
            take_profit_levels (list): List of [tp1, tp2] prices

        Returns:
            bool: True if executed successfully, False otherwise
        """
        try:
            # Split into two equal parts
            position_size_1 = position_size * 0.5
            position_size_2 = position_size * 0.5

            # Round to allowed lot sizes
            symbol_info = self.connector.get_symbol_info(signal.symbol)
            lot_step = symbol_info['lot_step']
            min_lot = symbol_info['min_lot']

            position_size_1 = round(position_size_1 / lot_step) * lot_step
            position_size_2 = position_size - position_size_1

            # Ensure minimum lot sizes
            if position_size_1 < min_lot or position_size_2 < min_lot:
                # Can't split, use single position approach
                self.logger.warning(
                    f"Position size too small to split: {position_size}. Using single position."
                )

                # Use combined position with first target
                comment = f"Signal_{signal.id}_{signal.strategy_name}"

                order_result = self.connector.place_order(
                    order_type=order_type,
                    symbol=signal.symbol,
                    volume=position_size,
                    price=0.0,  # Market price
                    stop_loss=stop_loss,
                    take_profit=take_profit_levels[0],  # First target
                    comment=comment
                )

                # Record the trade
                trade = Trade(
                    strategy_name=signal.strategy_name,
                    signal_id=signal.id,
                    symbol=signal.symbol,
                    order_type=signal.signal_type,
                    volume=position_size,
                    open_price=order_result['price'],
                    open_time=datetime.utcnow(),
                    stop_loss=stop_loss,
                    take_profit=take_profit_levels[0],
                    comment=comment
                )
                self.trade_repository.add(trade)

                self.logger.info(
                    f"Executed {signal.signal_type} order: {position_size} lots of {signal.symbol} "
                    f"at ${order_result['price']:.2f}, SL=${stop_loss:.2f}, TP=${take_profit_levels[0]:.2f}"
                )

                return True

            # Execute two-part strategy as per the plan
            # First part with first target
            comment_1 = f"Signal_{signal.id}_{signal.strategy_name}_Part1"

            order_result_1 = self.connector.place_order(
                order_type=order_type,
                symbol=signal.symbol,
                volume=position_size_1,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_levels[0],  # First target at 1:1 R:R
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
                take_profit=take_profit_levels[0],
                comment=comment_1
            )
            self.trade_repository.add(trade_1)

            # Second part with second target
            comment_2 = f"Signal_{signal.id}_{signal.strategy_name}_Part2"

            order_result_2 = self.connector.place_order(
                order_type=order_type,
                symbol=signal.symbol,
                volume=position_size_2,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_levels[1],  # Second target (extended)
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
                take_profit=take_profit_levels[1],
                comment=comment_2
            )
            self.trade_repository.add(trade_2)

            self.logger.info(
                f"Successfully executed {signal.signal_type} with partial profit: "
                f"position 1: {position_size_1} lots at ${order_result_1['price']:.2f}, "
                f"TP=${take_profit_levels[0]:.2f}, position 2: {position_size_2} lots at "
                f"${order_result_2['price']:.2f}, TP=${take_profit_levels[1]:.2f}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error executing partial profit strategy: {str(e)}")
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
                        f"Closed position {position['ticket']} at ${close_result['close_price']:.2f}, "
                        f"profit: ${close_result['profit']:.2f}"
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