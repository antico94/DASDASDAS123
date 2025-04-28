# execution/order_manager.py
import json
import time
from datetime import datetime, timedelta
from db_logger.db_logger import DBLogger
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository, StrategySignalRepository, AccountSnapshotRepository
from data.models import Trade
from risk_management.position_sizing import PositionSizer
from risk_management.risk_validator import RiskValidator
from config import Config


class EnhancedOrderManager:
    """Enhanced class to manage order execution with dynamic risk allocation, session awareness,
    partial profit-taking and advanced trailing stops."""

    def __init__(self, connector=None, trade_repository=None, account_repository=None,
                 signal_repository=None, position_sizer=None, risk_validator=None):
        """Initialize the enhanced order manager.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None (creates new).
            trade_repository (TradeRepository, optional): Trade repository. Defaults to None (creates new).
            account_repository (AccountSnapshotRepository, optional): Account repository. Defaults to None.
            signal_repository (StrategySignalRepository, optional): Signal repository. Defaults to None (creates new).
            position_sizer (PositionSizer, optional): Position sizer. Defaults to None (creates new).
            risk_validator (RiskValidator, optional): Risk validator. Defaults to None (creates new).
        """
        self.connector = connector or MT5Connector()
        self.trade_repository = trade_repository or TradeRepository()
        self.account_repository = account_repository or AccountSnapshotRepository()
        self.signal_repository = signal_repository or StrategySignalRepository()
        self.position_sizer = position_sizer or PositionSizer(connector=self.connector)
        self.risk_validator = risk_validator or RiskValidator(connector=self.connector)

        # Maximum position scaling (for adding to winning positions)
        self.max_scale_in_count = 1  # Maximum number of times to scale into a position

        # Base risk percentage (will be dynamically adjusted)
        self.base_risk_percent = Config.MAX_RISK_PER_TRADE_PERCENT

        # Maximum total risk exposure across all positions
        self.max_total_risk_percent = Config.MAX_DAILY_RISK_PERCENT  # Default to max daily risk

        # Track recent strategy performance for dynamic risk adjustment
        self.recent_trades = []  # Store last N trade results
        self.max_recent_trades = 10  # Number of recent trades to track

    def process_pending_signals(self):
        """Process all pending trade signals with enhanced trade management.

        Returns:
            int: Number of signals processed
        """
        # Get pending signals
        pending_signals = self.signal_repository.get_pending_signals()

        if not pending_signals:
            DBLogger.log_event("DEBUG", "No pending signals to process", "OrderManager")
            return 0

        DBLogger.log_event("INFO", f"Processing {len(pending_signals)} pending signals", "OrderManager")

        processed_count = 0
        for signal in pending_signals:
            try:
                # Skip signals marked as simulation only
                if signal.comment == "SIMULATION_MODE":
                    DBLogger.log_event("INFO", f"Skipping signal {signal.id} marked as simulation", "OrderManager")
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
                DBLogger.log_error("OrderManager", f"Error processing signal {signal.id}", exception=e)

        if processed_count > 0:
            DBLogger.log_event("INFO", f"Processed {processed_count} signals", "OrderManager")
        return processed_count

    def _execute_signal(self, signal):
        """Execute a trading signal with enhanced trade management.

        Args:
            signal (StrategySignal): The signal to execute

        Returns:
            bool: True if executed successfully, False otherwise
        """
        DBLogger.log_event("INFO", f"Executing signal {signal.id}: {signal.signal_type} for {signal.symbol}",
                           "OrderManager")

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
                DBLogger.log_event("WARNING", f"Unknown signal type: {signal.signal_type}", "OrderManager")
                return False

        except Exception as e:
            DBLogger.log_error("OrderManager", f"Error executing signal {signal.id}", exception=e)
            return False

    def _compute_dynamic_risk_percent(self, signal, metadata):
        """Calculate dynamic risk percentage based on market conditions,
        strategy confidence, session, and recent performance.

        Args:
            signal (StrategySignal): The signal to calculate risk for
            metadata (dict): Signal metadata with additional information

        Returns:
            float: Risk percentage to use for this trade
        """
        # Start with base risk
        risk_percent = self.base_risk_percent

        # 1. Adjust by current session (time-based)
        current_hour = datetime.utcnow().hour

        # London/NY overlap (higher liquidity, can be more aggressive)
        if 13 <= current_hour < 17:
            risk_percent *= 1.2  # Increase risk by 20% during high liquidity
            DBLogger.log_event("DEBUG", "London/NY overlap: Increasing risk by 20%", "OrderManager")
        # Asian session (lower liquidity, be more conservative)
        elif 0 <= current_hour < 6:
            risk_percent *= 0.8  # Reduce risk by 20% during low liquidity
            DBLogger.log_event("DEBUG", "Asian session: Reducing risk by 20%", "OrderManager")

        # 2. Adjust by signal strength (if available in metadata)
        signal_strength = metadata.get('strength', 0.5)

        # Scale risk by signal strength (stronger signals get more risk)
        strength_factor = 0.8 + (signal_strength * 0.4)  # Range from 0.8 to 1.2
        risk_percent *= strength_factor

        # 3. Adjust by recent performance (anti-martingale approach)
        if self.recent_trades:
            # Calculate win rate from recent trades
            wins = sum(1 for result in self.recent_trades if result)
            win_rate = wins / len(self.recent_trades)

            # If winning > 70%, slightly increase risk
            if win_rate > 0.7:
                risk_percent *= 1.1
                DBLogger.log_event("DEBUG", f"High win rate ({win_rate:.2f}): Increasing risk by 10%", "OrderManager")
            # If winning < 30%, slightly decrease risk
            elif win_rate < 0.3:
                risk_percent *= 0.9
                DBLogger.log_event("DEBUG", f"Low win rate ({win_rate:.2f}): Decreasing risk by 10%", "OrderManager")

        # 4. Check for strategy confluence (multiple strategies agreeing)
        # This would require tracking active positions from other strategies
        # and checking if they align with this signal - outside current scope

        # 5. Calculate current open risk
        current_risk = self._calculate_current_risk_exposure()
        available_risk = self.max_total_risk_percent - current_risk

        # If available risk is less than calculated risk, reduce to available
        if available_risk < risk_percent:
            old_risk = risk_percent
            risk_percent = max(0, available_risk)
            DBLogger.log_event("INFO",
                               f"Reducing risk from {old_risk:.2f}% to {risk_percent:.2f}% "
                               f"due to max exposure limit ({self.max_total_risk_percent:.2f}%)",
                               "OrderManager")

            # If we have almost no risk available, consider skipping trade
            if risk_percent < self.base_risk_percent * 0.25:  # Less than 25% of normal risk
                DBLogger.log_event("WARNING",
                                   f"Available risk too low ({risk_percent:.2f}%), consider skipping trade",
                                   "OrderManager")

        # Ensure risk percent is within reasonable bounds (0.1% to 2%)
        risk_percent = max(0.1, min(risk_percent, 2.0))

        return risk_percent

    def _calculate_current_risk_exposure(self):
        """Calculate the current risk exposure from all open positions.

        Returns:
            float: Current risk exposure as percentage of account
        """
        # Get account information
        account_info = self.connector.get_account_info()
        account_balance = account_info.get('balance', 0)

        if account_balance == 0:
            return 0  # Avoid division by zero

        # Get all open positions
        positions = self.connector.get_positions()

        # Calculate total risk amount across all positions
        total_risk = 0
        for position in positions:
            entry = position.get('open_price', 0)
            stop = position.get('stop_loss', 0)
            volume = position.get('volume', 0)

            # Skip positions without stops
            if stop == 0:
                continue

            # Calculate position risk in currency units
            pip_value = self._calculate_pip_value(position['symbol'], volume)
            risk_amount = abs(entry - stop) * pip_value

            # Add to total
            total_risk += risk_amount

        # Convert to percentage of account
        risk_percent = (total_risk / account_balance) * 100

        return risk_percent

    def _calculate_pip_value(self, symbol, volume):
        """Calculate the value of 1 pip for a given symbol and volume.

        Args:
            symbol (str): The trading symbol
            volume (float): The trade volume in lots

        Returns:
            float: The value of 1 pip in account currency
        """
        # For XAU/USD, 1 pip is typically $0.1 per lot
        # Full calculation would require symbol information from broker
        if symbol == "XAUUSD":
            return 0.1 * volume * 10  # 1 pip = $1 for 1.0 lot (100 oz)
        else:
            # Default fallback
            return 0.1 * volume * 10  # Generic estimate

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
            DBLogger.log_error("OrderManager", "Signal is None")
            return False

        if signal.signal_type not in ["BUY", "SELL"]:
            DBLogger.log_error("OrderManager", f"Invalid signal type: {signal.signal_type}")
            return False

        if signal.price is None or not isinstance(signal.price, (int, float)) or signal.price <= 0:
            DBLogger.log_error("OrderManager", f"Invalid signal price: {signal.price}")
            return False

        if signal.symbol is None or not isinstance(signal.symbol, str):
            DBLogger.log_error("OrderManager", f"Invalid symbol: {signal.symbol}")
            return False

        # Convert signal type to MT5 order type
        order_type = 0 if signal.signal_type == "BUY" else 1  # 0=BUY, 1=SELL

        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
            DBLogger.log_event("WARNING", "Metadata is None, using empty dictionary", "OrderManager")

        # Calculate stop loss based on strategy
        stop_loss = self._calculate_strategy_stop_loss(signal, metadata, order_type)

        # Log the stop loss calculation
        DBLogger.log_event("DEBUG", f"Calculated stop loss: {stop_loss} for {signal.signal_type} signal",
                           "OrderManager")

        # Validate stop loss
        if not self.risk_validator.validate_stop_loss(
                symbol=signal.symbol,
                order_type=order_type,
                entry_price=signal.price,
                stop_loss_price=stop_loss
        ):
            DBLogger.log_event("WARNING", f"Invalid stop loss for signal {signal.id}", "OrderManager")
            return False

        # Check if we can open a new position based on risk rules
        can_open, reason = self.risk_validator.can_open_new_position(signal.symbol)
        if not can_open:
            DBLogger.log_event("WARNING", f"Risk validation failed: {reason}", "OrderManager")
            return False

        # Calculate dynamic risk percentage based on market conditions
        dynamic_risk_percent = self._compute_dynamic_risk_percent(signal, metadata)

        # Calculate position size based on dynamic risk
        atr_value = metadata.get('atr', None)

        # Store the original max risk percent
        original_risk_percent = self.position_sizer.max_risk_percent

        # Temporarily set the position sizer's risk percentage to our dynamic value
        self.position_sizer.max_risk_percent = dynamic_risk_percent

        position_size = self.position_sizer.calculate_position_size(
            symbol=signal.symbol,
            entry_price=signal.price,
            stop_loss_price=stop_loss,
            atr_value=atr_value
        )

        # Restore the original risk percent
        self.position_sizer.max_risk_percent = original_risk_percent

        # Validate position size
        if position_size is None or not self.position_sizer.validate_position_size(signal.symbol, position_size):
            DBLogger.log_event("WARNING", f"Invalid position size calculated: {position_size}", "OrderManager")
            return False

        # Calculate take-profit levels based on risk-reward ratios and strategy
        take_profit_levels = self._calculate_strategy_take_profits(
            signal, metadata, stop_loss, order_type
        )

        # Use two-part trade execution for risk management
        return self._execute_with_partial_profit(
            signal, metadata, order_type, stop_loss, position_size, take_profit_levels, dynamic_risk_percent
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

        # Adjust stop loss based on session volatility
        stop_loss = self._adjust_stop_for_session(signal.price, stop_loss, order_type)

        # Final validation
        if stop_loss <= 0:
            DBLogger.log_event("WARNING", f"Invalid calculated stop loss: {stop_loss}, using fallback", "OrderManager")
            # Fallback: 1% from price
            stop_loss = signal.price * 0.99 if order_type == 0 else signal.price * 1.01

        # Log the strategy-specific stop calculation
        DBLogger.log_event("DEBUG",
                           f"Calculated {strategy_name} stop loss: {stop_loss} "
                           f"for {signal.signal_type} at {signal.price}",
                           "OrderManager")

        return stop_loss

    def _adjust_stop_for_session(self, entry_price, stop_loss, order_type):
        """Adjust stop loss distance based on current trading session.

        Args:
            entry_price (float): Entry price
            stop_loss (float): Originally calculated stop loss
            order_type (int): Order type (0=BUY, 1=SELL)

        Returns:
            float: Adjusted stop loss price
        """
        # Get current UTC hour
        current_hour = datetime.utcnow().hour

        # Calculate the initial stop distance
        initial_distance = abs(entry_price - stop_loss)
        adjusted_distance = initial_distance

        # Asian session (lower liquidity, wider stops)
        if 0 <= current_hour < 6:
            adjusted_distance = initial_distance * 1.2  # 20% wider during Asian session
            DBLogger.log_event("DEBUG",
                               f"Asian session: Widening stop by 20% from {initial_distance:.2f} to {adjusted_distance:.2f}",
                               "OrderManager")
        # European pre-London (moderate liquidity)
        elif 6 <= current_hour < 8:
            adjusted_distance = initial_distance * 1.1  # 10% wider
        # London/NY overlap (highest liquidity, can use tighter stops)
        elif 13 <= current_hour < 17:
            adjusted_distance = initial_distance * 0.95  # 5% tighter during peak liquidity
            DBLogger.log_event("DEBUG",
                               f"London/NY overlap: Tightening stop by 5% from {initial_distance:.2f} to {adjusted_distance:.2f}",
                               "OrderManager")

        # Apply the adjusted distance
        if order_type == 0:  # BUY
            adjusted_stop = entry_price - adjusted_distance
        else:  # SELL
            adjusted_stop = entry_price + adjusted_distance

        return adjusted_stop

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

        DBLogger.log_event("DEBUG",
                           f"Calculated {strategy_name} take-profits: {[tp1, tp2]} "
                           f"(based on risk={risk} with R:R={[r1, r2]})",
                           "OrderManager")

        return [tp1, tp2]

    def _execute_with_partial_profit(self, signal, metadata, order_type, stop_loss, position_size, take_profit_levels,
                                     risk_percent):
        """Execute an entry with the two-part profit strategy for effective risk management.

        Args:
            signal (StrategySignal): The entry signal
            metadata (dict): Signal metadata
            order_type (int): Order type (0=BUY, 1=SELL)
            stop_loss (float): Stop loss price
            position_size (float): Total position size
            take_profit_levels (list): List of [tp1, tp2] prices
            risk_percent (float): Risk percentage used for this trade

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
                DBLogger.log_event("WARNING",
                                   f"Position size too small to split: {position_size}. Using single position.",
                                   "OrderManager")

                # Use combined position with first target
                comment = f"Signal_{signal.id}_{signal.strategy_name}"

                # Log order request
                DBLogger.log_order_request(
                    order_type=signal.signal_type,
                    symbol=signal.symbol,
                    volume=position_size,
                    price=0.0,  # Market price
                    stop_loss=stop_loss,
                    take_profit=take_profit_levels[0],
                    strategy=signal.strategy_name,
                    message=f"Placing {signal.signal_type} order (combined): {position_size} lots of {signal.symbol}"
                )

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

                # Log execution
                DBLogger.log_order_execution(
                    execution_type="OPENED",
                    symbol=signal.symbol,
                    volume=position_size,
                    price=order_result['price'],
                    ticket=order_result['ticket'],
                    strategy=signal.strategy_name,
                    message=f"Executed {signal.signal_type} order: {position_size} lots of {signal.symbol} "
                            f"at ${order_result['price']:.2f}, SL=${stop_loss:.2f}, TP=${take_profit_levels[0]:.2f}"
                )

                # Update recent trades tracking (placeholder until we know the outcome)
                self._track_trade_for_risk_adjustment(signal.id, None)

                return True

            # Execute two-part strategy as per the plan
            # First part with first target
            comment_1 = f"Signal_{signal.id}_{signal.strategy_name}_Part1"

            # Log first order request
            DBLogger.log_order_request(
                order_type=signal.signal_type,
                symbol=signal.symbol,
                volume=position_size_1,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_levels[0],
                strategy=signal.strategy_name,
                message=f"Placing {signal.signal_type} order (Part 1): {position_size_1} lots of {signal.symbol}"
            )

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

            # Log first order execution
            DBLogger.log_order_execution(
                execution_type="OPENED",
                symbol=signal.symbol,
                volume=position_size_1,
                price=order_result_1['price'],
                ticket=order_result_1['ticket'],
                strategy=signal.strategy_name,
                message=f"Executed {signal.signal_type} order (Part 1): {position_size_1} lots at ${order_result_1['price']:.2f}"
            )

            # Second part with second target
            comment_2 = f"Signal_{signal.id}_{signal.strategy_name}_Part2"

            # Log second order request
            DBLogger.log_order_request(
                order_type=signal.signal_type,
                symbol=signal.symbol,
                volume=position_size_2,
                price=0.0,  # Market price
                stop_loss=stop_loss,
                take_profit=take_profit_levels[1],
                strategy=signal.strategy_name,
                message=f"Placing {signal.signal_type} order (Part 2): {position_size_2} lots of {signal.symbol}"
            )

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

            # Log second order execution
            DBLogger.log_order_execution(
                execution_type="OPENED",
                symbol=signal.symbol,
                volume=position_size_2,
                price=order_result_2['price'],
                ticket=order_result_2['ticket'],
                strategy=signal.strategy_name,
                message=f"Executed {signal.signal_type} order (Part 2): {position_size_2} lots at ${order_result_2['price']:.2f}"
            )

            DBLogger.log_event("INFO",
                               f"Successfully executed {signal.signal_type} with partial profit: "
                               f"position 1: {position_size_1} lots at ${order_result_1['price']:.2f}, "
                               f"TP=${take_profit_levels[0]:.2f}, position 2: {position_size_2} lots at "
                               f"${order_result_2['price']:.2f}, TP=${take_profit_levels[1]:.2f}",
                               "OrderManager")

            # Update recent trades tracking (placeholder until we know the outcome)
            self._track_trade_for_risk_adjustment(signal.id, None)

            return True

        except Exception as e:
            DBLogger.log_error("OrderManager", f"Error executing partial profit strategy", exception=e)
            return False

    def _track_trade_for_risk_adjustment(self, signal_id, profit=None):
        """Track trade outcome for dynamic risk adjustment.

        Args:
            signal_id (int): ID of the signal that generated the trade
            profit (float, optional): Trade profit (if known). None if trade is still open.
        """
        # If profit is None, we're just recording a new trade
        if profit is None:
            # We'll update the result when the trade closes
            return

        # If profit is provided, we're recording a closed trade
        is_win = profit > 0

        # Add to recent trades list (True for win, False for loss)
        self.recent_trades.append(is_win)

        # Keep only the most recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)

        # Log the updated performance
        wins = sum(1 for result in self.recent_trades if result)
        win_rate = wins / len(self.recent_trades) if self.recent_trades else 0

        DBLogger.log_event("INFO",
                           f"Updated recent performance: {wins}/{len(self.recent_trades)} wins ({win_rate:.1%} win rate)",
                           "OrderManager")

    def _execute_close_signal(self, signal, metadata):
        """Execute a close signal to exit an existing position.

        Args:
            signal (StrategySignal): The close signal
            metadata (dict): Signal metadata

        Returns:
            bool: True if executed successfully, False otherwise
        """
        DBLogger.log_event("INFO", f"Executing CLOSE signal for {signal.symbol}", "OrderManager")

        try:
            # Get the positions to close
            positions = self.connector.get_positions(signal.symbol)

            if not positions:
                DBLogger.log_event("WARNING", f"No open positions found for {signal.symbol} to close",
                                   "OrderManager")
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

                # Log close request
                DBLogger.log_order_request(
                    order_type="CLOSE",
                    symbol=signal.symbol,
                    volume=position['volume'],
                    ticket=position['ticket'],
                    message=f"Closing position {position['ticket']}: {position['volume']} lots of {signal.symbol}"
                )

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

                            # Update trade tracking for risk adjustment
                            self._track_trade_for_risk_adjustment(trade.signal_id, close_result['profit'])
                            break

                    # Log close execution
                    DBLogger.log_order_execution(
                        execution_type="CLOSED",
                        symbol=signal.symbol,
                        volume=close_result['close_volume'],
                        price=close_result['close_price'],
                        ticket=position['ticket'],
                        profit=close_result['profit'],
                        message=f"Closed position {position['ticket']} at ${close_result['close_price']:.2f}, "
                                f"profit: ${close_result['profit']:.2f}"
                    )
                    closed_count += 1

            if closed_count > 0:
                return True
            else:
                DBLogger.log_event("WARNING", f"No matching positions were closed for {signal.symbol}",
                                   "OrderManager")
                return False

        except Exception as e:
            DBLogger.log_error("OrderManager", f"Error executing close signal", exception=e)
            return False

    def check_scaling_opportunities(self):
        """Check open positions for scaling in opportunities.

        This method evaluates open positions to see if any qualify for scaling in
        (adding to profitable positions).

        Returns:
            int: Number of positions scaled into
        """
        # Get all open positions
        positions = self.connector.get_positions()

        if not positions:
            return 0

        scaled_count = 0

        # Extract unique trades (combining Part1/Part2 of the same signal)
        unique_trades = {}
        for position in positions:
            # Extract signal ID and strategy from comment
            signal_id = self._extract_signal_id_from_comment(position['comment'])
            if not signal_id:
                continue

            # Group by signal ID
            if signal_id not in unique_trades:
                unique_trades[signal_id] = []
            unique_trades[signal_id].append(position)

        # For each unique trade, check scaling opportunity
        for signal_id, positions in unique_trades.items():
            # Skip if no positions for this signal
            if not positions:
                continue

            # Get first position to extract common data
            first_pos = positions[0]
            symbol = first_pos['symbol']
            direction = first_pos['type']  # 0=BUY, 1=SELL

            # Check if we already have scaled in positions
            has_scale_in = any("ScaleIn" in pos['comment'] for pos in positions)

            # If already scaled in to maximum, skip
            if has_scale_in and len([p for p in positions if "ScaleIn" in p['comment']]) >= self.max_scale_in_count:
                continue

            # Find the original entry price (average if multiple positions)
            entry_prices = [pos['open_price'] for pos in positions if "ScaleIn" not in pos['comment']]
            if not entry_prices:
                continue

            avg_entry = sum(entry_prices) / len(entry_prices)

            # Get current price
            current_price = first_pos['current_price']

            # Calculate profit in risk multiples (R)
            # Find the initial risk
            initial_stops = [pos['stop_loss'] for pos in positions if "ScaleIn" not in pos['comment']]
            if not initial_stops:
                continue

            initial_stop = initial_stops[0]  # Use first position's stop
            initial_risk = abs(avg_entry - initial_stop)

            if initial_risk <= 0:
                continue  # Skip if can't determine risk

            # Calculate current profit in R
            if direction == 0:  # BUY
                current_profit_r = (current_price - avg_entry) / initial_risk
            else:  # SELL
                current_profit_r = (avg_entry - current_price) / initial_risk

            # If profit is at least 1R, consider scaling in
            if current_profit_r >= 1.0:
                # Get strategy name
                strategy_name = self._extract_strategy_name_from_comment(first_pos['comment'])
                if not strategy_name:
                    continue

                # Determine scale-in size (half of original position)
                original_volume = sum(pos['volume'] for pos in positions if "ScaleIn" not in pos['comment'])
                scale_in_volume = original_volume * 0.5

                # Ensure minimum lot size
                symbol_info = self.connector.get_symbol_info(symbol)
                min_lot = symbol_info['lot_step']
                if scale_in_volume < min_lot:
                    continue  # Too small to scale in

                # Round to allowed lot size
                lot_step = symbol_info['lot_step']
                scale_in_volume = round(scale_in_volume / lot_step) * lot_step

                # Create a new stop loss at breakeven or better
                if direction == 0:  # BUY
                    new_stop = max(avg_entry, initial_stop)  # At least breakeven
                else:  # SELL
                    new_stop = min(avg_entry, initial_stop)  # At least breakeven

                # Calculate take profit (same as the second target from originals)
                take_profits = [pos['take_profit'] for pos in positions if "Part2" in pos['comment']]
                if not take_profits:
                    # Fallback: 2x risk from current price
                    if direction == 0:  # BUY
                        take_profit = current_price + initial_risk
                    else:  # SELL
                        take_profit = current_price - initial_risk
                else:
                    take_profit = take_profits[0]  # Use the part2 take profit

                # Create the comment for scaling
                comment = f"Signal_{signal_id}_{strategy_name}_ScaleIn"

                # Log order request
                DBLogger.log_order_request(
                    order_type="BUY" if direction == 0 else "SELL",
                    symbol=symbol,
                    volume=scale_in_volume,
                    price=0.0,  # Market price
                    stop_loss=new_stop,
                    take_profit=take_profit,
                    strategy=strategy_name,
                    message=f"Scaling in to profitable position: {scale_in_volume} lots of {symbol} at market"
                )

                try:
                    # Place the scale-in order
                    order_result = self.connector.place_order(
                        order_type=direction,
                        symbol=symbol,
                        volume=scale_in_volume,
                        price=0.0,  # Market price
                        stop_loss=new_stop,
                        take_profit=take_profit,
                        comment=comment
                    )

                    # Log execution
                    DBLogger.log_order_execution(
                        execution_type="OPENED",
                        symbol=symbol,
                        volume=scale_in_volume,
                        price=order_result['price'],
                        ticket=order_result['ticket'],
                        strategy=strategy_name,
                        message=f"Scaled in to {symbol} position: {scale_in_volume} lots at ${order_result['price']:.2f}, "
                                f"stop at ${new_stop:.2f} (breakeven or better), target at ${take_profit:.2f}"
                    )

                    scaled_count += 1

                except Exception as e:
                    DBLogger.log_error("OrderManager", f"Error scaling in to {symbol} position", exception=e)
                    continue

        return scaled_count

    def _extract_signal_id_from_comment(self, comment):
        """Extract signal ID from a position comment.

        Args:
            comment (str): Position comment

        Returns:
            int: Signal ID or None if not found
        """
        if not comment:
            return None

        parts = comment.split('_')
        if len(parts) >= 2 and parts[0] == "Signal":
            try:
                return int(parts[1])
            except ValueError:
                return None
        return None

    def _extract_strategy_name_from_comment(self, comment):
        """Extract strategy name from a position comment.

        Args:
            comment (str): Position comment

        Returns:
            str: Strategy name or None if not found
        """
        if not comment:
            return None

        parts = comment.split('_')
        if len(parts) >= 3 and parts[0] == "Signal":
            # Handle both formats: Signal_ID_Strategy or Signal_ID_Strategy_PartX
            return parts[2]
        return None