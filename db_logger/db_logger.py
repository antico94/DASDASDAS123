# db_logger/db_logger.py
import json
import traceback
from datetime import datetime
from sqlalchemy.orm import Session
from data.db_session import DatabaseSession
# Import from models module in the same package
from db_logger.models import (
    OrderRequestLog, OrderExecutionLog, PositionLog,
    AccountSnapshotLog, ErrorLog, EventLog
)


class DBLogger:
    """Database logger for the trading bot."""

    @classmethod
    def log_order_request(cls, order_type, symbol, volume, price=None,
                          stop_loss=None, take_profit=None, ticket=None,
                          strategy=None, message=None):
        """Log an order request."""
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            # Create default message if not provided
            if not message:
                message = f"{order_type} order request for {symbol}: {volume} lots"
                if price:
                    message += f" at price {price}"
                if stop_loss:
                    message += f", SL={stop_loss}"
                if take_profit:
                    message += f", TP={take_profit}"

            # Create log entry
            log_entry = OrderRequestLog(
                time=datetime.utcnow(),
                type=order_type,
                message=message,
                symbol=symbol,
                volume=volume,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                ticket=ticket,
                strategy=strategy
            )

            session.add(log_entry)
            session.commit()
        except Exception as e:
            cls.log_error("DBLogger", "Error logging order request", exception=e)
            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    @classmethod
    def log_order_execution(cls, execution_type, symbol, volume, price,
                            ticket=None, profit=None, strategy=None, message=None):
        """Log an order execution.

        Args:
            execution_type (str): Type of execution (OPENED, MODIFIED, CLOSED)
            symbol (str): Trading symbol
            volume (float): Order volume
            price (float): Execution price
            ticket (int, optional): Ticket number
            profit (float, optional): Profit/loss (for closed positions)
            strategy (str, optional): Strategy name
            message (str, optional): Custom message
        """
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            # Create default message if not provided
            if not message:
                message = f"{execution_type}: {symbol} {volume} lots at {price}"
                if ticket:
                    message += f" (ticket #{ticket})"
                if profit is not None:
                    message += f", profit: {profit}"

            # Create log entry
            log_entry = OrderExecutionLog(
                time=datetime.utcnow(),
                type=execution_type,
                message=message,
                symbol=symbol,
                volume=volume,
                price=price,
                ticket=ticket,
                profit=profit,
                strategy=strategy
            )

            session.add(log_entry)
            session.commit()
        except Exception as e:
            cls.log_error("DBLogger", "Error logging order execution", exception=e)
            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    @classmethod
    def log_position(cls, position_type, symbol, ticket, volume, open_price=None,
                     current_price=None, stop_loss=None, take_profit=None,
                     profit=None, strategy=None, message=None):
        """Log a position update.

        Args:
            position_type (str): Type of position update (OPEN, MODIFIED, CLOSED, PARTIAL_CLOSE)
            symbol (str): Trading symbol
            ticket (int): Position ticket number
            volume (float): Position volume
            open_price (float, optional): Opening price
            current_price (float, optional): Current price
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            profit (float, optional): Current profit/loss
            strategy (str, optional): Strategy name
            message (str, optional): Custom message
        """
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            # Create default message if not provided
            if not message:
                message = f"Position {position_type}: {symbol} #{ticket}, {volume} lots"
                if open_price:
                    message += f", opened at {open_price}"
                if current_price:
                    message += f", current price {current_price}"
                if profit is not None:
                    message += f", profit: {profit}"

            # Create log entry
            log_entry = PositionLog(
                time=datetime.utcnow(),
                type=position_type,
                message=message,
                symbol=symbol,
                ticket=ticket,
                volume=volume,
                open_price=open_price,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                profit=profit,
                strategy=strategy
            )

            session.add(log_entry)
            session.commit()
        except Exception as e:
            cls.log_error("DBLogger", "Error logging position", exception=e)
            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    @classmethod
    def log_account_snapshot(cls, balance, equity, margin=None, free_margin=None,
                             margin_level=None, open_positions=None, message=None):
        """Log an account snapshot.

        Args:
            balance (float): Account balance
            equity (float): Account equity
            margin (float, optional): Used margin
            free_margin (float, optional): Free margin
            margin_level (float, optional): Margin level percentage
            open_positions (int, optional): Number of open positions
            message (str, optional): Custom message
        """
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            # Create default message if not provided
            if not message:
                message = f"Account snapshot: balance={balance}, equity={equity}"
                if margin is not None:
                    message += f", margin={margin}"
                if free_margin is not None:
                    message += f", free margin={free_margin}"
                if margin_level is not None:
                    message += f", margin level={margin_level}%"
                if open_positions is not None:
                    message += f", open positions={open_positions}"

            # Create log entry
            log_entry = AccountSnapshotLog(
                time=datetime.utcnow(),
                type="SNAPSHOT",
                message=message,
                balance=balance,
                equity=equity,
                margin=margin,
                free_margin=free_margin,
                margin_level=margin_level,
                open_positions=open_positions
            )

            session.add(log_entry)
            session.commit()
        except Exception as e:
            cls.log_error("DBLogger", "Error logging account snapshot", exception=e)
            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    @classmethod
    def log_error(cls, component, message, exception=None, error_type="ERROR"):
        """Log an error.

        Args:
            component (str): Component that generated the error
            message (str): Error message
            exception (Exception, optional): Exception object
            error_type (str, optional): Type of error (ERROR, WARNING)
        """
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            exception_type = None
            stacktrace = None

            if exception:
                exception_type = type(exception).__name__
                stacktrace = traceback.format_exc()
                message = f"{message}: {str(exception)}"

            # Create log entry
            log_entry = ErrorLog(
                time=datetime.utcnow(),
                type=error_type,
                message=message,
                component=component,
                exception_type=exception_type,
                stacktrace=stacktrace
            )

            session.add(log_entry)
            session.commit()

            # Also print to console for immediate visibility
            print(f"[{error_type}] {component}: {message}")
            if stacktrace:
                print(stacktrace)

        except Exception as e:
            # If we can't log to the database, at least print to console
            print(f"[META-ERROR] Failed to log error: {str(e)}")
            print(f"Original error: {message}")
            if exception:
                print(traceback.format_exc())

            if session:
                session.rollback()
        finally:
            if session:
                session.close()

    @classmethod
    def log_event(cls, event_type, message, component=None, details=None):
        """Log a general event.

        Args:
            event_type (str): Type of event (INFO, DEBUG, SIGNAL, STRATEGY)
            message (str): Event message
            component (str, optional): Component that generated the event
            details (dict, optional): Additional details (will be JSON-encoded)
        """
        # Initialize session to None before try block
        session = None
        try:
            session = DatabaseSession.get_session()

            details_str = None
            if details:
                if isinstance(details, dict):
                    details_str = json.dumps(details)
                else:
                    details_str = str(details)

            # Create log entry
            log_entry = EventLog(
                time=datetime.utcnow(),
                type=event_type,
                message=message,
                component=component,
                details=details_str
            )

            session.add(log_entry)
            session.commit()
        except Exception as e:
            cls.log_error("DBLogger", "Error logging event", exception=e)
            if session:
                session.rollback()
        finally:
            if session:
                session.close()