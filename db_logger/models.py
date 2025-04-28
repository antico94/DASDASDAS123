# db_logger/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()


class OrderRequestLog(Base):
    """Log table for order requests."""
    __tablename__ = 'order_requests' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # BUY, SELL, MODIFY, CLOSE
    message = Column(Text, nullable=False)

    # Additional fields
    symbol = Column(String(20))
    volume = Column(Float)
    price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    ticket = Column(Integer)
    strategy = Column(String(50))


class OrderExecutionLog(Base):
    """Log table for order executions."""
    __tablename__ = 'order_executions' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # OPENED, MODIFIED, CLOSED
    message = Column(Text, nullable=False)

    # Additional fields
    symbol = Column(String(20))
    volume = Column(Float)
    price = Column(Float)
    ticket = Column(Integer)
    profit = Column(Float)
    strategy = Column(String(50))


class PositionLog(Base):
    """Log table for position updates."""
    __tablename__ = 'positions' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # OPEN, MODIFIED, CLOSED, PARTIAL_CLOSE
    message = Column(Text, nullable=False)

    # Additional fields
    symbol = Column(String(20))
    ticket = Column(Integer)
    volume = Column(Float)
    open_price = Column(Float)
    current_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    profit = Column(Float)
    strategy = Column(String(50))


class AccountSnapshotLog(Base):
    """Log table for account snapshots."""
    __tablename__ = 'account_snapshots' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # SNAPSHOT
    message = Column(Text, nullable=False)

    # Additional fields
    balance = Column(Float)
    equity = Column(Float)
    margin = Column(Float)
    free_margin = Column(Float)
    margin_level = Column(Float)
    open_positions = Column(Integer)


class ErrorLog(Base):
    """Log table for errors."""
    __tablename__ = 'errors' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # ERROR, WARNING
    message = Column(Text, nullable=False)

    # Additional fields
    component = Column(String(100))
    exception_type = Column(String(100))
    stacktrace = Column(Text)


class EventLog(Base):
    """Log table for general events."""
    __tablename__ = 'events' # Changed: Removed 'logs.' prefix
    __table_args__ = {'schema': 'logs'} # Added: Explicitly specify the schema

    id = Column(Integer, primary_key=True)
    time = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    type = Column(String(50), nullable=False)  # INFO, DEBUG, SIGNAL, STRATEGY
    message = Column(Text, nullable=False)

    # Additional fields
    component = Column(String(100))
    details = Column(Text)