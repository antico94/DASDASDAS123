# data/models.py
import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from data.db_session import Base


class OHLCData(Base):
    """Model for OHLC price data."""

    __tablename__ = 'ohlc_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    tick_volume = Column(Integer)
    spread = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<OHLCData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"


class StrategySignal(Base):
    """Model for strategy-generated trading signals."""

    __tablename__ = 'strategy_signals'

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # 'BUY', 'SELL', 'CLOSE', etc.
    price = Column(Float, nullable=False)
    strength = Column(Float)  # Signal strength/confidence (0-1)
    signal_metadata = Column(String(1000))  # JSON string with additional signal data (renamed from 'metadata')
    is_executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<StrategySignal(strategy='{self.strategy_name}', symbol='{self.symbol}', type='{self.signal_type}')>"


class Trade(Base):
    """Model for executed trades."""

    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), nullable=False, index=True)
    signal_id = Column(Integer, ForeignKey('strategy_signals.id'), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    order_type = Column(String(20), nullable=False)  # 'BUY', 'SELL'
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    open_time = Column(DateTime, nullable=False)
    close_price = Column(Float)
    close_time = Column(DateTime)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    profit = Column(Float)
    commission = Column(Float, default=0)
    swap = Column(Float, default=0)
    comment = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationship to the signal that generated this trade
    signal = relationship("StrategySignal", backref="trades")

    def __repr__(self):
        return f"<Trade(id={self.id}, strategy='{self.strategy_name}', symbol='{self.symbol}', type='{self.order_type}')>"


class AccountSnapshot(Base):
    """Model for tracking account balance and performance."""

    __tablename__ = 'account_snapshots'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin = Column(Float)
    free_margin = Column(Float)
    margin_level = Column(Float)
    open_positions = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<AccountSnapshot(timestamp='{self.timestamp}', balance={self.balance}, equity={self.equity})>"