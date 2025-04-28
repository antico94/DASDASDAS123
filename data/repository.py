# data/repository.py
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from db_logger.db_logger import DBLogger
from data.db_session import DatabaseSession
from data.models import OHLCData, StrategySignal, Trade, AccountSnapshot


class BaseRepository:
    """Base repository class with common methods."""

    def __init__(self, model_class):
        """Initialize the repository for a specific model.

        Args:
            model_class: The SQLAlchemy model class
        """
        self.model_class = model_class

    def add(self, item):
        """Add a new item to the database.

        Args:
            item: The model instance to add

        Returns:
            The added item with ID populated
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            session.add(item)
            session.commit()
            DBLogger.log_event("DEBUG", f"Added {self.model_class.__name__} to database: {item}", "repository")
            return item
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("repository", f"Error adding {self.model_class.__name__}", exception=e)
            raise
        finally:
            if session:
                session.close()

    def update(self, item):
        """Update an existing item in the database.

        Args:
            item: The model instance to update

        Returns:
            The updated item
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            session.merge(item)
            session.commit()
            DBLogger.log_event("DEBUG", f"Updated {self.model_class.__name__} in database: {item}", "repository")
            return item
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("repository", f"Error updating {self.model_class.__name__}", exception=e)
            raise
        finally:
            if session:
                session.close()

    def delete(self, item_id):
        """Delete an item from the database by ID.

        Args:
            item_id: The ID of the item to delete

        Returns:
            bool: True if deleted, False if not found
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            item = session.query(self.model_class).filter(self.model_class.id == item_id).first()
            if item:
                session.delete(item)
                session.commit()
                DBLogger.log_event("DEBUG", f"Deleted {self.model_class.__name__} from database: ID {item_id}",
                                   "repository")
                return True
            return False
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("repository", f"Error deleting {self.model_class.__name__}", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_by_id(self, item_id):
        """Get an item by ID.

        Args:
            item_id: The ID of the item

        Returns:
            The item or None if not found
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            item = session.query(self.model_class).filter(self.model_class.id == item_id).first()
            return item
        except Exception as e:
            DBLogger.log_error("repository", f"Error getting {self.model_class.__name__} by ID", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_all(self, limit=None):
        """Get all items.

        Args:
            limit (int, optional): Maximum number of items to return

        Returns:
            list: All items
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(self.model_class)
            if limit:
                query = query.limit(limit)
            return query.all()
        except Exception as e:
            DBLogger.log_error("repository", f"Error getting all {self.model_class.__name__}s", exception=e)
            raise
        finally:
            if session:
                session.close()


class OHLCDataRepository(BaseRepository):
    """Repository for OHLC price data."""

    def __init__(self):
        super().__init__(OHLCData)

    def add_or_update(self, ohlc_data):
        """Add or update OHLC data in the database.

        Args:
            ohlc_data (OHLCData): The OHLC data to add or update

        Returns:
            OHLCData: The added or updated OHLC data
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            # Check if record already exists
            existing = session.query(OHLCData).filter(
                OHLCData.symbol == ohlc_data.symbol,
                OHLCData.timeframe == ohlc_data.timeframe,
                OHLCData.timestamp == ohlc_data.timestamp
            ).first()

            if existing:
                # Update existing record
                existing.open = ohlc_data.open
                existing.high = ohlc_data.high
                existing.low = ohlc_data.low
                existing.close = ohlc_data.close
                existing.volume = ohlc_data.volume
                existing.tick_volume = ohlc_data.tick_volume
                existing.spread = ohlc_data.spread
                session.commit()
                return existing
            else:
                # Add new record
                session.add(ohlc_data)
                session.commit()
                return ohlc_data
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("repository", "Error adding or updating OHLC data", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_latest_candles(self, symbol, timeframe, count=100):
        """Get the latest candles for a symbol and timeframe.

        Args:
            symbol (str): The trading symbol
            timeframe (str): The timeframe (e.g., 'M5', 'H1')
            count (int): The number of candles to retrieve

        Returns:
            list: The latest candles ordered by timestamp
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            candles = session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe
            ).order_by(desc(OHLCData.timestamp)).limit(count).all()

            # Convert to list and reverse to get chronological order
            return list(reversed(candles))
        except Exception as e:
            DBLogger.log_error("repository", "Error getting latest candles", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_candles_range(self, symbol, timeframe, from_date, to_date=None):
        """Get candles for a specific date range.

        Args:
            symbol (str): The trading symbol
            timeframe (str): The timeframe (e.g., 'M5', 'H1')
            from_date (datetime): The start date
            to_date (datetime, optional): The end date. Defaults to now.

        Returns:
            list: Candles in the date range
        """
        if to_date is None:
            to_date = datetime.utcnow()

        session = None
        try:
            session = DatabaseSession.get_session()
            candles = session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe,
                OHLCData.timestamp >= from_date,
                OHLCData.timestamp <= to_date
            ).order_by(OHLCData.timestamp).all()

            return candles
        except Exception as e:
            DBLogger.log_error("repository", "Error getting candles range", exception=e)
            raise
        finally:
            if session:
                session.close()


class StrategySignalRepository(BaseRepository):
    """Repository for strategy signals."""

    def __init__(self):
        super().__init__(StrategySignal)

    def get_recent_signals(self, strategy_name=None, symbol=None, limit=50):
        """Get recent signals with optional filtering.

        Args:
            strategy_name (str, optional): Filter by strategy name
            symbol (str, optional): Filter by symbol
            limit (int, optional): Maximum number of signals to return

        Returns:
            list: Recent signals ordered by timestamp
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(StrategySignal).order_by(desc(StrategySignal.timestamp))

            if strategy_name:
                query = query.filter(StrategySignal.strategy_name == strategy_name)

            if symbol:
                query = query.filter(StrategySignal.symbol == symbol)

            return query.limit(limit).all()
        except Exception as e:
            DBLogger.log_error("repository", "Error getting recent signals", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_pending_signals(self, symbol=None):
        """Get signals that haven't been executed yet.

        Args:
            symbol (str, optional): Filter by symbol

        Returns:
            list: Pending signals
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(StrategySignal).filter(
                StrategySignal.is_executed == False
            ).order_by(StrategySignal.timestamp)

            if symbol:
                query = query.filter(StrategySignal.symbol == symbol)

            return query.all()
        except Exception as e:
            DBLogger.log_error("repository", "Error getting pending signals", exception=e)
            raise
        finally:
            if session:
                session.close()

    def mark_as_executed(self, signal_id):
        """Mark a signal as executed.

        Args:
            signal_id (int): The signal ID

        Returns:
            StrategySignal: The updated signal
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            signal = session.query(StrategySignal).filter(
                StrategySignal.id == signal_id
            ).first()

            if signal:
                signal.is_executed = True
                session.commit()
                DBLogger.log_event("DEBUG", f"Marked signal {signal_id} as executed", "repository")
                return signal
            return None
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("repository", "Error marking signal as executed", exception=e)
            raise
        finally:
            if session:
                session.close()


class TradeRepository(BaseRepository):
    """Repository for trades."""

    def __init__(self):
        super().__init__(Trade)

    def get_open_trades(self, strategy_name=None, symbol=None):
        """Get currently open trades.

        Args:
            strategy_name (str, optional): Filter by strategy name
            symbol (str, optional): Filter by symbol

        Returns:
            list: Open trades
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(Trade).filter(Trade.close_time == None)

            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)

            if symbol:
                query = query.filter(Trade.symbol == symbol)

            return query.all()
        except Exception as e:
            DBLogger.log_error("repository", "Error getting open trades", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_trades_by_date_range(self, from_date, to_date=None, strategy_name=None):
        """Get trades within a date range.

        Args:
            from_date (datetime): Start date
            to_date (datetime, optional): End date. Defaults to now.
            strategy_name (str, optional): Filter by strategy name

        Returns:
            list: Trades in the date range
        """
        if to_date is None:
            to_date = datetime.utcnow()

        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(Trade).filter(
                Trade.open_time >= from_date,
                Trade.open_time <= to_date
            )

            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)

            return query.order_by(Trade.open_time).all()
        except Exception as e:
            DBLogger.log_error("repository", "Error getting trades by date range", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_trades_performance(self, strategy_name=None, from_date=None):
        """Get performance statistics for trades.

        Args:
            strategy_name (str, optional): Filter by strategy name
            from_date (datetime, optional): Starting date. Defaults to 30 days ago.

        Returns:
            dict: Performance statistics
        """
        if from_date is None:
            from_date = datetime.utcnow() - timedelta(days=30)

        session = None
        try:
            session = DatabaseSession.get_session()
            # Base query for closed trades
            query = session.query(
                func.count(Trade.id).label('total_trades'),
                func.sum(Trade.profit).label('total_profit'),
                func.avg(Trade.profit).label('average_profit'),
                func.min(Trade.profit).label('worst_trade'),
                func.max(Trade.profit).label('best_trade')
            ).filter(
                Trade.close_time != None,
                Trade.open_time >= from_date
            )

            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)

            result = query.first()

            # Count winning trades
            winning_query = session.query(func.count(Trade.id)).filter(
                Trade.close_time != None,
                Trade.open_time >= from_date,
                Trade.profit > 0
            )

            if strategy_name:
                winning_query = winning_query.filter(Trade.strategy_name == strategy_name)

            winning_trades = winning_query.scalar() or 0

            # Calculate win rate
            win_rate = (winning_trades / result.total_trades) * 100 if result.total_trades else 0

            return {
                'total_trades': result.total_trades or 0,
                'winning_trades': winning_trades,
                'losing_trades': (result.total_trades or 0) - winning_trades,
                'win_rate': win_rate,
                'total_profit': result.total_profit or 0,
                'average_profit': result.average_profit or 0,
                'worst_trade': result.worst_trade or 0,
                'best_trade': result.best_trade or 0
            }
        except Exception as e:
            DBLogger.log_error("repository", "Error getting trade performance", exception=e)
            raise
        finally:
            if session:
                session.close()


class AccountSnapshotRepository(BaseRepository):
    """Repository for account snapshots."""

    def __init__(self):
        super().__init__(AccountSnapshot)

    def get_latest_snapshot(self):
        """Get the most recent account snapshot.

        Returns:
            AccountSnapshot: The latest snapshot or None if none exists
        """
        session = None
        try:
            session = DatabaseSession.get_session()
            return session.query(AccountSnapshot).order_by(
                desc(AccountSnapshot.timestamp)
            ).first()
        except Exception as e:
            DBLogger.log_error("repository", "Error getting latest snapshot", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_snapshots_range(self, from_date, to_date=None):
        """Get account snapshots for a specific date range.

        Args:
            from_date (datetime): The start date
            to_date (datetime, optional): The end date. Defaults to now.

        Returns:
            list: Account snapshots in the date range
        """
        if to_date is None:
            to_date = datetime.utcnow()

        session = None
        try:
            session = DatabaseSession.get_session()
            snapshots = session.query(AccountSnapshot).filter(
                AccountSnapshot.timestamp >= from_date,
                AccountSnapshot.timestamp <= to_date
            ).order_by(AccountSnapshot.timestamp).all()

            return snapshots
        except Exception as e:
            DBLogger.log_error("repository", "Error getting account snapshots range", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_daily_snapshots(self, days=30):
        """Get one snapshot per day for the last N days.

        Args:
            days (int, optional): Number of days to look back. Defaults to 30.

        Returns:
            list: Daily account snapshots
        """
        from_date = datetime.utcnow() - timedelta(days=days)

        session = None
        try:
            session = DatabaseSession.get_session()
            # This query gets one snapshot per day (the latest of each day)
            # The specific implementation may vary depending on SQL Server version
            # This is a simplified approach
            snapshots = []
            current_date = from_date.date()
            end_date = datetime.utcnow().date()

            while current_date <= end_date:
                day_start = datetime.combine(current_date, datetime.min.time())
                day_end = datetime.combine(current_date, datetime.max.time())

                snapshot = session.query(AccountSnapshot).filter(
                    AccountSnapshot.timestamp >= day_start,
                    AccountSnapshot.timestamp <= day_end
                ).order_by(desc(AccountSnapshot.timestamp)).first()

                if snapshot:
                    snapshots.append(snapshot)

                current_date += timedelta(days=1)

            return snapshots
        except Exception as e:
            DBLogger.log_error("repository", "Error getting daily account snapshots", exception=e)
            raise
        finally:
            if session:
                session.close()
