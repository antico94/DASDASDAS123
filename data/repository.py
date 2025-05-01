# data/repository.py
from datetime import datetime, timedelta, timezone # Import timezone
from sqlalchemy import func, desc
from db_logger.db_logger import DBLogger
from data.db_session import DatabaseSession # Assuming DatabaseSession handles session creation
from data.models import OHLCData, StrategySignal, Trade, AccountSnapshot
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, ResourceClosedError # Import specific exceptions
from sqlalchemy.orm import Session # Import Session type hint


class BaseRepository:
    """Base repository class with common methods."""

    def __init__(self, model_class):
        """Initialize the repository for a specific model.

        Args:
            model_class: The SQLAlchemy model class
        """
        self.model_class = model_class

    # Helper methods for session management (can be added here or in a separate mixin)
    # Adding them here for simplicity within the repository context.
    def _get_session(self, session: Session = None) -> Session:
        """Helper to get a session, either provided or a new one."""
        return session if session is not None else DatabaseSession.get_session()

    def _close_session_if_local(self, session: Session, local_session: Session):
        """Helper to close the session only if it was created locally."""
        if session is None and local_session and local_session.is_active:
            try:
                local_session.close()
                # DBLogger.log_event("DEBUG", "Local database session closed.", "Repository") # Optional logging
            except Exception as e:
                DBLogger.log_error("Repository", f"Error closing local session: {e}")


    def add(self, item, session: Session = None):
        """Add a new item to the database. Uses provided session or gets a new one.

        Args:
            item: The model instance to add
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            The added item with ID populated
        """
        local_session = self._get_session(session)
        try:
            local_session.add(item)
            if session is None: # Only commit if session was created here
                local_session.commit()
            # Note: Accessing item attributes after commit might load from DB,
            # which could cause ResourceClosedError if local_session is closed.
            # If item repr/str accesses attributes, ensure session is open or expunge.
            # For now, assuming basic repr doesn't trigger this.
            DBLogger.log_event("DEBUG", f"Added {self.model_class.__name__} to database (ID may not be populated if session passed): {item}", "Repository")
            return item
        except IntegrityError:
            if session is None: # Only rollback if session was created here
                local_session.rollback()
            DBLogger.log_event("DEBUG", f"Duplicate {self.model_class.__name__} ignored: {item}", "Repository")
            # Depending on desired behavior, you might return None or the existing item
            return None # Return None for duplicate ignored
        except SQLAlchemyError as e:
            if session is None and local_session.is_active: # Only rollback if session was created here and is active
                local_session.rollback()
            DBLogger.log_error("Repository", f"Error adding {self.model_class.__name__}", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)


    def update(self, item, session: Session = None):
        """Update an existing item in the database. Uses provided session or gets a new one.

        Args:
            item: The model instance to update
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            The updated item
        """
        local_session = self._get_session(session)
        try:
            # Use merge to handle detached objects
            merged_item = local_session.merge(item)
            if session is None: # Only commit if session was created here
                local_session.commit()
            DBLogger.log_event("DEBUG", f"Updated {self.model_class.__name__} in database: {merged_item}", "Repository")
            return merged_item
        except SQLAlchemyError as e:
            if session is None and local_session.is_active: # Only rollback if session was created here and is active
                local_session.rollback()
            DBLogger.log_error("Repository", f"Error updating {self.model_class.__name__}", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)


    def delete(self, item_id, session: Session = None):
        """Delete an item from the database by ID. Uses provided session or gets a new one.

        Args:
            item_id: The ID of the item to delete
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            bool: True if deleted, False if not found
        """
        local_session = self._get_session(session)
        try:
            item = local_session.query(self.model_class).filter(self.model_class.id == item_id).first()
            if item:
                local_session.delete(item)
                if session is None: # Only commit if session was created here
                    local_session.commit()
                DBLogger.log_event("DEBUG", f"Deleted {self.model_class.__name__} from database: ID {item_id}",
                               "Repository")
                return True
            return False
        except SQLAlchemyError as e:
            if session is None and local_session.is_active: # Only rollback if session was created here and is active
                local_session.rollback()
            DBLogger.log_error("Repository", f"Error deleting {self.model_class.__name__}", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)

    def get_by_id(self, item_id, session: Session = None):
        """Get an item by ID. Uses provided session or gets a new one.

        Args:
            item_id: The ID of the item
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            The item or None if not found
        """
        local_session = self._get_session(session)
        try:
            item = local_session.query(self.model_class).filter(self.model_class.id == item_id).first()
            # If a local session was used, expunge the object before closing
            if session is None and item:
                 local_session.expunge(item)
            return item
        except SQLAlchemyError as e:
            DBLogger.log_error("Repository", f"Error getting {self.model_class.__name__} by ID", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)

    def get_all(self, limit=None, session: Session = None):
        """Get all items. Uses provided session or gets a new one.

        Args:
            limit (int, optional): Maximum number of items to return
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            list: All items
        """
        local_session = self._get_session(session)
        try:
            query = local_session.query(self.model_class)
            if limit:
                query = query.limit(limit)
            items = query.all()
            # If a local session was used, expunge objects before closing
            if session is None:
                 for item in items:
                     local_session.expunge(item)
            return items
        except SQLAlchemyError as e:
            DBLogger.log_error("Repository", f"Error getting all {self.model_class.__name__}s", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)


class OHLCDataRepository(BaseRepository):
    """Repository for OHLC price data."""

    def __init__(self):
        super().__init__(OHLCData)

    def add_or_update(self, ohlc_data, session: Session = None):
        """Add or update OHLC data in the database. Uses provided session or gets a new one.

        Args:
            ohlc_data (OHLCData): The OHLC data to add or update
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            OHLCData: The added or updated OHLC data
        """
        local_session = self._get_session(session)
        try:
            # Check if record already exists based on primary key (symbol, timeframe, timestamp)
            existing = local_session.query(OHLCData).filter(
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
                DBLogger.log_event("DEBUG", f"Updated existing {self.model_class.__name__}: {ohlc_data.symbol} {ohlc_data.timeframe} {ohlc_data.timestamp}", "Repository")
                return existing
            else:
                # Add new record
                local_session.add(ohlc_data)
                DBLogger.log_event("DEBUG", f"Added new {self.model_class.__name__}: {ohlc_data.symbol} {ohlc_data.timeframe} {ohlc_data.timestamp}", "Repository")
                return ohlc_data

            # Commit only if the session was created locally
            if session is None:
                local_session.commit()

        except SQLAlchemyError as e:
            if session is None and local_session.is_active: # Only rollback if session was created here and is active
                local_session.rollback()
            DBLogger.log_error("Repository", "Error adding or updating OHLC data", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)


    def get_latest_candles(self, symbol, timeframe, count=100, session: Session = None):
        """Get the latest candles for a symbol and timeframe. Uses provided session or gets a new one.

        Args:
            symbol (str): The trading symbol
            timeframe (str): The timeframe (e.g., 'M5', 'H1')
            count (int): The number of candles to retrieve
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            list: The latest candles ordered by timestamp
        """
        local_session = self._get_session(session)
        try:
            candles = local_session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe
            ).order_by(desc(OHLCData.timestamp)).limit(count).all()

            # If a local session was used, expunge objects before closing
            if session is None:
                 for candle in candles:
                     local_session.expunge(candle)

            # Convert to list and reverse to get chronological order
            return list(reversed(candles))
        except SQLAlchemyError as e:
            DBLogger.log_error("Repository", "Error getting latest candles", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)

    def get_candles_range(self, symbol, timeframe, from_date, to_date=None, session: Session = None):
        """Get candles for a specific date range. Uses provided session or gets a new one.

        Args:
            symbol (str): The trading symbol
            timeframe (str): The timeframe (e.g., 'M5', 'H1')
            from_date (datetime): The start date
            to_date (datetime, optional): The end date. Defaults to now.
            session (Session, optional): SQLAlchemy session to use. Defaults to None.

        Returns:
            list: Candles in the date range
        """
        if to_date is None:
            to_date = datetime.utcnow().replace(tzinfo=None) # Ensure UTC naive

        local_session = self._get_session(session)
        try:
            # Ensure dates are UTC naive for comparison with stored timestamps
            if from_date.tzinfo is not None:
                 from_date = from_date.astimezone(timezone.utc).replace(tzinfo=None)
            if to_date.tzinfo is not None:
                 to_date = to_date.astimezone(timezone.utc).replace(tzinfo=None)

            candles = local_session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe,
                OHLCData.timestamp >= from_date,
                OHLCData.timestamp <= to_date
            ).order_by(OHLCData.timestamp).all()

            # If a local session was used, expunge objects before closing
            if session is None:
                for candle in candles:
                    local_session.expunge(candle)

            DBLogger.log_event("DEBUG", f"Retrieved {len(candles)} candles for {symbol} {timeframe} from DB.", "Repository")
            return candles

        except SQLAlchemyError as e:
            DBLogger.log_error("Repository", "Error getting candles range", exception=e)
            raise
        finally:
            self._close_session_if_local(session, local_session)


class StrategySignalRepository(BaseRepository):
    """Repository for strategy signals."""

    def __init__(self):
        super().__init__(StrategySignal)

    # Note: Methods in this class and below do not need explicit session
    # parameters if they are only called from contexts that manage their own
    # sessions or if they are intended to always use a new session per call.
    # The BaseRepository methods they inherit already handle session management
    # if no session is passed.

    def get_recent_signals(self, strategy_name=None, symbol=None, limit=50):
        """Get recent signals with optional filtering."""
        # Inherits session handling from BaseRepository.get_all or similar query pattern
        return super().get_all(limit=limit, filter_by={'strategy_name': strategy_name, 'symbol': symbol}) # Example if BaseRepository had filter_by

    def get_pending_signals(self, symbol=None):
        """Get signals that haven't been executed yet."""
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(StrategySignal).filter(
                StrategySignal.is_executed == False
            ).order_by(StrategySignal.timestamp)

            if symbol:
                query = query.filter(StrategySignal.symbol == symbol)

            signals = query.all()
            # Expunge if local session
            for signal in signals:
                session.expunge(signal)
            return signals
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting pending signals", exception=e)
            raise
        finally:
            if session:
                session.close()


    def mark_as_executed(self, signal_id):
        """Mark a signal as executed."""
        session = None
        try:
            session = DatabaseSession.get_session()
            signal = session.query(StrategySignal).filter(
                StrategySignal.id == signal_id
            ).first()

            if signal:
                signal.is_executed = True
                session.commit()
                DBLogger.log_event("DEBUG", f"Marked signal {signal_id} as executed", "Repository")
                # Expunge the updated object
                session.expunge(signal)
                return signal
            return None
        except Exception as e:
            if session:
                session.rollback()
            DBLogger.log_error("Repository", "Error marking signal as executed", exception=e)
            raise
        finally:
            if session:
                session.close()


class TradeRepository(BaseRepository):
    """Repository for trades."""

    def __init__(self):
        super().__init__(Trade)

    def get_open_trades(self, strategy_name=None, symbol=None):
        """Get currently open trades."""
        session = None
        try:
            session = DatabaseSession.get_session()
            query = session.query(Trade).filter(Trade.close_time == None)

            if strategy_name:
                query = query.filter(Trade.strategy_name == strategy_name)

            if symbol:
                query = query.filter(Trade.symbol == symbol)

            trades = query.all()
            # Expunge if local session
            for trade in trades:
                session.expunge(trade)
            return trades
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting open trades", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_trades_by_date_range(self, from_date, to_date=None, strategy_name=None):
        """Get trades within a date range."""
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

            trades = query.order_by(Trade.open_time).all()
            # Expunge if local session
            for trade in trades:
                session.expunge(trade)
            return trades
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting trades by date range", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_trades_performance(self, strategy_name=None, from_date=None):
        """Get performance statistics for trades."""
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
            total_trades_count = result.total_trades or 0
            win_rate = (winning_trades / total_trades_count) * 100 if total_trades_count > 0 else 0

            # No need to expunge scalar results or simple dicts

            return {
                'total_trades': total_trades_count,
                'winning_trades': winning_trades,
                'losing_trades': total_trades_count - winning_trades,
                'win_rate': win_rate,
                'total_profit': result.total_profit or 0,
                'average_profit': result.average_profit or 0,
                'worst_trade': result.worst_trade or 0,
                'best_trade': result.best_trade or 0
            }
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting trade performance", exception=e)
            raise
        finally:
            if session:
                session.close()


class AccountSnapshotRepository(BaseRepository):
    """Repository for account snapshots."""

    def __init__(self):
        super().__init__(AccountSnapshot)

    def get_latest_snapshot(self):
        """Get the most recent account snapshot."""
        session = None
        try:
            session = DatabaseSession.get_session()
            snapshot = session.query(AccountSnapshot).order_by(
                desc(AccountSnapshot.timestamp)
            ).first()
            # Expunge if local session
            if snapshot:
                 session.expunge(snapshot)
            return snapshot
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting latest snapshot", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_snapshots_range(self, from_date, to_date=None):
        """Get account snapshots for a specific date range."""
        if to_date is None:
            to_date = datetime.utcnow()

        session = None
        try:
            session = DatabaseSession.get_session()
            snapshots = session.query(AccountSnapshot).filter(
                AccountSnapshot.timestamp >= from_date,
                AccountSnapshot.timestamp <= to_date
            ).order_by(AccountSnapshot.timestamp).all()

            # Expunge if local session
            for snapshot in snapshots:
                 session.expunge(snapshot)
            return snapshots
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting account snapshots range", exception=e)
            raise
        finally:
            if session:
                session.close()

    def get_daily_snapshots(self, days=30):
        """Get one snapshot per day for the last N days."""
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

            # Expunge if local session
            for snapshot in snapshots:
                 session.expunge(snapshot)

            return snapshots
        except Exception as e:
            DBLogger.log_error("Repository", "Error getting daily account snapshots", exception=e)
            raise
        finally:
            if session:
                session.close()
