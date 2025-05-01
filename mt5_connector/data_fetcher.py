# mt5_connector/data_fetcher.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
# Removed yfinance import
from datetime import datetime, timedelta, timezone # Import timezone
from db_logger.db_logger import DBLogger
from mt5_connector.connection import MT5Connector
from data.models import OHLCData
from data.db_session import DatabaseSession # Assuming DatabaseSession handles session creation
from data.repository import OHLCDataRepository # Assuming OHLCDataRepository is in data/repository.py
from sqlalchemy.exc import ResourceClosedError # Import specific exception
from sqlalchemy.orm import Session # Import Session type hint

class MT5DataFetcher:
    """Class to fetch and process data from MetaTrader 5, using tick volume as volume."""

    # Removed yfinance interval map
    # Removed yfinance symbol map

    # Mapping of timeframe strings to approximate seconds per candle
    # Used to estimate the number of candles to fetch with copy_rates_from_pos
    SECONDS_PER_CANDLE = {
        'M1': 60,
        'M5': 300,
        'M15': 900,
        'M30': 1800,
        'H1': 3600,
        'H4': 14400,
        'D1': 86400, # Approximation, doesn't account for weekends/holidays
        'W1': 604800 # Approximation
    }


    def __init__(self, connector=None, repository=None):
        """Initialize the data fetcher.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None (creates new one).
            repository (OHLCDataRepository, optional): OHLC data repository. Defaults to None (creates new one).
        """
        self.connector = connector or MT5Connector()
        # The repository instance should ideally not hold a session directly,
        # but rather accept one in its methods or use a scoped_session.
        # We'll modify repository methods to accept a session.
        self.repository = repository or OHLCDataRepository()

    def _timeframe_to_mt5(self, timeframe):
        """Convert a string timeframe to MT5 timeframe constant.

        Args:
            timeframe (str): Timeframe string (e.g., 'M5', 'H1')

        Returns:
            int: MT5 timeframe constant
        """
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1
        }

        mt5_timeframe = timeframe_map.get(timeframe.upper())
        if mt5_timeframe is None:
            error_msg = f"Invalid timeframe: {timeframe}"
            DBLogger.log_error("MT5DataFetcher", error_msg)
            raise ValueError(error_msg)

        return mt5_timeframe

    # Removed _get_yfinance_interval method
    # Removed _get_yfinance_symbol method


    def get_ohlc_data(self, symbol, timeframe, count=300, save_to_db=True, session: Session = None):
        """Get OHLC data from MT5, using tick volume as volume, for the latest 'count' candles.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            count (int, optional): Number of candles. Defaults to 300.
            save_to_db (bool, optional): Whether to save data to DB. Defaults to True.
            session (Session, optional): SQLAlchemy session to use for DB operations. Defaults to None.

        Returns:
            list: OHLC data
        """
        self.connector.ensure_connection()

        mt5_timeframe = self._timeframe_to_mt5(timeframe)
        # Removed yfinance interval and symbol variables

        # Get rates from MT5 using copy_rates_from_pos
        DBLogger.log_event("DEBUG", f"Fetching latest {count} {timeframe} candles for {symbol} from MT5 using copy_rates_from_pos", "MT5DataFetcher")
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            error_msg = f"Failed to get OHLC data for {symbol} {timeframe} from MT5: Error code {error_code}"
            DBLogger.log_error("MT5DataFetcher", error_msg)
            # If MT5 data fetch fails, we cannot proceed meaningfully
            raise RuntimeError(error_msg)

        DBLogger.log_event("DEBUG", f"Received {len(rates)} {timeframe} candles for {symbol} from MT5", "MT5DataFetcher")

        # Removed yfinance data fetching logic

        # Convert MT5 rates to list of OHLCData objects
        ohlc_data = []
        for rate in rates:
            # Convert time to UTC naive datetime for matching
            time_utc_naive = datetime.fromtimestamp(rate['time'], tz=timezone.utc).replace(tzinfo=None)

            # Use tick_volume directly for the volume field
            volume_from_mt5_tick = float(rate['tick_volume'])

            candle = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=time_utc_naive, # Store as UTC naive
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=volume_from_mt5_tick, # Use tick volume from MT5
                tick_volume=int(rate['tick_volume']), # Keep tick volume from MT5
                spread=int(rate['spread']) # Keep spread from MT5
            )

            ohlc_data.append(candle)

            # Save to database if requested, passing the session
            if save_to_db:
                self.repository.add_or_update(candle, session=session)

        return ohlc_data

    def get_ohlc_data_range(self, symbol, timeframe, from_date, to_date=None, save_to_db=True, session: Session = None):
        """Get OHLC data from MT5, using tick volume as volume, for a specific date range.
           Uses copy_rates_from_pos and filters to avoid MT5 server timezone issues with date ranges.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            from_date (datetime): Start date (assumed UTC naive)
            to_date (datetime, optional): End date (assumed UTC naive). Defaults to None (current time UTC naive).
            save_to_db (bool, optional): Whether to save data to DB. Defaults to True.
            session (Session, optional): SQLAlchemy session to use for DB operations. Defaults to None.

        Returns:
            list: OHLC data
        """
        self.connector.ensure_connection()

        mt5_timeframe = self._timeframe_to_mt5(timeframe)
        # Removed yfinance interval and symbol variables

        # Ensure from_date and to_date are UTC naive
        if from_date.tzinfo is not None:
             from_date = from_date.astimezone(timezone.utc).replace(tzinfo=None)
             DBLogger.log_event("WARNING", "from_date was timezone-aware, converted to UTC naive.", "MT5DataFetcher")

        if to_date is None:
            to_date = datetime.utcnow().replace(tzinfo=None) # Use current UTC naive time
        elif to_date.tzinfo is not None:
             to_date = to_date.astimezone(timezone.utc).replace(tzinfo=None)
             DBLogger.log_event("WARNING", "to_date was timezone-aware, converted to UTC naive.", "MT5DataFetcher")

        # --- Refactored MT5 data fetching using copy_rates_from_pos ---

        # Estimate the number of candles needed to cover the date range
        duration_seconds = (to_date - from_date).total_seconds()
        seconds_per_candle = self.SECONDS_PER_CANDLE.get(timeframe.upper(), 60) # Default to 60 seconds if not mapped
        # Ensure seconds_per_candle is not zero to avoid division by zero
        if seconds_per_candle <= 0:
             DBLogger.log_error("MT5DataFetcher", f"Invalid seconds per candle for timeframe {timeframe}: {seconds_per_candle}")
             # Fallback or raise an error
             seconds_per_candle = 60 # Fallback to 60 seconds

        estimated_count = int(duration_seconds / seconds_per_candle) if seconds_per_candle > 0 else 0


        # Add a buffer to the estimated count to ensure we get enough data
        # A 20% buffer plus a minimum of 100 candles buffer
        buffer_count = max(int(estimated_count * 0.20), 100)
        fetch_count = estimated_count + buffer_count

        # Ensure fetch_count is at least the required number of candles for the duration
        # Plus a minimum to get some context
        fetch_count = max(fetch_count, 200) # Ensure we fetch at least 200 candles

        DBLogger.log_event("DEBUG", f"Estimating {estimated_count} candles for range. Fetching {fetch_count} candles from MT5 using copy_rates_from_pos.", "MT5DataFetcher")

        # Get rates from MT5 using copy_rates_from_pos (fetching recent data)
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, fetch_count)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            if error_code == 0:  # No error, just no data
                DBLogger.log_event("WARNING", f"No recent data found for {symbol} {timeframe} from MT5 using copy_rates_from_pos.", "MT5DataFetcher")
                return []
            error_msg = f"Failed to get OHLC data from MT5 using copy_rates_from_pos: Error code {error_code}"
            DBLogger.log_error("MT5DataFetcher", error_msg)
            raise RuntimeError(error_msg)

        DBLogger.log_event("DEBUG", f"Received {len(rates)} {timeframe} candles from MT5 using copy_rates_from_pos.", "MT5DataFetcher")

        # Filter MT5 rates by the desired date range (UTC naive)
        filtered_rates = []
        for rate in rates:
            rate_time_utc_naive = datetime.fromtimestamp(rate['time'], tz=timezone.utc).replace(tzinfo=None)
            if from_date <= rate_time_utc_naive <= to_date:
                filtered_rates.append(rate)

        if not filtered_rates:
             DBLogger.log_event("WARNING", f"No MT5 data found within the requested range {from_date} to {to_date} after filtering.", "MT5DataFetcher")
             return []

        DBLogger.log_event("DEBUG", f"Filtered MT5 data to {len(filtered_rates)} candles within the range.", "MT5DataFetcher")

        # --- End of Refactored MT5 data fetching ---


        # Removed yfinance volume data fetching logic


        # Convert filtered MT5 rates to list of OHLCData objects
        ohlc_data = []
        for rate in filtered_rates:
            # Convert time to UTC naive datetime for matching
            time_utc_naive = datetime.fromtimestamp(rate['time'], tz=timezone.utc).replace(tzinfo=None)

            # Use tick_volume directly for the volume field
            volume_from_mt5_tick = float(rate['tick_volume'])

            candle = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=time_utc_naive, # Store as UTC naive
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=volume_from_mt5_tick, # Use tick volume from MT5
                tick_volume=int(rate['tick_volume']), # Keep tick volume from MT5
                spread=int(rate['spread']) # Keep spread from MT5
            )

            ohlc_data.append(candle)

            # Save to database if requested, passing the session
            if save_to_db:
                self.repository.add_or_update(candle, session=session)

        return ohlc_data

    def sync_missing_data(self, symbol, timeframe, days_back=7):
        """Sync missing OHLC data (from MT5, using tick volume as volume) for the specified period.
           Manages its own database session for the sync process.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            days_back (int, optional): Number of days to look back. Defaults to 7.

        Returns:
            int: Number of candles synced
        """
        self.connector.ensure_connection()

        # Use UTC naive datetimes for consistency
        from_date = (datetime.utcnow() - timedelta(days=days_back)).replace(tzinfo=None)
        to_date = datetime.utcnow().replace(tzinfo=None)

        DBLogger.log_event("DEBUG", f"Syncing missing {timeframe} data for {symbol} from {from_date} to {to_date}", "MT5DataFetcher")

        # --- Database Session Management for Sync ---
        session = DatabaseSession.get_session()
        synced_count = 0
        try:
            # First, get existing data from the database within the range using the session
            existing_data = self.repository.get_candles_range(
                symbol=symbol,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date,
                session=session # Pass the session
            )

            # Create a set of existing timestamps for quick lookup
            existing_timestamps = {candle.timestamp for candle in existing_data}

            # Get data from MT5 for the range (now uses tick volume for volume)
            # save_to_db is False here because we manually add below within the session
            mt5_data = self.get_ohlc_data_range(
                symbol=symbol,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date,
                save_to_db=False,
                session=session # Pass the session
            )

            # Determine which candles need to be saved (based on timestamp not being in existing_timestamps)
            for candle in mt5_data:
                if candle.timestamp not in existing_timestamps:
                    try:
                        self.repository.add(candle, session=session) # Pass the session
                        synced_count += 1
                    except ResourceClosedError as e:
                         DBLogger.log_error("MT5DataFetcher", f"ResourceClosedError during add in sync_missing_data: {e}")
                         # Attempt to rollback and break or handle as appropriate
                         session.rollback()
                         DBLogger.log_event("ERROR", "Session rolled back due to ResourceClosedError during sync. Attempting to close session.", "MT5DataFetcher")
                         # Depending on desired behavior, you might want to break or retry
                         # Ensure the session is closed if a ResourceClosedError occurs here
                         try:
                             session.close()
                             DBLogger.log_event("DEBUG", "Database session closed after ResourceClosedError.", "MT5DataFetcher")
                         except Exception as close_e:
                             DBLogger.log_error("MT5DataFetcher", f"Error closing session after ResourceClosedError: {close_e}")
                         break # Exit loop on error
                    except Exception as e:
                         DBLogger.log_error("MT5DataFetcher", f"Error adding candle during sync_missing_data: {e}")
                         # Rollback and break on other errors during add
                         session.rollback()
                         DBLogger.log_event("ERROR", "Session rolled back due to error during candle add.", "MT5DataFetcher")
                         # Ensure the session is closed on error
                         try:
                             session.close()
                             DBLogger.log_event("DEBUG", "Database session closed after error during candle add.", "MT5DataFetcher")
                         except Exception as close_e:
                             DBLogger.log_error("MT5DataFetcher", f"Error closing session after error during candle add: {close_e}")
                         break # Exit loop on error


            # Commit the session if any candles were synced and no errors occurred during add loop
            # Only commit if the session is still active (not closed by an error in the loop)
            if synced_count > 0 and session.is_active:
                try:
                    session.commit()
                    DBLogger.log_event("DEBUG", f"Synced {synced_count} missing candles for {symbol} {timeframe}", "MT5DataFetcher")
                except Exception as commit_e:
                    DBLogger.log_error("MT5DataFetcher", f"Error during session commit in sync_missing_data: {commit_e}")
                    session.rollback() # Attempt rollback on commit error
                    DBLogger.log_event("ERROR", "Session rolled back due to commit error.", "MT5DataFetcher")
                    raise # Re-raise the commit error

            elif not session.is_active:
                 DBLogger.log_event("WARNING", "Session was closed due to an error during sync, commit skipped.", "MT5DataFetcher")
            else:
                 # If no candles were synced and no errors occurred, rollback to be safe
                 session.rollback()
                 DBLogger.log_event("DEBUG", "No candles synced, session rolled back.", "MT5DataFetcher")


        except Exception as e:
            # Rollback the session in case of any other error during the main try block
            if session.is_active:
                 session.rollback()
                 DBLogger.log_event("ERROR", "Session rolled back due to error during sync_missing_data.", "MT5DataFetcher")
            DBLogger.log_error("MT5DataFetcher", f"Error during sync_missing_data for {symbol} {timeframe}: {e}")
            # Re-raise the exception if necessary
            raise
        finally:
            # Always close the session if it's still active
            if session and session.is_active: # Added check for session existence
                 try:
                     session.close()
                     DBLogger.log_event("DEBUG", f"Database session closed after sync for {symbol} {timeframe}.", "MT5DataFetcher")
                 except Exception as close_e:
                     DBLogger.log_error("MT5DataFetcher", f"Error closing session in finally block: {close_e}")


        # --- End of Database Session Management ---

        return synced_count

    # In mt5_connector/data_fetcher.py
    def get_latest_data_to_dataframe(self, symbol, timeframe, count=300):
        """Get the latest OHLC data (with tick volume as volume) as a pandas DataFrame."""
        # Fetch data using a new session (this method is for retrieval only, manages its own session)
        session = DatabaseSession.get_session()
        try:
            # Fetch data and immediately detach from session
            ohlc_data = session.query(OHLCData).filter(
                OHLCData.symbol == symbol,
                OHLCData.timeframe == timeframe
            ).order_by(OHLCData.timestamp.desc()).limit(count).all()

            # Detach all objects from the session
            for obj in ohlc_data:
                session.expunge(obj)

            # Convert to Pandas DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': candle.timestamp,
                    'open': float(candle.open),
                    'high': float(candle.high),
                    'low': float(candle.low),
                    'close': float(candle.close),
                    'volume': float(candle.volume), # This now comes from tick_volume via the DB
                    'tick_volume': int(candle.tick_volume),
                    'spread': int(candle.spread)
                }
                for candle in ohlc_data
            ])

            # Set timestamp as index
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                # IMPORTANT: We need to sort the index properly after reversing the order from SQL
                df.sort_index(inplace=True)  # Sort chronologically

            # Log the actual data retrieved
            DBLogger.log_event("DEBUG",
                               f"Retrieved {len(df)} candles for {symbol} {timeframe}",
                               "DataFetcher")

            return df
        finally:
            session.close()

    # Add to mt5_connector/data_fetcher.py
    def verify_data_sufficiency(self, symbol, timeframe, required_candles):
        """Verify that sufficient historical data exists for the given symbol and timeframe.

        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe (e.g., 'M5', 'H1', 'H4')
            required_candles (int): Minimum number of candles required

        Returns:
            tuple: (bool, int) - (is_sufficient, actual_candle_count)
        """
        try:
            # Get the data (this method manages its own session)
            data = self.get_latest_data_to_dataframe(symbol, timeframe, required_candles)

            # Check if we have enough candles
            candle_count = len(data)
            is_sufficient = candle_count >= required_candles

            # Log the result
            if is_sufficient:
                DBLogger.log_event("DEBUG",
                                   f"Data sufficiency check passed for {symbol} {timeframe}: "
                                   f"have {candle_count} candles, need {required_candles}",
                                   "DataFetcher")
            else:
                DBLogger.log_event("WARNING",
                                   f"Insufficient data for {symbol} {timeframe}: "
                                   f"have {candle_count} candles, need {required_candles}",
                                   "DataFetcher")

            return is_sufficient, candle_count

        except Exception as e:
            DBLogger.log_error("DataFetcher",
                               f"Error verifying data sufficiency for {symbol} {timeframe}",
                               exception=e)
            return False, 0
