# mt5_connector/data_fetcher.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from db_logger.db_logger import DBLogger
from mt5_connector.connection import MT5Connector
from data.models import OHLCData
from data.db_session import DatabaseSession
from data.repository import OHLCDataRepository


class MT5DataFetcher:
    """Class to fetch and process data from MetaTrader 5."""

    def __init__(self, connector=None, repository=None):
        """Initialize the data fetcher.

        Args:
            connector (MT5Connector, optional): MT5 connector. Defaults to None (creates new one).
            repository (OHLCDataRepository, optional): OHLC data repository. Defaults to None (creates new one).
        """
        self.connector = connector or MT5Connector()
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

    def get_ohlc_data(self, symbol, timeframe, count=200, save_to_db=True):
        """Get OHLC data from MT5.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            count (int, optional): Number of candles. Defaults to 100.
            save_to_db (bool, optional): Whether to save data to DB. Defaults to True.

        Returns:
            list: OHLC data
        """
        self.connector.ensure_connection()

        mt5_timeframe = self._timeframe_to_mt5(timeframe)

        # Get rates from MT5
        DBLogger.log_event("DEBUG", f"Fetching {count} {timeframe} candles for {symbol}", "MT5DataFetcher")
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            error_msg = f"Failed to get OHLC data for {symbol} {timeframe}: Error code {error_code}"
            DBLogger.log_error("MT5DataFetcher", error_msg)
            raise RuntimeError(error_msg)

        DBLogger.log_event("DEBUG", f"Received {len(rates)} {timeframe} candles for {symbol}", "MT5DataFetcher")

        # Convert to list of OHLCData objects
        ohlc_data = []
        for rate in rates:
            # Convert time to datetime
            time_utc = datetime.fromtimestamp(rate['time'])

            candle = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=time_utc,
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=float(rate['real_volume']),
                tick_volume=int(rate['tick_volume']),
                spread=int(rate['spread'])
            )

            ohlc_data.append(candle)

            # Save to database if requested
            if save_to_db:
                self.repository.add_or_update(candle)

        return ohlc_data

    def get_ohlc_data_range(self, symbol, timeframe, from_date, to_date=None, save_to_db=True):
        """Get OHLC data for a specific date range.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            from_date (datetime): Start date
            to_date (datetime, optional): End date. Defaults to None (current time).
            save_to_db (bool, optional): Whether to save data to DB. Defaults to True.

        Returns:
            list: OHLC data
        """
        self.connector.ensure_connection()

        mt5_timeframe = self._timeframe_to_mt5(timeframe)

        if to_date is None:
            to_date = datetime.utcnow()

        # Get rates from MT5
        DBLogger.log_event("DEBUG", f"Fetching {timeframe} candles for {symbol} from {from_date} to {to_date}", "MT5DataFetcher")

        # Convert dates to Unix timestamps
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())

        # MT5 expects the from and to parameters to be in local server timezone
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, from_timestamp, to_timestamp)

        if rates is None or len(rates) == 0:
            error_code = mt5.last_error()
            if error_code == 0:  # No error, just no data
                DBLogger.log_event("WARNING", f"No data found for {symbol} {timeframe} from {from_date} to {to_date}", "MT5DataFetcher")
                return []
            error_msg = f"Failed to get OHLC data: Error code {error_code}"
            DBLogger.log_error("MT5DataFetcher", error_msg)
            raise RuntimeError(error_msg)

        DBLogger.log_event("DEBUG", f"Received {len(rates)} {timeframe} candles for {symbol}", "MT5DataFetcher")

        # Convert to list of OHLCData objects
        ohlc_data = []
        for rate in rates:
            # Convert time to datetime
            time_utc = datetime.fromtimestamp(rate['time'])

            candle = OHLCData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=time_utc,
                open=float(rate['open']),
                high=float(rate['high']),
                low=float(rate['low']),
                close=float(rate['close']),
                volume=float(rate['real_volume']),
                tick_volume=int(rate['tick_volume']),
                spread=int(rate['spread'])
            )

            ohlc_data.append(candle)

            # Save to database if requested
            if save_to_db:
                self.repository.add_or_update(candle)

        return ohlc_data

    def sync_missing_data(self, symbol, timeframe, days_back=7):
        """Sync missing OHLC data for the specified period.

        Args:
            symbol (str): Symbol name
            timeframe (str): Timeframe (e.g., 'M5', 'H1')
            days_back (int, optional): Number of days to look back. Defaults to 7.

        Returns:
            int: Number of candles synced
        """
        self.connector.ensure_connection()

        from_date = datetime.utcnow() - timedelta(days=days_back)
        to_date = datetime.utcnow()

        DBLogger.log_event("DEBUG", f"Syncing missing {timeframe} data for {symbol}", "MT5DataFetcher")

        # First, get existing data from the database
        existing_data = self.repository.get_candles_range(
            symbol=symbol,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date
        )

        # Create a set of existing timestamps for quick lookup
        existing_timestamps = {candle.timestamp for candle in existing_data}

        # Get data from MT5
        mt5_data = self.get_ohlc_data_range(
            symbol=symbol,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date,
            save_to_db=False  # Don't save yet, we'll do it manually
        )

        # Determine which candles need to be saved
        synced_count = 0
        for candle in mt5_data:
            if candle.timestamp not in existing_timestamps:
                self.repository.add(candle)
                synced_count += 1

        if synced_count > 0:
            DBLogger.log_event("DEBUG", f"Synced {synced_count} missing candles for {symbol} {timeframe}", "MT5DataFetcher")

        return synced_count

    def get_latest_data_to_dataframe(self, symbol, timeframe, count=300):
        """Get the latest OHLC data as a pandas DataFrame."""
        # Fetch data using a new session
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
                    'volume': float(candle.volume),
                    'tick_volume': int(candle.tick_volume),
                    'spread': int(candle.spread)
                }
                for candle in ohlc_data
            ])

            # Set timestamp as index
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

            return df
        finally:
            session.close()