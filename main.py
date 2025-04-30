# main.py
import os
import sys
import time
import signal
import argparse
import logging
from datetime import datetime, timedelta
from config import Config
from container import Container
from data.db_session import DatabaseSession
from data.models import AccountSnapshot, Base
from db_logger.setup import initialize_logging
from db_logger.logging_setup import setup_logging
from db_logger.db_logger import DBLogger

# Global flag for the main loop
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals."""
    global running
    logging.info("Shutdown signal received, closing gracefully...")
    DBLogger.log_event("INFO", "Shutdown signal received, closing gracefully...", "main")
    running = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=f"{Config.APP_NAME} v{Config.VERSION}")
    parser.add_argument('--env', default='development', help='Environment (development, production, testing)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-trade', action='store_true', help='Run in simulation mode without placing real trades')
    return parser.parse_args()


def initialize_database():
    """Initialize database schema if it doesn't exist yet."""
    logging.info("Checking database...")
    DatabaseSession.initialize()

    # Check if tables exist by trying to query one of our tables
    session = DatabaseSession.get_session()
    try:
        # Try to get one record from the account_snapshots table
        result = session.execute("SELECT 1 FROM account_snapshots LIMIT 1")
        exists = result.fetchone() is not None
        session.close()

        if exists:
            logging.info("Database already initialized")
            return
    except Exception:
        # Table doesn't exist, we'll create all tables
        pass
    finally:
        session.close()

    logging.info("Initializing database schema...")
    Base.metadata.create_all(DatabaseSession._engine)
    logging.info("Database schema initialized successfully")

    # Initialize logging tables
    initialize_logging()


def sync_historical_data(data_fetcher, force=False):
    """Sync historical price data for configured symbol and timeframes if needed.

    Ensures sufficient history is available for all strategies, including:
    - Triple MA (requires 200+ candles for 200 SMA)
    - Ichimoku (requires extended history for cloud calculations)

    Args:
        data_fetcher: The data fetcher instance
        force (bool): Force sync even if not needed
    """
    # Define timeframes to sync
    timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']

    # Calculate minimum candles needed for different strategies
    # Triple MA requires slow_period (200) + buffer
    triple_ma_candles = Config.TRIPLE_MA_SLOW_PERIOD + 50  # 250 candles
    # Ichimoku requires senkou_b + displacement + buffer
    ichimoku_candles = Config.IC_SENKOU_B_PERIOD + 26 + 30  # 108 candles
    # Get the maximum required candles across all strategies
    min_required_candles = max(triple_ma_candles, ichimoku_candles, 300)  # Use 300 as safe minimum

    # Calculate days needed for each timeframe to ensure sufficient history
    # This mapping estimates how many days are needed for each timeframe to get min_required_candles
    days_needed = {
        'M5': min(90, max(10, int(min_required_candles * 5 / (60 * 24) * 1.5))),  # 5-min candles with 50% buffer
        'M15': min(90, max(10, int(min_required_candles * 15 / (60 * 24) * 1.5))),  # 15-min candles with 50% buffer
        'M30': min(90, max(15, int(min_required_candles * 30 / (60 * 24) * 1.5))),  # 30-min candles with 50% buffer
        'H1': min(120, max(20, int(min_required_candles * 60 / (60 * 24) * 1.5))),  # 1-hour candles with 50% buffer
        'H4': min(240, max(40, int(min_required_candles * 4 * 60 / (60 * 24) * 1.5)))  # 4-hour candles with 50% buffer
    }

    # Mapping of which timeframes are used by which strategies (for verification)
    strategy_timeframes = {
        'Triple MA': {
            'timeframe': Config.TRIPLE_MA_TIMEFRAME,
            'required_candles': triple_ma_candles
        },
        'Ichimoku': {
            'timeframe': Config.IC_TIMEFRAME,
            'required_candles': ichimoku_candles
        }
    }

    # Check if we need to sync
    last_sync_time = get_last_sync_time()
    now = datetime.utcnow()

    # First, check if critical strategies have enough data
    needs_sync = force
    if not needs_sync:
        for strategy_name, info in strategy_timeframes.items():
            is_sufficient, count = data_fetcher.verify_data_sufficiency(
                Config.SYMBOL, info['timeframe'], info['required_candles']
            )
            if not is_sufficient:
                logging.warning(f"Forcing sync: Insufficient data for {strategy_name} strategy. "
                                f"Have {count} candles, need {info['required_candles']}.")
                needs_sync = True
                break

    # Also sync if it's been more than 8 hours
    if not needs_sync and (last_sync_time is None or (now - last_sync_time) > timedelta(hours=8)):
        needs_sync = True
        logging.info("Regular sync needed (>8 hours since last sync)")

    if needs_sync:
        logging.info(f"Syncing historical data for {Config.SYMBOL}...")
        DBLogger.log_event("INFO", f"Syncing historical data for {Config.SYMBOL}...", "data_sync")

        for timeframe in timeframes:
            try:
                days_back = days_needed.get(timeframe, 90)  # Default to 90 days if timeframe not in mapping

                logging.info(f"Syncing {timeframe} data for {days_back} days back...")

                # First validate we can connect to MT5
                if not data_fetcher.connector._is_connected:
                    data_fetcher.connector.ensure_connection()
                    if not data_fetcher.connector._is_connected:
                        error_msg = f"Cannot sync {timeframe} data - MT5 connection failed"
                        logging.error(error_msg)
                        DBLogger.log_error("data_sync", error_msg)
                        continue

                # Sync the data
                synced_count = data_fetcher.sync_missing_data(
                    symbol=Config.SYMBOL,
                    timeframe=timeframe,
                    days_back=days_back
                )

                if synced_count > 0:
                    success_msg = f"Synced {synced_count} candles for {Config.SYMBOL} {timeframe} ({days_back} days back)"
                    logging.info(success_msg)
                    DBLogger.log_event("INFO", success_msg, "data_sync")
                else:
                    logging.info(f"Data for {Config.SYMBOL} {timeframe} already up to date")

                # Verify sufficiency for critical timeframes
                for strategy_name, info in strategy_timeframes.items():
                    if info['timeframe'] == timeframe:
                        is_sufficient, count = data_fetcher.verify_data_sufficiency(
                            Config.SYMBOL, timeframe, info['required_candles']
                        )
                        if not is_sufficient:
                            warning_msg = (f"WARNING: Insufficient historical data for {strategy_name} after sync. "
                                           f"Have {count} candles, need {info['required_candles']}. "
                                           f"Strategy may not operate correctly.")
                            logging.warning(warning_msg)
                            DBLogger.log_event("WARNING", warning_msg, "data_sync")
                        else:
                            logging.info(f"Verified sufficient data for {strategy_name}: {count} candles available")

            except Exception as e:
                logging.error(f"Error syncing {timeframe} data: {str(e)}")
                DBLogger.log_error("data_sync", f"Error syncing {timeframe} data", exception=e)

        # Update last sync time
        update_last_sync_time(now)
    else:
        logging.info("Historical data is already up to date")

        # Even if sync is not needed, still log data sufficiency for critical strategies
        for strategy_name, info in strategy_timeframes.items():
            is_sufficient, count = data_fetcher.verify_data_sufficiency(
                Config.SYMBOL, info['timeframe'], info['required_candles']
            )
            logging.info(f"Data sufficiency for {strategy_name}: {count}/{info['required_candles']} candles "
                         f"({'SUFFICIENT' if is_sufficient else 'INSUFFICIENT'})")


def get_last_sync_time():
    """Get the timestamp of the last successful data sync.

    Returns:
        datetime or None: Last sync time or None if never synced
    """
    # We could create a separate table for this, but for simplicity
    # we'll use a special account snapshot with a comment
    session = DatabaseSession.get_session()
    try:
        sync_record = session.query(AccountSnapshot).filter(
            AccountSnapshot.comment == "DATA_SYNC"
        ).order_by(AccountSnapshot.timestamp.desc()).first()

        if sync_record:
            return sync_record.timestamp
        return None
    except Exception as e:
        logging.warning(f"Error retrieving last sync time: {str(e)}")
        DBLogger.log_error("data_sync", "Error retrieving last sync time", exception=e)
        return None
    finally:
        session.close()


def update_last_sync_time(timestamp):
    """Update the timestamp of the last successful data sync.

    Args:
        timestamp (datetime): The sync timestamp
    """
    session = DatabaseSession.get_session()
    try:
        # Create a special account snapshot to mark the sync time
        sync_record = AccountSnapshot(
            timestamp=timestamp,
            balance=0,
            equity=0,
            margin=0,
            free_margin=0,
            margin_level=0,
            open_positions=0,
            comment="DATA_SYNC"
        )
        session.add(sync_record)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.warning(f"Error updating last sync time: {str(e)}")
        DBLogger.log_error("data_sync", "Error updating last sync time", exception=e)
    finally:
        session.close()


def take_account_snapshot(connector, account_repo):
    """Take a snapshot of the current account state."""
    try:
        account_info = connector.get_account_info()

        # Get open positions count
        positions = connector.get_positions()
        open_positions = len(positions)

        # Create account snapshot
        snapshot = AccountSnapshot(
            timestamp=datetime.utcnow(),
            balance=account_info['balance'],
            equity=account_info['equity'],
            margin=account_info['margin'],
            free_margin=account_info['free_margin'],
            margin_level=account_info['margin_level'],
            open_positions=open_positions
        )

        # Save to database
        account_repo.add(snapshot)

        logging.debug(
            f"Account snapshot: balance=${account_info['balance']:.2f}, equity=${account_info['equity']:.2f}")

        # Log to the dedicated account snapshots table
        DBLogger.log_account_snapshot(
            balance=account_info['balance'],
            equity=account_info['equity'],
            margin=account_info['margin'],
            free_margin=account_info['free_margin'],
            margin_level=account_info['margin_level'],
            open_positions=open_positions
        )

    except Exception as e:
        logging.error(f"Error taking account snapshot: {str(e)}")
        DBLogger.log_error("account_snapshot", "Error taking account snapshot", exception=e)


def run_strategies(enabled_strategies, container, signal_repo, simulation_mode=False):
    """Run all enabled trading strategies within a database session.

    This ensures that StrategySignal objects are managed by an active session
    when their attributes are accessed. Data needed after the session is closed
    is extracted within the session scope.

    Args:
        enabled_strategies (list): List of enabled strategy names
        container: The dependency injection container
        signal_repo: The signal repository (consider if this still needs to manage sessions internally or can use the passed session)
        simulation_mode (bool): Whether to run in simulation mode

    Returns:
        int: Number of signals generated
    """
    generated_signals = []
    # New lists to store the extracted data (signal type and symbol)
    # These will hold simple Python strings, safe to access after the session closes.
    signal_types_for_log = []
    symbols_for_log = []

    logging.info("Running strategies...")
    DBLogger.log_event("INFO", "Running strategies", "strategies")

    try:
        # Use a session context manager for the core database operations.
        # All interactions requiring StrategySignal objects to be 'bound'
        # should happen within this block.
        with DatabaseSession.get_session() as session:  # Adjust if using session_scope
            # Ensure any repository methods called within this block use this 'session' instance.

            # Run Triple Moving Average strategy if enabled
            if "triple_ma" in enabled_strategies:
                logging.debug("Running Triple Moving Average strategy")
                triple_ma_strategy = container.triple_ma_strategy()
                signals = triple_ma_strategy.generate_signals()  # Returns StrategySignal objects

                # Process and save signals within the active session scope
                for signal in signals:
                    # Add the signal object to the current session.
                    session.add(signal)

                    if simulation_mode:
                        signal.comment = "SIMULATION_MODE"

                    generated_signals.append(signal)  # Keep the SQLAlchemy object reference

                    # Extract data needed for logging *while signal is bound to session*
                    signal_types_for_log.append(signal.signal_type)  # <-- Extract data here
                    symbols_for_log.append(signal.symbol)  # <-- Extract data here

                    # Log the signal to events table. Accessing attributes here is safe.
                    signal_data = {
                        "strategy": "triple_ma",
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "price": signal.price,
                        "strength": signal.strength
                    }
                    DBLogger.log_event("SIGNAL",
                                       f"Generated {signal.signal_type} signal for {signal.symbol} at {signal.price}",
                                       "triple_ma", signal_data)
                logging.debug(f"Processed {len(signals)} signals from Triple Moving Average strategy")

            # Run Breakout strategy if enabled
            if "breakout" in enabled_strategies:
                logging.debug("Running Breakout strategy")
                breakout_strategy = container.breakout_strategy()
                signals = breakout_strategy.generate_signals()
                for signal in signals:
                    session.add(signal)
                    if simulation_mode:
                        signal.comment = "SIMULATION_MODE"
                    generated_signals.append(signal)
                    # Extract data needed for logging *while signal is bound to session*
                    signal_types_for_log.append(signal.signal_type)  # <-- Extract data here
                    symbols_for_log.append(signal.symbol)  # <-- Extract data here
                    signal_data = {
                        "strategy": "breakout",
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "price": signal.price,
                        "strength": signal.strength
                    }
                    DBLogger.log_event("SIGNAL",
                                       f"Generated {signal.signal_type} signal for {signal.symbol} at {signal.price}",
                                       "breakout", signal_data)
                logging.debug(f"Processed {len(signals)} signals from Breakout strategy")

            # Run Range-Bound strategy if enabled
            if "range_bound" in enabled_strategies:
                logging.debug("Running Range-Bound strategy")
                range_bound_strategy = container.range_bound_strategy()
                signals = range_bound_strategy.generate_signals()
                for signal in signals:
                    session.add(signal)
                    if simulation_mode:
                        signal.comment = "SIMULATION_MODE"
                    generated_signals.append(signal)
                    # Extract data needed for logging *while signal is bound to session*
                    signal_types_for_log.append(signal.signal_type)  # <-- Extract data here
                    symbols_for_log.append(signal.symbol)  # <-- Extract data here
                    signal_data = {
                        "strategy": "range_bound",
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "price": signal.price,
                        "strength": signal.strength
                    }
                    DBLogger.log_event("SIGNAL",
                                       f"Generated {signal.signal_type} signal for {signal.symbol} at {signal.price}",
                                       "range_bound", signal_data)
                logging.debug(f"Processed {len(signals)} signals from Range-Bound strategy")

            # Run Momentum Scalping strategy if enabled
            if "momentum_scalping" in enabled_strategies:
                logging.debug("Running Momentum Scalping strategy")
                momentum_strategy = container.momentum_scalping_strategy()
                signals = momentum_strategy.generate_signals()
                for signal in signals:
                    session.add(signal)
                    if simulation_mode:
                        signal.comment = "SIMULATION_MODE"
                    generated_signals.append(signal)
                    # Extract data needed for logging *while signal is bound to session*
                    signal_types_for_log.append(signal.signal_type)  # <-- Extract data here
                    symbols_for_log.append(signal.symbol)  # <-- Extract data here
                    signal_data = {
                        "strategy": "momentum_scalping",
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "price": signal.price,
                        "strength": signal.strength
                    }
                    DBLogger.log_event("SIGNAL",
                                       f"Generated {signal.signal_type} signal for {signal.symbol} at {signal.price}",
                                       "momentum_scalping", signal_data)
                logging.debug(f"Processed {len(signals)} signals from Momentum Scalping strategy")

            # Run Ichimoku strategy if enabled
            if "ichimoku" in enabled_strategies:
                logging.debug("Running Ichimoku strategy")
                ichimoku_strategy = container.ichimoku_strategy()
                signals = ichimoku_strategy.generate_signals()
                for signal in signals:
                    session.add(signal)
                    if simulation_mode:
                        signal.comment = "SIMULATION_MODE"
                    generated_signals.append(signal)
                    # Extract data needed for logging *while signal is bound to session*
                    signal_types_for_log.append(signal.signal_type)  # <-- Extract data here
                    symbols_for_log.append(signal.symbol)  # <-- Extract data here
                    signal_data = {
                        "strategy": "ichimoku",
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type,
                        "price": signal.price,
                        "strength": signal.strength
                    }
                    DBLogger.log_event("SIGNAL",
                                       f"Generated {signal.signal_type} signal for {signal.symbol} at {signal.price}",
                                       "ichimoku", signal_data)
                logging.debug(f"Processed {len(signals)} signals from Ichimoku strategy")

            # The 'with' block will automatically commit the session here if no exceptions occurred,
            # and then close the session.
            # Objects in generated_signals become detached after this point.

        # --- Code Execution Continues Here AFTER the session is closed ---

        if generated_signals:
            # Use the lists containing extracted data for logging,
            # as the objects in generated_signals are now detached.
            logging.info(
                f"Generated {len(generated_signals)} signals: {signal_types_for_log} for symbols {symbols_for_log}")
            # Use the extracted lists for the DBLogger event as well
            DBLogger.log_event("INFO", f"Generated {len(generated_signals)} signals", "strategies",
                               {"signal_types": signal_types_for_log, "symbols": symbols_for_log})


    except Exception as e:
        # The session context manager should handle rollback on exception.
        logging.error(f"Error running strategies: {str(e)}", exc_info=True)  # Log traceback
        DBLogger.log_error("strategies", "Error running strategies", exception=e)
        return 0

    logging.info("Finished running strategies.")
    DBLogger.log_event("INFO", "Finished running strategies", "strategies")
    return len(generated_signals)


def process_trades(order_manager, trailing_stop_manager, simulation_mode=False):
    """Process pending signals and manage open trades."""
    try:
        if simulation_mode:
            logging.info("Simulation mode: skipping real trade execution")
            DBLogger.log_event("INFO", "Simulation mode: skipping real trade execution", "execution")
            return

        # Process pending signals
        order_manager.process_pending_signals()

        # Update trailing stops
        trailing_stop_manager.update_trailing_stops()

    except Exception as e:
        logging.error(f"Error processing trades: {str(e)}")
        DBLogger.log_error("execution", "Error processing trades", exception=e)


def configure_logging(verbose=False):
    """Configure logging level based on verbose flag."""
    # Configure standard Python logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = "logs/trading_bot.log"

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Setup logging with both Python logging and DB logging
    setup_logging(
        console_level=log_level,
        file_level=logging.DEBUG,
        db_level=log_level,
        log_file=log_file
    )

    logging.info(f"Logging configured: {'verbose' if verbose else 'normal'} mode")
    DBLogger.log_event("INFO", f"Logging configured: {'verbose' if verbose else 'normal'} mode", "system")


def main():
    """Main application entry point."""
    global running

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments
    args = parse_arguments()

    # Set environment
    os.environ['TRADING_BOT_ENV'] = args.env

    # Initialize database automatically (no need for --init-db flag)
    initialize_database()

    # Configure logging
    configure_logging(args.verbose)

    # Configure container
    container = Container()
    container.config.from_dict({
        # General settings
        "app_name": Config.APP_NAME,
        "version": Config.VERSION,
        "symbol": Config.SYMBOL,

        # Moving Average Strategy settings
        "ma_timeframe": Config.MA_TIMEFRAME,
        "ma_fast_period": Config.MA_FAST_PERIOD,
        "ma_slow_period": Config.MA_SLOW_PERIOD,

        # Triple Moving Average Strategy settings
        "triple_ma_timeframe": Config.TRIPLE_MA_TIMEFRAME,
        "triple_ma_fast_period": Config.TRIPLE_MA_FAST_PERIOD,
        "triple_ma_medium_period": Config.TRIPLE_MA_MEDIUM_PERIOD,
        "triple_ma_slow_period": Config.TRIPLE_MA_SLOW_PERIOD,

        # Breakout Strategy settings
        "bo_timeframe": Config.BO_TIMEFRAME,
        "bo_lookback_periods": Config.BO_LOOKBACK_PERIODS,
        "bo_min_range_bars": Config.BO_MIN_RANGE_BARS,
        "bo_volume_threshold": Config.BO_VOLUME_THRESHOLD,

        # Range-Bound Strategy settings
        "rb_timeframe": Config.RB_TIMEFRAME,
        "rb_lookback_periods": Config.RB_LOOKBACK_PERIODS,
        "rb_min_range_bars": Config.RB_MIN_RANGE_BARS,
        "rb_rsi_period": Config.RB_RSI_PERIOD,
        "rb_rsi_overbought": Config.RB_RSI_OVERBOUGHT,
        "rb_rsi_oversold": Config.RB_RSI_OVERSOLD,
        "rb_adx_period": Config.RB_ADX_PERIOD,
        "rb_adx_threshold": Config.RB_ADX_THRESHOLD,

        # Momentum Scalping Strategy settings (new parameters)
        "ms_timeframe": Config.MS_TIMEFRAME,
        "ms_rsi_period": Config.MS_RSI_PERIOD,
        "ms_rsi_threshold_high": Config.MS_RSI_THRESHOLD_HIGH,
        "ms_rsi_threshold_low": Config.MS_RSI_THRESHOLD_LOW,
        "ms_stoch_k_period": Config.MS_STOCH_K_PERIOD,
        "ms_stoch_d_period": Config.MS_STOCH_D_PERIOD,
        "ms_stoch_slowing": Config.MS_STOCH_SLOWING,
        "ms_macd_fast": Config.MS_MACD_FAST,
        "ms_macd_slow": Config.MS_MACD_SLOW,
        "ms_macd_signal": Config.MS_MACD_SIGNAL,
        "ms_momentum_period": Config.MS_MOMENTUM_PERIOD,
        "ms_volume_threshold": Config.MS_VOLUME_THRESHOLD,
        "ms_max_spread": Config.MS_MAX_SPREAD,
        "ms_consider_session": Config.MS_CONSIDER_SESSION,

        # Ichimoku Strategy settings
        "ic_timeframe": Config.IC_TIMEFRAME,
        "ic_tenkan_period": Config.IC_TENKAN_PERIOD,
        "ic_kijun_period": Config.IC_KIJUN_PERIOD,
        "ic_senkou_b_period": Config.IC_SENKOU_B_PERIOD
    })

    logging.info(f"Starting {Config.APP_NAME} v{Config.VERSION} in {args.env} environment")
    logging.info(f"Trading mode: {'SIMULATION' if args.no_trade else 'LIVE'}")

    # Log startup to events table
    DBLogger.log_event("INFO", f"Starting {Config.APP_NAME} v{Config.VERSION} in {args.env} environment", "system")
    DBLogger.log_event("INFO", f"Trading mode: {'SIMULATION' if args.no_trade else 'LIVE'}", "system")

    # Connect to MT5
    try:
        connector = container.mt5_connector()
        account_info = connector.get_account_info()
        logging.info(f"Connected to MetaTrader 5 successfully. Account balance: ${account_info['balance']:.2f}")

        # Log connection to events table
        DBLogger.log_event("INFO",
                           f"Connected to MetaTrader 5 successfully. Account balance: ${account_info['balance']:.2f}",
                           "connection")

    except Exception as e:
        logging.error(f"Failed to connect to MetaTrader 5: {str(e)}")
        DBLogger.log_error("connection", "Failed to connect to MetaTrader 5", exception=e)
        return 1

    # Initialize repositories
    signal_repo = container.signal_repository()
    account_repo = container.account_repository()

    # Initialize data fetcher
    data_fetcher = container.data_fetcher()

    # Sync historical data automatically (no need for --sync-data flag)
    sync_historical_data(data_fetcher)

    # Initialize trade processors
    order_manager = container.order_manager()
    trailing_stop_manager = container.trailing_stop_manager()

    # Main loop
    last_snapshot_time = datetime.min
    snapshot_interval = timedelta(minutes=15)  # Take snapshot every 15 minutes

    last_strategy_time = datetime.min
    strategy_interval = timedelta(minutes=5)  # Run strategies every 5 minutes

    last_trade_time = datetime.min
    trade_interval = timedelta(seconds=30)  # Process trades every 30 seconds

    last_sync_check_time = datetime.min
    sync_check_interval = timedelta(hours=4)  # Check if sync needed every 4 hours

    logging.info("Bot is now running. Press Ctrl+C to stop.")
    DBLogger.log_event("INFO", "Bot is now running", "system")

    try:
        while running:
            now = datetime.utcnow()

            # Ensure MT5 connection is active
            connector.ensure_connection()

            # Take account snapshot at regular intervals
            if now - last_snapshot_time > snapshot_interval:
                take_account_snapshot(connector, account_repo)
                last_snapshot_time = now

            # Run strategies at regular intervals
            if now - last_strategy_time > strategy_interval:
                run_strategies(
                    Config.STRATEGIES_ENABLED,
                    container,
                    signal_repo,
                    simulation_mode=args.no_trade
                )
                last_strategy_time = now

            # Process trades and update trailing stops at regular intervals
            if now - last_trade_time > trade_interval:
                process_trades(
                    order_manager,
                    trailing_stop_manager,
                    simulation_mode=args.no_trade
                )
                last_trade_time = now

            # Periodically check if we need to sync data
            if now - last_sync_check_time > sync_check_interval:
                sync_historical_data(data_fetcher)
                last_sync_check_time = now

            # Sleep to avoid high CPU usage
            time.sleep(1)

    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
        DBLogger.log_error("system", "Error in main loop", exception=e)
        return 1
    finally:
        # Clean up resources
        connector.disconnect()
        DatabaseSession.close_session()
        logging.info(f"{Config.APP_NAME} shutdown complete")
        DBLogger.log_event("INFO", f"{Config.APP_NAME} shutdown complete", "system")

    return 0


if __name__ == "__main__":
    sys.exit(main())