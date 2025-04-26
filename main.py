# main.py (updated)
import os
import sys
import time
import signal
import argparse
from datetime import datetime, timedelta
from logging.logger import app_logger
from config import Config
from container import Container
from data.db_session import DatabaseSession
from data.models import AccountSnapshot
from mt5_connector.connection import MT5Connector
from mt5_connector.data_fetcher import MT5DataFetcher

# Initialize container
container = Container()
container.config.from_dict({
    "app_name": Config.APP_NAME,
    "version": Config.VERSION,
    "symbol": Config.SYMBOL,
    "ma_timeframe": Config.MA_TIMEFRAME,
    "ma_fast_period": Config.MA_FAST_PERIOD,
    "ma_slow_period": Config.MA_SLOW_PERIOD
})

# Global flag for the main loop
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals."""
    global running
    app_logger.info("Shutdown signal received, closing gracefully...")
    running = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=f"{Config.APP_NAME} v{Config.VERSION}")
    parser.add_argument('--env', default='development', help='Environment (development, production, testing)')
    parser.add_argument('--init-db', action='store_true', help='Initialize database schema')
    parser.add_argument('--sync-data', action='store_true', help='Sync historical data')
    return parser.parse_args()


def initialize_database():
    """Initialize database schema."""
    app_logger.info("Initializing database schema...")
    DatabaseSession.initialize()
    DatabaseSession.create_tables()
    app_logger.info("Database schema initialized successfully")


def sync_historical_data():
    """Sync historical price data for configured symbol and timeframes."""
    app_logger.info("Syncing historical data...")

    # Initialize MT5 connector and data fetcher
    connector = container.mt5_connector()
    data_fetcher = container.data_fetcher()

    # Define timeframes to sync
    timeframes = ['M5', 'M15', 'M30', 'H1', 'H4']

    # Sync data for each timeframe
    for timeframe in timeframes:
        app_logger.info(f"Syncing {Config.SYMBOL} {timeframe} data...")
        try:
            synced_count = data_fetcher.sync_missing_data(
                symbol=Config.SYMBOL,
                timeframe=timeframe,
                days_back=30  # Sync last 30 days
            )
            app_logger.info(f"Synced {synced_count} candles for {Config.SYMBOL} {timeframe}")
        except Exception as e:
            app_logger.error(f"Error syncing {Config.SYMBOL} {timeframe} data: {str(e)}")

    connector.disconnect()
    app_logger.info("Historical data sync completed")


def take_account_snapshot():
    """Take a snapshot of the current account state."""
    try:
        connector = container.mt5_connector()
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
        account_repo = container.account_repository()
        account_repo.add(snapshot)

        app_logger.debug(f"Account snapshot taken: balance={account_info['balance']}, equity={account_info['equity']}")

    except Exception as e:
        app_logger.error(f"Error taking account snapshot: {str(e)}")


def run_strategies():
    """Run all enabled trading strategies."""
    try:
        signal_repo = container.signal_repository()

        # Get strategies from config
        enabled_strategies = Config.STRATEGIES_ENABLED

        generated_signals = []

        # Run Moving Average strategy if enabled
        if "moving_average" in enabled_strategies:
            ma_strategy = container.moving_average_strategy()
            signals = ma_strategy.generate_signals()

            # Save signals to database
            for signal in signals:
                signal_repo.add(signal)
                generated_signals.append(signal)

        # Run Breakout strategy if enabled
        if "breakout" in enabled_strategies:
            breakout_strategy = container.breakout_strategy()
            signals = breakout_strategy.generate_signals()

            # Save signals to database
            for signal in signals:
                signal_repo.add(signal)
                generated_signals.append(signal)

        # Run Range-Bound strategy if enabled
        if "range_bound" in enabled_strategies:
            range_bound_strategy = container.range_bound_strategy()
            signals = range_bound_strategy.generate_signals()

            # Save signals to database
            for signal in signals:
                signal_repo.add(signal)
                generated_signals.append(signal)

        # Run Momentum Scalping strategy if enabled
        if "momentum_scalping" in enabled_strategies:
            momentum_strategy = container.momentum_scalping_strategy()
            signals = momentum_strategy.generate_signals()

            # Save signals to database
            for signal in signals:
                signal_repo.add(signal)
                generated_signals.append(signal)

        # Run Ichimoku strategy if enabled
        if "ichimoku" in enabled_strategies:
            ichimoku_strategy = container.ichimoku_strategy()
            signals = ichimoku_strategy.generate_signals()

            # Save signals to database
            for signal in signals:
                signal_repo.add(signal)
                generated_signals.append(signal)

        if generated_signals:
            app_logger.info(f"Generated {len(generated_signals)} total signals")

    except Exception as e:
        app_logger.error(f"Error running strategies: {str(e)}")
        return 0

    return len(generated_signals)



def process_trades():
    """Process pending signals and manage open trades."""
    try:
        # Process pending signals
        order_manager = container.order_manager()
        order_manager.process_pending_signals()

        # Update trailing stops
        trailing_stop_manager = container.trailing_stop_manager()
        trailing_stop_manager.update_trailing_stops()

    except Exception as e:
        app_logger.error(f"Error processing trades: {str(e)}")


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
    app_logger.info(f"Starting {Config.APP_NAME} v{Config.VERSION} in {args.env} environment")

    # Initialize database if requested
    if args.init_db:
        initialize_database()

    # Sync historical data if requested
    if args.sync_data:
        sync_historical_data()

    # Connect to MT5
    try:
        connector = container.mt5_connector()
        account_info = connector.get_account_info()
        app_logger.info(f"Connected to MetaTrader 5 successfully. Account balance: {account_info['balance']}")
    except Exception as e:
        app_logger.error(f"Failed to connect to MetaTrader 5: {str(e)}")
        return 1

    # Main loop
    last_snapshot_time = datetime.min
    snapshot_interval = timedelta(minutes=15)  # Take snapshot every 15 minutes

    last_strategy_time = datetime.min
    strategy_interval = timedelta(minutes=5)  # Run strategies every 5 minutes

    last_trade_time = datetime.min
    trade_interval = timedelta(seconds=30)  # Process trades every 30 seconds

    app_logger.info("Entering main loop...")

    try:
        while running:
            now = datetime.utcnow()

            # Ensure MT5 connection is active
            connector.ensure_connection()

            # Take account snapshot at regular intervals
            if now - last_snapshot_time > snapshot_interval:
                take_account_snapshot()
                last_snapshot_time = now

            # Run strategies at regular intervals
            if now - last_strategy_time > strategy_interval:
                run_strategies()
                last_strategy_time = now

            # Process trades and update trailing stops at regular intervals
            if now - last_trade_time > trade_interval:
                process_trades()
                last_trade_time = now

            # Sleep to avoid high CPU usage
            time.sleep(1)

    except Exception as e:
        app_logger.error(f"Error in main loop: {str(e)}")
        return 1
    finally:
        # Clean up resources
        connector.disconnect()
        DatabaseSession.close_session()
        app_logger.info(f"{Config.APP_NAME} shutdown complete")

    return 0




if __name__ == "__main__":
    sys.exit(main())