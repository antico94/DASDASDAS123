# container.py (updated)
from dependency_injector import containers, providers
from logging.logger import app_logger
from data.repository import (
    OHLCDataRepository,
    StrategySignalRepository,
    TradeRepository,
    AccountSnapshotRepository
)
from mt5_connector.connection import MT5Connector
from mt5_connector.data_fetcher import MT5DataFetcher
from risk_management.position_sizing import PositionSizer
from risk_management.risk_validator import RiskValidator
from execution.order_manager import OrderManager
from execution.trailing_stop import TrailingStopManager
from strategies.moving_average import MovingAverageStrategy


class Container(containers.DeclarativeContainer):
    """Dependency Injection container for the trading bot."""

    config = providers.Configuration()

    # Core components
    mt5_connector = providers.Singleton(MT5Connector)

    # Repositories
    ohlc_repository = providers.Singleton(OHLCDataRepository)
    signal_repository = providers.Singleton(StrategySignalRepository)
    trade_repository = providers.Singleton(TradeRepository)
    account_repository = providers.Singleton(AccountSnapshotRepository)

    # Data services
    data_fetcher = providers.Singleton(
        MT5DataFetcher,
        connector=mt5_connector,
        repository=ohlc_repository
    )

    # Risk management
    position_sizer = providers.Singleton(
        PositionSizer,
        connector=mt5_connector
    )

    risk_validator = providers.Singleton(
        RiskValidator,
        connector=mt5_connector,
        trade_repository=trade_repository,
        account_repository=account_repository
    )

    # Order execution
    order_manager = providers.Singleton(
        OrderManager,
        connector=mt5_connector,
        trade_repository=trade_repository,
        signal_repository=signal_repository,
        position_sizer=position_sizer,
        risk_validator=risk_validator
    )

    trailing_stop_manager = providers.Singleton(
        TrailingStopManager,
        connector=mt5_connector,
        trade_repository=trade_repository,
        data_fetcher=data_fetcher
    )

    # Strategies
    moving_average_strategy = providers.Factory(
        MovingAverageStrategy,
        symbol=config.symbol,
        timeframe=config.ma_timeframe,
        fast_period=config.ma_fast_period,
        slow_period=config.ma_slow_period,
        data_fetcher=data_fetcher
    )