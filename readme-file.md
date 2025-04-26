# XAU/USD Trading Bot

A robust, modular Python trading bot for automated XAU/USD (Gold) trading using MetaTrader 5. This bot implements five different technical analysis strategies, comprehensive risk management, and full database logging.

## Features

- **Multiple Trading Strategies**:
  - Moving Average Trend Following (H1)
  - Support/Resistance Breakout (M15)
  - Range-Bound Mean Reversion (M15) 
  - Momentum Scalping (M5)
  - Ichimoku Cloud (H1)

- **Robust Risk Management**:
  - Per-trade risk limits (% of account)
  - Daily risk limits
  - Maximum drawdown protection
  - Position sizing based on risk
  - Stop-loss validation
  - Maximum open positions limit

- **Advanced Order Management**:
  - Automated trade execution
  - Partial profit-taking
  - Breakeven stops
  - Trailing stops
  - Multiple take-profit targets

- **Technical Infrastructure**:
  - Dependency injection for modular design
  - SQLAlchemy ORM for database persistence
  - Extensive logging
  - Unit test coverage with pytest
  - Multiple environment configurations (dev, prod, test)

## Project Structure

```
xauusd-trading-bot/
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── development.py
│   ├── production.py
│   └── testing.py
├── data/
│   ├── __init__.py
│   ├── db_session.py
│   ├── models.py
│   └── repository.py
├── execution/
│   ├── __init__.py
│   ├── order_manager.py
│   └── trailing_stop.py
├── logging/
│   ├── __init__.py
│   └── logger.py
├── mt5_connector/
│   ├── __init__.py
│   ├── connection.py
│   └── data_fetcher.py
├── risk_management/
│   ├── __init__.py
│   ├── position_sizing.py
│   └── risk_validator.py
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── breakout.py
│   ├── ichimoku.py
│   ├── momentum_scalping.py
│   ├── moving_average.py
│   └── range_bound.py
├── tests/
│   ├── test_breakout_strategy.py
│   ├── test_ichimoku_strategy.py
│   ├── test_momentum_scalping_strategy.py
│   ├── test_moving_average_strategy.py
│   └── test_range_bound_strategy.py
├── container.py
├── main.py
└── requirements.txt
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- MetaTrader 5 terminal installed
- SQL Server or compatible database

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/xauusd-trading-bot.git
   cd xauusd-trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the database connection in `config/development.py` or `config/production.py`.

5. Configure MetaTrader 5 settings in the same configuration files.

## Configuration

Edit the appropriate configuration file based on your environment:

```python
# Example configuration settings

# MT5 settings
MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN = 12345678
MT5_PASSWORD = "your_password"
MT5_SERVER = "BrokerName-Live"

# Database settings
DB_SERVER = "localhost"
DB_NAME = "XAUUSDTradingBot"
DB_USERNAME = "your_username"
DB_PASSWORD = "your_password"

# Risk management
MAX_RISK_PER_TRADE_PERCENT = 1.0
MAX_DAILY_RISK_PERCENT = 5.0
MAX_DRAWDOWN_PERCENT = 15.0
MAX_POSITIONS = 5

# Strategy settings
STRATEGIES_ENABLED = ["moving_average", "breakout", "range_bound", "momentum_scalping", "ichimoku"]
```

## Usage

### Running the Bot

```bash
# For development environment
python main.py --env development

# For production environment
python main.py --env production

# Initialize database schema
python main.py --init-db

# Sync historical data
python main.py --sync-data
```

### Running Tests

```bash
python -m pytest tests/
```

## Trading Strategies Details

### 1. Moving Average Trend Following Strategy

- **Timeframe**: H1 (1-Hour)
- **Indicators**: 20 EMA and 50 EMA
- **Entry Logic**: 
  - Buy when 20 EMA crosses above 50 EMA
  - Sell when 20 EMA crosses below 50 EMA
- **Exit Logic**: 
  - Take partial profit at 1:1 risk-to-reward
  - Trail stop behind 20 EMA for remainder

### 2. Breakout Strategy

- **Timeframe**: M15 (15-Minute)
- **Entry Logic**: 
  - Buy when price breaks above a consolidated range with volume
  - Sell when price breaks below a consolidated range with volume
- **Exit Logic**:
  - Take partial profit at 1:1 risk-to-reward
  - Take second profit by projecting range height
  - Stop-loss placed just inside the broken level

### 3. Range-Bound Mean Reversion Strategy

- **Timeframe**: M15 (15-Minute)
- **Indicators**: RSI, ADX, Bollinger Bands
- **Entry Logic**:
  - Buy at support when RSI is oversold and ADX is low (non-trending)
  - Sell at resistance when RSI is overbought and ADX is low
- **Exit Logic**:
  - Take partial profit at range midpoint
  - Take full profit at opposite range boundary
  - Stop-loss just outside the range

### 4. Momentum Scalping Strategy

- **Timeframe**: M5 (5-Minute)
- **Indicators**: 20 EMA and MACD
- **Entry Logic**:
  - Buy when price crosses above 20 EMA with MACD histogram turning positive
  - Sell when price crosses below 20 EMA with MACD histogram turning negative
- **Exit Logic**:
  - Take partial profit at 1:1 risk-to-reward
  - Move stop to breakeven
  - Target 2:1 reward-to-risk for remainder

### 5. Ichimoku Cloud Strategy

- **Timeframe**: H1 (1-Hour)
- **Entry Logic**:
  - Buy when Tenkan-sen crosses above Kijun-sen, price is above the Cloud, and Chikou Span confirms
  - Sell when Tenkan-sen crosses below Kijun-sen, price is below the Cloud, and Chikou Span confirms
- **Exit Logic**:
  - Trail stop with Kijun-sen
  - Take partial profit at 1.5:1 reward-to-risk
  - Target 3:1 reward-to-risk for remainder

## Risk Management

The bot incorporates multiple layers of risk management:

1. **Position Sizing**: Calculates position size based on account balance, risk percentage, and stop-loss distance.
2. **Risk Validation**: Validates trades against risk limits before execution.
3. **Trade Management**: Implements systematic partial profit-taking and stop management.
4. **Daily Risk Limits**: Prevents exceeding maximum daily risk.
5. **Drawdown Protection**: Stops trading if account drawdown exceeds specified threshold.

## License

MIT License

## Acknowledgments

- MetaTrader 5 for providing the trading API
- SQLAlchemy for database ORM
- Dependency Injector for clean architecture
