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

## Quick Start

1. Make sure MetaTrader 5 is installed and configured with an account
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Edit configuration in `config/development.py` or `config/production.py`
   - Set MT5 terminal path, login, password, and server
   - Configure database settings
   - Adjust risk parameters if needed

4. Run the bot:
   ```bash
   # For normal mode
   python main.py

   # For verbose logging
   python main.py --verbose

   # For simulation mode (no real trades)
   python main.py --no-trade
   
   # For production environment
   python main.py --env production
   ```

## Usage Options

The bot has been improved to work with minimal configuration:

- **Automatic Initialization**: The database is automatically initialized the first time you run the bot
- **Automatic Data Sync**: Historical data is synced automatically when you start the bot
- **Simulation Mode**: Use `--no-trade` to run without placing real trades (signals are still generated)
- **Verbosity Control**: Use `--verbose` to see detailed logs, or omit for minimal output

## Project Structure

```
xauusd-trading-bot/
├── config/                 # Configuration files
├── custom_logging/         # Logging functionality
├── data/                   # Database models and repositories
├── execution/              # Order execution and management
├── mt5_connector/          # MetaTrader 5 connectivity
├── risk_management/        # Risk validation and position sizing
├── strategies/             # Trading strategy implementations
├── tests/                  # Unit tests
├── main.py                 # Main application entry point
└── requirements.txt        # Dependencies
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
