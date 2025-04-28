# test_risk_validator.py
import pytest
import datetime
from unittest.mock import Mock, MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from risk_management.risk_validator import RiskValidator
from mt5_connector.connection import MT5Connector
from data.repository import TradeRepository, AccountSnapshotRepository


class TestRiskValidator:
    """Unit tests for RiskValidator class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock MT5Connector."""
        connector = Mock(spec=MT5Connector)

        # Setup mock account info
        connector.get_account_info.return_value = {
            'balance': 10000.0,
            'equity': 10000.0,
            'margin': 0.0,
            'free_margin': 10000.0,
            'margin_level': 0.0
        }

        # Setup mock positions
        connector.get_positions.return_value = []

        # Setup mock symbol info for XAUUSD
        connector.get_symbol_info.return_value = {
            'name': 'XAUUSD',
            'bid': 2000.0,
            'ask': 2001.0,
            'point': 0.01,
            'digits': 2,
            'min_lot': 0.01,
            'max_lot': 10.0,
            'lot_step': 0.01,
            'trade_mode': 0
        }

        return connector

    @pytest.fixture
    def mock_trade_repo(self):
        """Create a mock TradeRepository."""
        repo = Mock(spec=TradeRepository)
        repo.get_trades_by_date_range.return_value = []
        return repo

    @pytest.fixture
    def mock_account_repo(self):
        """Create a mock AccountSnapshotRepository."""
        repo = Mock(spec=AccountSnapshotRepository)

        # Create mock snapshots
        snapshots = []
        current_time = datetime.datetime.utcnow()

        for i in range(30):
            time_point = current_time - datetime.timedelta(days=i)
            snapshot = MagicMock()
            snapshot.timestamp = time_point
            snapshot.balance = 10000.0
            snapshot.equity = 10000.0
            snapshot.margin = 0.0
            snapshot.free_margin = 10000.0
            snapshot.margin_level = 0.0
            snapshots.append(snapshot)

        repo.get_daily_snapshots.return_value = snapshots

        return repo

    @pytest.fixture
    def risk_validator(self, mock_connector, mock_trade_repo, mock_account_repo):
        """Create a RiskValidator instance with mocked dependencies."""
        validator = RiskValidator(
            connector=mock_connector,
            trade_repository=mock_trade_repo,
            account_repository=mock_account_repo
        )
        # Override config-based limits for testing
        validator.max_positions = 5
        validator.max_daily_risk_percent = 5.0
        validator.max_drawdown_percent = 15.0

        return validator

    def test_can_open_new_position_success(self, risk_validator, mock_connector):
        """Test can_open_new_position when all checks pass."""
        # Arrange - default fixture setup is good for this test

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is True
        assert "All risk checks passed" in reason

    def test_can_open_new_position_max_positions_reached(self, risk_validator, mock_connector):
        """Test can_open_new_position when max positions limit is reached."""
        # Arrange
        positions = []
        for i in range(5):  # Max positions is 5
            positions.append({
                'ticket': i,
                'symbol': 'XAUUSD',
                'type': 0,
                'volume': 0.1,
                'open_price': 2000.0,
                'open_time': datetime.datetime.utcnow(),
                'current_price': 2000.0,
                'stop_loss': 1990.0,
                'take_profit': 2010.0,
                'profit': 0.0,
                'comment': f"Test position {i}"
            })
        mock_connector.get_positions.return_value = positions

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is False
        assert "Maximum number of positions" in reason

    def test_can_open_new_position_daily_risk_exceeded(self, risk_validator, mock_trade_repo):
        """Test can_open_new_position when daily risk limit is exceeded."""
        # Arrange - Create some losing trades for today
        today = datetime.datetime.utcnow().date()
        today_start = datetime.datetime.combine(today, datetime.datetime.min.time())

        trades = []
        for i in range(3):
            trade = MagicMock()
            trade.open_time = today_start + datetime.timedelta(hours=i)
            trade.profit = -200.0  # Each trade lost $200
            trades.append(trade)

        mock_trade_repo.get_trades_by_date_range.return_value = trades

        # The total loss is $600, which is 6% of the $10,000 account balance
        # This exceeds our 5% daily risk limit

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is False
        assert "Daily loss" in reason

    def test_can_open_new_position_drawdown_exceeded(self, risk_validator, mock_connector, mock_account_repo):
        """Test can_open_new_position when max drawdown is exceeded."""
        # Arrange
        # Create account snapshots with a peak equity of $12,000
        snapshots = []
        current_time = datetime.datetime.utcnow()

        for i in range(30):
            time_point = current_time - datetime.timedelta(days=i)
            snapshot = MagicMock()
            snapshot.timestamp = time_point

            # 10 days ago we had peak equity
            if i == 10:
                snapshot.equity = 12000.0
            else:
                snapshot.equity = 10000.0

            snapshot.balance = 10000.0
            snapshots.append(snapshot)

        mock_account_repo.get_daily_snapshots.return_value = snapshots

        # Current equity is $10,000, which is a 16.7% drawdown from the peak of $12,000
        # This exceeds our 15% max drawdown limit

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is False
        assert "drawdown" in reason

    def test_can_open_new_position_connector_error(self, risk_validator, mock_connector):
        """Test can_open_new_position when connector raises an error."""
        # Arrange
        mock_connector.get_positions.side_effect = Exception("Connection error")

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is False
        assert "Error in risk validation" in reason

    def test_can_open_new_position_account_info_missing(self, risk_validator, mock_connector):
        """Test can_open_new_position when account info is missing."""
        # Arrange
        mock_connector.get_account_info.return_value = None

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is False
        assert "Failed to get account info" in reason

    def test_calculate_daily_loss(self, risk_validator, mock_trade_repo):
        """Test _calculate_daily_loss method."""
        # Arrange
        today = datetime.datetime.utcnow().date()
        today_start = datetime.datetime.combine(today, datetime.datetime.min.time())

        trades = []
        # Add some profitable trades
        for i in range(2):
            trade = MagicMock()
            trade.open_time = today_start + datetime.timedelta(hours=i)
            trade.profit = 100.0  # Each trade gained $100
            trades.append(trade)

        # Add some losing trades
        for i in range(2, 5):
            trade = MagicMock()
            trade.open_time = today_start + datetime.timedelta(hours=i)
            trade.profit = -150.0  # Each trade lost $150
            trades.append(trade)

        mock_trade_repo.get_trades_by_date_range.return_value = trades

        # Act
        daily_loss = risk_validator._calculate_daily_loss()

        # Assert
        # Total loss should be $450 (3 trades losing $150 each)
        assert daily_loss == 450.0

    def test_calculate_daily_loss_no_trades(self, risk_validator):
        """Test _calculate_daily_loss method when no trades exist."""
        # Act
        daily_loss = risk_validator._calculate_daily_loss()

        # Assert
        assert daily_loss == 0.0

    def test_calculate_daily_loss_error(self, risk_validator, mock_trade_repo):
        """Test _calculate_daily_loss method when repository raises an error."""
        # Arrange
        mock_trade_repo.get_trades_by_date_range.side_effect = Exception("Database error")

        # Act
        daily_loss = risk_validator._calculate_daily_loss()

        # Assert
        assert daily_loss == 0.0  # Should return a safe default value

    def test_calculate_drawdown(self, risk_validator, mock_connector):
        """Test _calculate_drawdown method."""
        # Arrange
        # Setup account snapshots with varying equity values
        snapshots = []
        current_time = datetime.datetime.utcnow()

        for i in range(30):
            snapshot = MagicMock()
            snapshot.timestamp = current_time - datetime.timedelta(days=i)

            # 15 days ago we had peak equity of $12,000
            if i == 15:
                snapshot.equity = 12000.0
            else:
                snapshot.equity = 10000.0

            snapshot.balance = 10000.0
            snapshots.append(snapshot)

        risk_validator.account_repository.get_daily_snapshots.return_value = snapshots

        # Current equity is $10,000
        mock_connector.get_account_info.return_value = {
            'balance': 10000.0,
            'equity': 10000.0
        }

        # Act
        drawdown = risk_validator._calculate_drawdown()

        # Assert
        # Drawdown should be (12000 - 10000) / 12000 = 0.167 or 16.7%
        assert drawdown == pytest.approx(0.167, 0.01)

    def test_calculate_drawdown_no_snapshots(self, risk_validator):
        """Test _calculate_drawdown method when no snapshots exist."""
        # Arrange
        risk_validator.account_repository.get_daily_snapshots.return_value = []

        # Act
        drawdown = risk_validator._calculate_drawdown()

        # Assert
        assert drawdown == 0.0

    def test_calculate_drawdown_error(self, risk_validator, mock_account_repo):
        """Test _calculate_drawdown method when repository raises an error."""
        # Arrange
        mock_account_repo.get_daily_snapshots.side_effect = Exception("Database error")

        # Act
        drawdown = risk_validator._calculate_drawdown()

        # Assert
        assert drawdown == 0.0  # Should return a safe default value

    def test_validate_stop_loss_buy_valid(self, risk_validator):
        """Test validate_stop_loss for a valid BUY stop loss."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0
        stop_loss_price = 1990.0  # $10 below entry

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is True

    def test_validate_stop_loss_sell_valid(self, risk_validator):
        """Test validate_stop_loss for a valid SELL stop loss."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 1  # SELL
        entry_price = 2000.0
        stop_loss_price = 2010.0  # $10 above entry

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is True

    def test_validate_stop_loss_buy_invalid_direction(self, risk_validator):
        """Test validate_stop_loss for BUY with stop loss above entry (invalid)."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0
        stop_loss_price = 2010.0  # Stop loss ABOVE entry for BUY is invalid

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_validate_stop_loss_sell_invalid_direction(self, risk_validator):
        """Test validate_stop_loss for SELL with stop loss below entry (invalid)."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 1  # SELL
        entry_price = 2000.0
        stop_loss_price = 1990.0  # Stop loss BELOW entry for SELL is invalid

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_validate_stop_loss_too_tight(self, risk_validator):
        """Test validate_stop_loss when stop loss is too close to entry."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0
        stop_loss_price = 1999.0  # Only $1 below entry, too tight for XAUUSD

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_validate_stop_loss_too_wide(self, risk_validator):
        """Test validate_stop_loss when stop loss is too far from entry."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0
        stop_loss_price = 1900.0  # $100 below entry, too wide for XAUUSD

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_validate_stop_loss_zero_stop(self, risk_validator):
        """Test validate_stop_loss with a zero stop loss."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0
        stop_loss_price = 0.0  # Invalid stop loss

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_validate_stop_loss_invalid_inputs(self, risk_validator):
        """Test validate_stop_loss with various invalid inputs."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = 2000.0

        # Act & Assert - Test with None stop loss
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, None)
        assert result is False

        # Test with invalid order type
        result = risk_validator.validate_stop_loss(symbol, 99, entry_price, 1990.0)
        assert result is False

        # Test with None entry price
        result = risk_validator.validate_stop_loss(symbol, order_type, None, 1990.0)
        assert result is False

        # Test with None symbol
        result = risk_validator.validate_stop_loss(None, order_type, entry_price, 1990.0)
        assert result is False

    def test_validate_stop_loss_error_in_symbol_info(self, risk_validator, mock_connector):
        """Test validate_stop_loss when symbol info returns an error."""
        # Arrange
        mock_connector.get_symbol_info.side_effect = Exception("Symbol not found")

        # Act
        result = risk_validator.validate_stop_loss("XAUUSD", 0, 2000.0, 1990.0)

        # Assert
        assert result is False

    def test_can_open_new_position_with_empty_positions(self, risk_validator, mock_connector):
        """Test can_open_new_position when positions is an empty list."""
        # Arrange
        mock_connector.get_positions.return_value = []

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is True
        assert "All risk checks passed" in reason

    def test_can_open_new_position_with_none_positions(self, risk_validator, mock_connector):
        """Test can_open_new_position when positions is None."""
        # Arrange
        mock_connector.get_positions.return_value = None

        # Act
        result, reason = risk_validator.can_open_new_position('XAUUSD')

        # Assert
        assert result is True
        assert "All risk checks passed" in reason

    def test_validate_stop_loss_negative_entry(self, risk_validator):
        """Test validate_stop_loss with a negative entry price."""
        # Arrange
        symbol = "XAUUSD"
        order_type = 0  # BUY
        entry_price = -100.0  # Invalid negative entry
        stop_loss_price = 1990.0

        # Act
        result = risk_validator.validate_stop_loss(symbol, order_type, entry_price, stop_loss_price)

        # Assert
        assert result is False

    def test_calculate_drawdown_with_account_info_error(self, risk_validator, mock_connector):
        """Test _calculate_drawdown when get_account_info fails."""
        # Arrange
        mock_connector.get_account_info.side_effect = Exception("Connection error")

        # Act
        drawdown = risk_validator._calculate_drawdown()

        # Assert
        assert drawdown == 0.0  # Should return safe default

    def test_calculate_drawdown_with_zero_peak_equity(self, risk_validator, mock_account_repo):
        """Test _calculate_drawdown when peak equity is zero."""
        # Arrange
        snapshots = []
        for i in range(5):
            snapshot = MagicMock()
            snapshot.equity = 0.0  # Zero equity snapshots
            snapshots.append(snapshot)

        mock_account_repo.get_daily_snapshots.return_value = snapshots

        # Current equity is non-zero
        risk_validator.connector.get_account_info.return_value = {
            'equity': 1000.0
        }

        # Act
        drawdown = risk_validator._calculate_drawdown()

        # Assert
        assert drawdown == 0.0  # Should avoid division by zero

    def test_calculate_daily_loss_with_none_profit(self, risk_validator, mock_trade_repo):
        """Test _calculate_daily_loss with trades having None profit."""
        # Arrange
        today = datetime.datetime.utcnow().date()
        today_start = datetime.datetime.combine(today, datetime.datetime.min.time())

        trades = []
        trade = MagicMock()
        trade.open_time = today_start
        trade.profit = None  # None profit should be handled gracefully
        trades.append(trade)

        mock_trade_repo.get_trades_by_date_range.return_value = trades

        # Act
        daily_loss = risk_validator._calculate_daily_loss()

        # Assert
        assert daily_loss == 0.0  # Should handle None profit gracefully