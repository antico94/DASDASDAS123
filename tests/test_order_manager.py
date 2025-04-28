# tests/test_order_manager.py
import unittest
from unittest.mock import MagicMock, patch
import json
from datetime import datetime
from execution.order_manager import EnhancedOrderManager
from data.models import StrategySignal


class TestOrderManager(unittest.TestCase):
    def setUp(self):
        # Create mocks
        self.connector = MagicMock()
        self.trade_repository = MagicMock()
        self.signal_repository = MagicMock()
        self.position_sizer = MagicMock()
        self.risk_validator = MagicMock()

        # Configure mocks with default returns
        self.connector.get_symbol_info.return_value = {
            'min_lot': 0.01,
            'max_lot': 10.0,
            'lot_step': 0.01
        }

        self.position_sizer.calculate_position_size.return_value = 0.1
        self.position_sizer.validate_position_size.return_value = True

        self.risk_validator.can_open_new_position.return_value = (True, "")
        self.risk_validator.validate_stop_loss.return_value = True

        self.connector.place_order.return_value = {
            'ticket': 12345,
            'volume': 0.1,
            'price': 3000.0,
            'symbol': 'XAUUSD',
            'type': 0,
            'stop_loss': 2950.0,
            'take_profit': 3050.0,
            'comment': 'Test'
        }

        # Create order manager instance
        self.order_manager = EnhancedOrderManager(
            connector=self.connector,
            trade_repository=self.trade_repository,
            signal_repository=self.signal_repository,
            position_sizer=self.position_sizer,
            risk_validator=self.risk_validator
        )

    def test_execute_signal_with_none_stop_loss(self):
        """Test executing a signal with None values in metadata."""
        # Create a signal with metadata containing None values
        signal = StrategySignal(
            id=1,
            strategy_name="Test",
            symbol="XAUUSD",
            timeframe="H1",
            timestamp=datetime.utcnow(),
            signal_type="BUY",
            price=3000.0,
            strength=0.8,
            signal_data=json.dumps({
                'stop_loss': None,
                'take_profit_1r': None,
                'take_profit_2r': None
            })
        )

        # Execute the signal
        result = self.order_manager._execute_signal(signal)

        # Check that the signal was processed without error
        self.assertTrue(result)

        # Verify that position_sizer.calculate_position_size was called with valid arguments
        self.position_sizer.calculate_position_size.assert_called_once()
        args = self.position_sizer.calculate_position_size.call_args[1]
        self.assertIsNotNone(args['stop_loss_price'])
        self.assertIsInstance(args['stop_loss_price'], (int, float))

        # Verify that a trade was created
        self.trade_repository.add.assert_called_once()


if __name__ == '__main__':
    unittest.main()