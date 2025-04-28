# utils/risk_reward_validator.py
"""
Utility module for validating risk-reward ratios of trades.
Can be used independently to check trade parameters.
"""
import logging

logger = logging.getLogger(__name__)


def validate_risk_reward(order_type, entry_price, stop_loss, take_profit, max_ratio=2.5):
    """Validate that risk-reward ratio is acceptable.

    Args:
        order_type (int): Order type (0=BUY, 1=SELL)
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        take_profit (float): Take profit price
        max_ratio (float): Maximum acceptable risk:reward ratio

    Returns:
        dict: Validation result with keys:
            - valid (bool): True if risk-reward is acceptable
            - risk (float): Calculated risk amount
            - reward (float): Calculated reward amount
            - ratio (float): Risk-reward ratio
            - message (str): Error/info message
    """
    result = {
        'valid': False,
        'risk': 0.0,
        'reward': 0.0,
        'ratio': 0.0,
        'message': ""
    }

    # Basic validation
    if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
        result['message'] = f"Invalid prices: entry={entry_price}, stop={stop_loss}, tp={take_profit}"
        return result

    # Calculate risk and reward
    if order_type == 0:  # BUY
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # SELL
        risk = stop_loss - entry_price
        reward = entry_price - take_profit

    result['risk'] = risk
    result['reward'] = reward

    # Validate both are positive
    if risk <= 0:
        result['message'] = f"Invalid risk: {risk} for {order_type} order"
        return result

    if reward <= 0:
        result['message'] = f"Invalid reward: {reward} for {order_type} order"
        return result

    # Calculate risk-reward ratio
    ratio = risk / reward
    result['ratio'] = ratio

    # Check against maximum acceptable ratio
    if ratio > max_ratio:
        result['message'] = f"Poor risk-reward ratio: {ratio:.2f}. Maximum acceptable is {max_ratio}"
        return result

    # All checks passed
    result['valid'] = True
    result['message'] = f"Valid risk-reward ratio: {ratio:.2f}"
    return result


def analyze_trade(symbol, signal_type, entry_price, stop_loss, take_profit):
    """Analyze a trade setup and print comprehensive risk analysis.

    Args:
        symbol (str): Trading symbol
        signal_type (str): 'BUY' or 'SELL'
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        take_profit (float): Take profit price

    Returns:
        bool: True if trade setup is valid, False otherwise
    """
    order_type = 0 if signal_type == "BUY" else 1

    # Run validation
    result = validate_risk_reward(order_type, entry_price, stop_loss, take_profit)

    # Print comprehensive analysis
    print(f"\n=== TRADE ANALYSIS: {signal_type} {symbol} ===")
    print(f"Entry Price: {entry_price:.2f}")
    print(f"Stop Loss:   {stop_loss:.2f}")
    print(f"Take Profit: {take_profit:.2f}")
    print(f"Risk:        {result['risk']:.2f} points")
    print(f"Reward:      {result['reward']:.2f} points")
    print(f"R:R Ratio:   {result['ratio']:.2f}")