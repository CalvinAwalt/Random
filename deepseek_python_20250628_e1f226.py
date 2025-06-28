# Connect to brokerage API
from alpaca_trade_api import REST

def execute_trades(signals):
    api = REST('API_KEY', 'SECRET_KEY')
    for ticker, signal in signals.items():
        if 'BUY' in signal['action']:
            api.submit_order(
                symbol=ticker,
                qty=calculate_position_size(signal['confidence']),
                side='buy',
                stop_loss=signal['stop_loss'],
                take_profit=signal['target_price']
            )