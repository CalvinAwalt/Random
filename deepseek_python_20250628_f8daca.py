from calvin_finance import QuantumMarketPredictor

predictor = QuantumMarketPredictor()
aapl_forecast = predictor.forecast('AAPL', timeframe='1W')