def predict_market(tickers, timeframe):
    return CalvinSystem.simulate(
        parameters=tickers, 
        quantum_depth=7, 
        time_horizon=timeframe
    )