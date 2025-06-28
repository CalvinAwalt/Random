import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from datetime import datetime, timedelta

class Stock:
    def __init__(self, symbol, initial_price, volatility):
        self.symbol = symbol
        self.price_history = [initial_price]
        self.volatility = volatility  # Percentage of price that can change
        self.current_price = initial_price
        self.last_change = 0  # Percentage change
        self.timestamp = datetime.now()
        
    def update_price(self):
        """Update stock price based on random walk with momentum"""
        # Random price change influenced by volatility and previous movement
        change_percent = (random.uniform(-1, 1) * self.volatility 
        change_percent += self.last_change * 0.3  # Add some momentum
        
        # Calculate new price
        new_price = self.current_price * (1 + change_percent)
        self.last_change = change_percent
        self.current_price = max(0.01, new_price)  # Prevent negative prices
        self.price_history.append(self.current_price)
        self.timestamp = datetime.now()
        return self.current_price

class Trader:
    def __init__(self, name, initial_capital):
        self.name = name
        self.capital = initial_capital
        self.portfolio = {}  # {symbol: shares}
        self.trade_history = []
        
    def buy(self, stock, shares):
        cost = stock.current_price * shares
        if cost > self.capital:
            print(f"{self.name}: Not enough capital to buy {shares} shares of {stock.symbol}")
            return False
            
        self.capital -= cost
        self.portfolio[stock.symbol] = self.portfolio.get(stock.symbol, 0) + shares
        self.trade_history.append({
            'type': 'buy',
            'symbol': stock.symbol,
            'shares': shares,
            'price': stock.current_price,
            'timestamp': datetime.now()
        })
        return True
        
    def sell(self, stock, shares):
        if self.portfolio.get(stock.symbol, 0) < shares:
            print(f"{self.name}: Not enough shares to sell {shares} of {stock.symbol}")
            return False
            
        proceeds = stock.current_price * shares
        self.capital += proceeds
        self.portfolio[stock.symbol] -= shares
        if self.portfolio[stock.symbol] == 0:
            del self.portfolio[stock.symbol]
            
        self.trade_history.append({
            'type': 'sell',
            'symbol': stock.symbol,
            'shares': shares,
            'price': stock.current_price,
            'timestamp': datetime.now()
        })
        return True
        
    def portfolio_value(self, market):
        """Calculate total portfolio value"""
        value = self.capital
        for symbol, shares in self.portfolio.items():
            if symbol in market.stocks:
                value += market.stocks[symbol].current_price * shares
        return value

class Market:
    def __init__(self):
        self.stocks = {}
        self.traders = []
        self.price_history = {}  # For visualization
        self.time_elapsed = 0
        
    def add_stock(self, symbol, initial_price, volatility):
        self.stocks[symbol] = Stock(symbol, initial_price, volatility)
        self.price_history[symbol] = deque([initial_price], maxlen=100)
        
    def add_trader(self, trader):
        self.traders.append(trader)
        
    def update_market(self):
        """Update all stock prices and execute trader strategies"""
        self.time_elapsed += 1
        
        # Update all stock prices
        for symbol, stock in self.stocks.items():
            stock.update_price()
            self.price_history[symbol].append(stock.current_price)
        
        # Execute trader strategies
        for trader in self.traders:
            self.execute_trader_strategy(trader)
    
    def execute_trader_strategy(self, trader):
        """Basic trading strategy - random buying/selling"""
        if random.random() < 0.3:  # 30% chance to trade
            stock = random.choice(list(self.stocks.values()))
            action = random.choice(['buy', 'sell'])
            shares = random.randint(1, 10)
            
            if action == 'buy':
                trader.buy(stock, shares)
            else:
                trader.sell(stock, shares)

class MarketVisualization:
    def __init__(self, market):
        self.market = market
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.lines = {}
        
        # Setup plot for each stock
        for symbol in self.market.stocks:
            line, = self.ax.plot([], [], label=symbol)
            self.lines[symbol] = line
            
        self.ax.legend()
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price')
        self.ax.set_title('Stock Market Simulation')
        
    def update_plot(self, frame):
        """Update the plot with new price data"""
        for symbol, line in self.lines.items():
            if symbol in self.market.price_history:
                data = self.market.price_history[symbol]
                line.set_data(range(len(data)), data)
                
        # Adjust axes
        self.ax.relim()
        self.ax.autoscale_view()
        return self.lines.values()

def run_simulation(duration=60):
    """Run the stock market simulation"""
    market = Market()
    
    # Add some stocks
    market.add_stock('AAPL', 150, 0.02)  # symbol, initial price, volatility
    market.add_stock('GOOG', 2800, 0.015)
    market.add_stock('TSLA', 700, 0.03)
    market.add_stock('AMZN', 3300, 0.025)
    
    # Add some traders
    market.add_trader(Trader('Alice', 100000))
    market.add_trader(Trader('Bob', 75000))
    market.add_trader(Trader('Charlie', 150000))
    
    # Setup visualization
    vis = MarketVisualization(market)
    ani = animation.FuncAnimation(
        vis.fig, vis.update_plot, interval=500, blit=True
    )
    
    plt.show(block=False)
    
    # Run the simulation
    start_time = time.time()
    while time.time() - start_time < duration:
        market.update_market()
        time.sleep(0.5)  # Slow down for visualization
        
        # Print some market data
        if market.time_elapsed % 5 == 0:
            print(f"\n--- Market Update (Time: {market.time_elapsed}) ---")
            for symbol, stock in market.stocks.items():
                print(f"{symbol}: ${stock.current_price:.2f} ({stock.last_change*100:.2f}%)")
            
            print("\nTrader Portfolio Values:")
            for trader in market.traders:
                print(f"{trader.name}: ${trader.portfolio_value(market):.2f}")
    
    return market

if __name__ == "__main__":
    print("Starting stock market simulation...")
    simulated_market = run_simulation(duration=120)  # Run for 2 minutes
    print("\nSimulation complete!")