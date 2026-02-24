import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import time

class MarketDataAgent:
    """
    Market Data Agent - Fetches and manages market data
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_cache = {}
        self.last_update = {}
        
    def fetch_realtime_data(self, symbol: str) -> Dict:
        """
        Fetch current market data for a symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price data
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'open': info.get('regularMarketOpen'),
                'high': info.get('dayHigh'),
                'low': info.get('dayLow'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
            }
            
            print(f"✓ Fetched data for {symbol}: ${data['price']}")
            return data
            
        except Exception as e:
            print(f"✗ Error fetching {symbol}: {str(e)}")
            return None
    
    def fetch_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical data
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            
            print(f"✓ Fetched {len(df)} days of historical data for {symbol}")
            return df
            
        except Exception as e:
            print(f"✗ Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def fetch_all_symbols(self) -> Dict[str, Dict]:
        """
        Fetch data for all symbols
        """
        results = {}
        for symbol in self.symbols:
            data = self.fetch_realtime_data(symbol)
            if data:
                results[symbol] = data
                self.data_cache[symbol] = data
                self.last_update[symbol] = datetime.now()
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest cached price
        """
        if symbol in self.data_cache:
            return self.data_cache[symbol]['price']
        return None


# Demo/Test
if __name__ == "__main__":
    # Test with some popular stocks
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    agent = MarketDataAgent(symbols)
    
    print("=" * 60)
    print("MARKET DATA AGENT - DEMO")
    print("=" * 60)
    
    # Fetch real-time data
    print("\n1. Fetching real-time data...")
    data = agent.fetch_all_symbols()
    
    # Display results
    print("\n2. Current Prices:")
    for symbol, info in data.items():
        print(f"   {symbol}: ${info['price']:.2f} (Vol: {info['volume']:,})")
    
    # Fetch historical data for one stock
    print("\n3. Fetching historical data for AAPL...")
    hist_data = agent.fetch_historical_data('AAPL', period='1mo')
    if hist_data is not None:
        print(f"   Last 5 days:")
        print(hist_data[['Close', 'Volume']].tail())