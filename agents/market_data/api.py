from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn


def sanitise(obj):
    """Recursively replace float NaN/Inf with None for safe JSON serialisation."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.floating):
            f = float(obj)
            return None if (f != f or f == float('inf') or f == float('-inf')) else f
        if isinstance(obj, _np.integer):
            return int(obj)
    except Exception:
        pass
    return sanitise(obj)



from agent import MarketDataAgent
from storage import MarketDataStorage

app = FastAPI(title="Market Data Agent API")

# Initialize
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
agent = MarketDataAgent(symbols)
storage = MarketDataStorage()

class PriceRequest(BaseModel):
    symbol: str

@app.get("/")
def root():
    return {"status": "Market Data Agent API Running"}

@app.get("/price/{symbol}")
def get_price(symbol: str):
    """Get current price for ANY symbol requested by the Gradio frontend"""
    sym = symbol.upper().strip()
    
    # Try original symbol first
    data = agent.fetch_realtime_data(sym)
    
    # If price is 0 or None, try with Indian exchange suffixes
    if not data or not data.get('price'):
        for suffix in ['.NS', '.BO']:
            data = agent.fetch_realtime_data(sym + suffix)
            if data and data.get('price'):
                break
    
    if data and data.get('price') is not None:
        storage.save_realtime_data(data)
        return sanitise(data)
    
    raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found or invalid") 

@app.get("/prices")
def get_all_prices():
    """Get prices for all tracked symbols"""
    data = agent.fetch_all_symbols()
    for symbol, info in data.items():
        storage.save_realtime_data(info)
    return sanitise(data)


@app.get("/historical/{symbol}")
def get_historical(symbol: str, period: str = "1mo"):
    """Get historical data"""
    df = agent.fetch_historical_data(symbol, period)
    if df is not None:
        storage.save_historical_data(symbol, df)
        return df.to_dict(orient='records')
    raise HTTPException(status_code=404, detail="Data not found")

@app.get("/latest")
def get_latest():
    """Get latest prices from database"""
    df = storage.get_latest_prices()
    return df.to_dict(orient='records')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)