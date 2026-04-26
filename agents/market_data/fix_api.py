from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as _np

# Fixed sanitise function to prevent infinite recursion
def sanitise(obj):
    """Recursively replace float NaN/Inf with None for safe JSON serialisation."""
    if isinstance(obj, float):
        if _np.isnan(obj) or _np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    
    # Handle Numpy types
    if isinstance(obj, (_np.floating, _np.complexfloating)):
        f = float(obj)
        return None if (_np.isnan(f) or _np.isinf(f)) else f
    if isinstance(obj, _np.integer):
        return int(obj)
        
    return obj # Return as is if no rules match

from agent import MarketDataAgent
from storage import MarketDataStorage

app = FastAPI(title="Market Data Agent API")

# Initialize without a fixed list (or keep the list as a starting cache)
# The agent logic should now be dynamic
agent = MarketDataAgent() 
storage = MarketDataStorage()

class PriceRequest(BaseModel):
    symbol: str

@app.get("/")
def root():
    return {"status": "Market Data Agent API Running", "mode": "Dynamic Search Enabled"}

@app.get("/price/{symbol}")
def get_price(symbol: str):
    """Get current price for ANY symbol requested by the Gradio frontend"""
    sym = symbol.upper().strip()
    
    # Fetch data dynamically via the agent
    data = agent.fetch_realtime_data(sym)
    
    if data and data.get('price') is not None:
        # Automatically persist any newly searched stock to your database
        storage.save_realtime_data(data)
        return sanitise(data)

    raise HTTPException(status_code=404, detail=f"Symbol '{sym}' not found or invalid")

@app.get("/prices")
def get_all_prices():
    """Get prices for all symbols currently tracked in the database"""
    # Instead of fetching from a hardcoded list, we pull what's currently in storage
    tracked_symbols = storage.get_all_tracked_symbols() # Ensure this method exists in your storage.py
    
    if not tracked_symbols:
        return {"message": "No symbols currently tracked. Search for a stock first."}
        
    data = agent.fetch_multiple_symbols(tracked_symbols)
    for symbol, info in data.items():
        storage.save_realtime_data(info)
    return sanitise(data)

@app.get("/historical/{symbol}")
def get_historical(symbol: str, period: str = "1y"):
    """Get historical data for any symbol"""
    sym = symbol.upper().strip()
    df = agent.fetch_historical_data(sym, period)
    
    if df is not None and not df.empty:
        storage.save_historical_data(sym, df)
        # Convert index to string (dates) for JSON safety
        return df.reset_index().to_dict(orient='records')
        
    raise HTTPException(status_code=404, detail=f"Historical data for {sym} not found")

@app.get("/latest")
def get_latest():
    """Get latest prices directly from the database"""
    df = storage.get_latest_prices()
    if df.empty:
        return []
    return df.to_dict(orient='records')

if __name__ == "__main__":
    # Ensure this matches the port in your Gradio app (MARKET_DATA_API = "http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)