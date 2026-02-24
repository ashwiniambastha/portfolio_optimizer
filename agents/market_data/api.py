from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

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
    """Get current price for a symbol"""
    data = agent.fetch_realtime_data(symbol)
    if data:
        storage.save_realtime_data(data)
        return data
    raise HTTPException(status_code=404, detail="Symbol not found")

@app.get("/prices")
def get_all_prices():
    """Get prices for all tracked symbols"""
    data = agent.fetch_all_symbols()
    for symbol, info in data.items():
        storage.save_realtime_data(info)
    return data

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