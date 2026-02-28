from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.risk_management.agent import RiskManagementAgent

app = FastAPI(title="Risk Management Agent API")

# Initialize
risk_agent = RiskManagementAgent(market_data_api="http://localhost:8000")

class RiskAssessmentRequest(BaseModel):
    symbol: str
    portfolio_value: float = 100000

@app.get("/")
def root():
    return {"status": "Risk Management Agent API Running"}

@app.get("/var/{symbol}")
def get_var(symbol: str, confidence: float = 0.95, portfolio_value: float = 100000):
    """Calculate Value at Risk"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_var(returns, confidence, portfolio_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cvar/{symbol}")
def get_cvar(symbol: str, confidence: float = 0.95, portfolio_value: float = 100000):
    """Calculate Conditional VaR"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_cvar(returns, confidence, portfolio_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/volatility/{symbol}")
def get_volatility(symbol: str):
    """Calculate volatility"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_volatility(returns)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sharpe/{symbol}")
def get_sharpe(symbol: str, risk_free_rate: float = 0.04):
    """Calculate Sharpe Ratio"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_sharpe_ratio(returns, risk_free_rate)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/beta/{symbol}")
def get_beta(symbol: str, market: str = "SPY"):
    """Calculate Beta"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_beta(returns, market)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drawdown/{symbol}")
def get_drawdown(symbol: str):
    """Calculate Maximum Drawdown"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.calculate_max_drawdown(returns)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stress/{symbol}")
def get_stress_test(symbol: str, portfolio_value: float = 100000):
    """Perform stress testing"""
    try:
        returns = risk_agent.fetch_historical_returns(symbol)
        if returns is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        result = risk_agent.stress_test(returns, portfolio_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess")
def assess_risk(request: RiskAssessmentRequest):
    """Comprehensive risk assessment"""
    try:
        result = risk_agent.assess_risk(request.symbol, request.portfolio_value)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/correlation")
def get_correlation(symbols: str):
    """Calculate correlation matrix
    symbols: comma-separated list (e.g., AAPL,GOOGL,MSFT)
    """
    try:
        symbol_list = [s.strip() for s in symbols.split(',')]
        result = risk_agent.calculate_correlation_matrix(symbol_list)
        
        if result is None:
            raise HTTPException(status_code=404, detail="Could not fetch data")
        
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)