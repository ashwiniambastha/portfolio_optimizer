"""
Portfolio Optimization Agent API
Week 5: REST API for portfolio optimization
FastAPI endpoints for Markowitz optimization
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

try:
    from agent import PortfolioOptimizationAgent
except:
    import sys
    sys.path.append('/home/claude/agents/portfolio_optimization')
    from agent import PortfolioOptimizationAgent


# Initialize FastAPI app
app = FastAPI(
    title="Portfolio Optimization Agent API",
    description="Modern Portfolio Theory optimization using Markowitz",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = PortfolioOptimizationAgent()


# Pydantic models
class OptimizeRequest(BaseModel):
    symbols: List[str]
    max_weight: float = 0.4
    min_weight: float = 0.0
    period: str = "1y"


class TargetReturnRequest(BaseModel):
    symbols: List[str]
    target_return: float
    max_weight: float = 0.4
    min_weight: float = 0.0
    period: str = "1y"


class SignalOptimizeRequest(BaseModel):
    symbols: List[str]
    portfolio_value: float = 100000
    max_weight: float = 0.4
    period: str = "1y"


# Health check
@app.get("/")
def root():
    """API health check"""
    return {
        "service": "Portfolio Optimization Agent",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/optimize/max-sharpe",
            "/optimize/min-variance",
            "/optimize/target-return",
            "/optimize/with-signals",
            "/efficient-frontier"
        ]
    }


# Maximum Sharpe ratio optimization
@app.post("/optimize/max-sharpe")
def optimize_max_sharpe(request: OptimizeRequest):
    """
    Optimize portfolio to maximize Sharpe ratio
    
    Body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "max_weight": 0.4,
            "min_weight": 0.0,
            "period": "1y"
        }
    
    Returns:
        Optimal weights and portfolio metrics
    """
    try:
        # Fetch returns data
        returns_df = agent.fetch_returns_data(request.symbols, request.period)
        
        if returns_df is None:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch returns data for all symbols"
            )
        
        # Optimize
        result = agent.optimize_max_sharpe(
            returns_df,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        
        # Add metadata
        result['symbols'] = request.symbols
        result['optimization_type'] = 'max_sharpe'
        result['period'] = request.period
        result['status'] = 'success'
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Minimum variance optimization
@app.post("/optimize/min-variance")
def optimize_min_variance(request: OptimizeRequest):
    """
    Optimize portfolio to minimize variance (risk)
    
    Body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "max_weight": 0.4,
            "min_weight": 0.0,
            "period": "1y"
        }
    
    Returns:
        Optimal weights and portfolio metrics
    """
    try:
        # Fetch returns data
        returns_df = agent.fetch_returns_data(request.symbols, request.period)
        
        if returns_df is None:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch returns data for all symbols"
            )
        
        # Optimize
        result = agent.optimize_min_variance(
            returns_df,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        
        # Add metadata
        result['symbols'] = request.symbols
        result['optimization_type'] = 'min_variance'
        result['period'] = request.period
        result['status'] = 'success'
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Target return optimization
@app.post("/optimize/target-return")
def optimize_target_return(request: TargetReturnRequest):
    """
    Optimize portfolio to minimize variance given target return
    
    Body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "target_return": 0.15,
            "max_weight": 0.4,
            "min_weight": 0.0,
            "period": "1y"
        }
    
    Returns:
        Optimal weights and portfolio metrics
    """
    try:
        # Fetch returns data
        returns_df = agent.fetch_returns_data(request.symbols, request.period)
        
        if returns_df is None:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch returns data for all symbols"
            )
        
        # Optimize
        result = agent.optimize_target_return(
            returns_df,
            target_return=request.target_return,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        
        # Add metadata
        result['symbols'] = request.symbols
        result['optimization_type'] = 'target_return'
        result['period'] = request.period
        result['status'] = 'success'
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optimization with alpha signals
@app.post("/optimize/with-signals")
def optimize_with_signals(request: SignalOptimizeRequest):
    """
    Optimize portfolio incorporating alpha signals
    
    Body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "portfolio_value": 100000,
            "max_weight": 0.4,
            "period": "1y"
        }
    
    Returns:
        Optimized portfolio with signal adjustments
    """
    try:
        result = agent.optimize_with_signals(
            request.symbols,
            portfolio_value=request.portfolio_value,
            max_weight=request.max_weight
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        result['optimization_type'] = 'with_signals'
        result['period'] = request.period
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Efficient frontier generation
@app.post("/efficient-frontier")
def generate_efficient_frontier(
    request: OptimizeRequest,
    num_portfolios: int = Query(50, description="Number of portfolios to generate")
):
    """
    Generate efficient frontier
    
    Body:
        {
            "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "max_weight": 0.4,
            "period": "1y"
        }
    
    Query params:
        num_portfolios: Number of portfolios (default 50)
    
    Returns:
        DataFrame with efficient frontier portfolios
    """
    try:
        # Fetch returns data
        returns_df = agent.fetch_returns_data(request.symbols, request.period)
        
        if returns_df is None:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch returns data for all symbols"
            )
        
        # Generate frontier
        frontier_df = agent.generate_efficient_frontier(
            returns_df,
            num_portfolios=num_portfolios,
            max_weight=request.max_weight
        )
        
        # Convert to dict for JSON
        result = {
            'status': 'success',
            'symbols': request.symbols,
            'num_portfolios': len(frontier_df),
            'frontier': frontier_df.to_dict(orient='records')
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Quick optimize endpoint (GET)
@app.get("/optimize/{optimization_type}")
def quick_optimize(
    optimization_type: str,
    symbols: str = Query(..., description="Comma-separated symbols (e.g., AAPL,GOOGL,MSFT)"),
    max_weight: float = Query(0.4, description="Maximum weight per asset"),
    period: str = Query("1y", description="Time period")
):
    """
    Quick optimization endpoint (GET version)
    
    Args:
        optimization_type: 'max-sharpe' or 'min-variance'
        symbols: Comma-separated stock symbols
        max_weight: Maximum weight per asset
        period: Time period
    
    Returns:
        Optimal portfolio
    """
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Fetch returns data
        returns_df = agent.fetch_returns_data(symbols_list, period)
        
        if returns_df is None:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch returns data"
            )
        
        # Optimize based on type
        if optimization_type == 'max-sharpe':
            result = agent.optimize_max_sharpe(returns_df, max_weight=max_weight)
        elif optimization_type == 'min-variance':
            result = agent.optimize_min_variance(returns_df, max_weight=max_weight)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid optimization type: {optimization_type}"
            )
        
        result['symbols'] = symbols_list
        result['optimization_type'] = optimization_type
        result['period'] = period
        result['status'] = 'success'
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting Portfolio Optimization Agent API on port 8003...")
    print("Endpoints:")
    print("  POST /optimize/max-sharpe")
    print("  POST /optimize/min-variance")
    print("  POST /optimize/target-return")
    print("  POST /optimize/with-signals")
    print("  POST /efficient-frontier")
    print("  GET  /optimize/{type}?symbols=AAPL,GOOGL,MSFT")
    print("\nSwagger docs: http://localhost:8003/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8003)