"""
Portfolio Optimization Agent — REST API
agents/portfolio_optimization/api.py

FastAPI service exposing Markowitz optimization endpoints.
Run with:
    cd agents/portfolio_optimization
    python api.py

Swagger UI → http://localhost:8003/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from agent import PortfolioOptimizationAgent

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Portfolio Optimization Agent API",
    description="Modern Portfolio Theory optimization using Markowitz Mean-Variance",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PortfolioOptimizationAgent()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    symbols:    List[str]
    max_weight: float = 0.4
    min_weight: float = 0.0
    period:     str   = "1y"


class TargetReturnRequest(BaseModel):
    symbols:       List[str]
    target_return: float
    max_weight:    float = 0.4
    min_weight:    float = 0.0
    period:        str   = "1y"


class SignalOptimizeRequest(BaseModel):
    symbols:         List[str]
    portfolio_value: float = 100_000
    max_weight:      float = 0.4
    period:          str   = "1y"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Portfolio Optimization Agent",
        "status":  "running",
        "version": "1.0.0",
        "endpoints": [
            "POST /optimize/max-sharpe",
            "POST /optimize/min-variance",
            "POST /optimize/target-return",
            "POST /optimize/with-signals",
            "POST /efficient-frontier",
            "GET  /optimize/{type}?symbols=AAPL,GOOGL,MSFT",
        ],
    }


@app.post("/optimize/max-sharpe")
def optimize_max_sharpe(req: OptimizeRequest):
    """Find the portfolio with the highest Sharpe ratio."""
    returns_df = agent.fetch_returns_data(req.symbols, req.period)
    if returns_df is None:
        raise HTTPException(status_code=400, detail="Could not fetch returns data")
    result = agent.optimize_max_sharpe(returns_df, req.max_weight, req.min_weight)
    result.update(symbols=req.symbols, period=req.period, status="success")
    return result


@app.post("/optimize/min-variance")
def optimize_min_variance(req: OptimizeRequest):
    """Find the minimum-variance portfolio."""
    returns_df = agent.fetch_returns_data(req.symbols, req.period)
    if returns_df is None:
        raise HTTPException(status_code=400, detail="Could not fetch returns data")
    result = agent.optimize_min_variance(returns_df, req.max_weight, req.min_weight)
    result.update(symbols=req.symbols, period=req.period, status="success")
    return result


@app.post("/optimize/target-return")
def optimize_target_return(req: TargetReturnRequest):
    """Minimise variance subject to a target expected return."""
    returns_df = agent.fetch_returns_data(req.symbols, req.period)
    if returns_df is None:
        raise HTTPException(status_code=400, detail="Could not fetch returns data")
    result = agent.optimize_target_return(
        returns_df, req.target_return, req.max_weight, req.min_weight
    )
    result.update(symbols=req.symbols, period=req.period, status="success")
    return result


@app.post("/optimize/with-signals")
def optimize_with_signals(req: SignalOptimizeRequest):
    """Max-Sharpe portfolio adjusted by Week 4 alpha signals."""
    result = agent.optimize_with_signals(
        req.symbols, req.portfolio_value, req.max_weight
    )
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    result["period"] = req.period
    return result


@app.post("/efficient-frontier")
def generate_efficient_frontier(
    req: OptimizeRequest,
    num_portfolios: int = Query(50, description="Number of frontier portfolios"),
):
    """Generate the Markowitz Efficient Frontier."""
    returns_df = agent.fetch_returns_data(req.symbols, req.period)
    if returns_df is None:
        raise HTTPException(status_code=400, detail="Could not fetch returns data")
    df = agent.generate_efficient_frontier(returns_df, num_portfolios, req.max_weight)
    return {
        "status":         "success",
        "symbols":        req.symbols,
        "num_portfolios": len(df),
        "frontier":       df.to_dict(orient="records"),
    }


@app.get("/optimize/{optimization_type}")
def quick_optimize(
    optimization_type: str,
    symbols:    str   = Query(..., description="Comma-separated symbols, e.g. AAPL,GOOGL,MSFT"),
    max_weight: float = Query(0.4),
    period:     str   = Query("1y"),
):
    """Quick GET endpoint — no request body required."""
    syms       = [s.strip().upper() for s in symbols.split(",")]
    returns_df = agent.fetch_returns_data(syms, period)
    if returns_df is None:
        raise HTTPException(status_code=400, detail="Could not fetch returns data")

    if optimization_type == "max-sharpe":
        result = agent.optimize_max_sharpe(returns_df, max_weight)
    elif optimization_type == "min-variance":
        result = agent.optimize_min_variance(returns_df, max_weight)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown optimization type: '{optimization_type}'. "
                   "Use 'max-sharpe' or 'min-variance'.",
        )
    result.update(symbols=syms, period=period, status="success")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Portfolio Optimization API on port 8003 ...")
    print("Swagger UI → http://localhost:8003/docs")
    uvicorn.run(app, host="0.0.0.0", port=8003)