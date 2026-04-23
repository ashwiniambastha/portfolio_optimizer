"""
Alpha Signal Agent API
Week 4: REST API for trading signals
FastAPI endpoints for technical analysis signals
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import pandas as pd
import numpy as np
import requests as req
import yfinance as yf

try:
    from agent import PortfolioOptimizationAgent
except:
    import sys
    sys.path.append('agents/alpha_signal')
    from agent import PortfolioOptimizationAgent

try:
    from signal_generator import SignalGenerator
except:
    import sys
    sys.path.append('agents/alpha_signal')
    from signal_generator import SignalGenerator

try:
    from indicators import TechnicalIndicators
except:
    import sys
    sys.path.append('agents/alpha_signal')
    from indicators import TechnicalIndicators


# Initialize FastAPI app
app = FastAPI(
    title="Alpha Signal Agent API",
    description="Trading signal generation using technical analysis",
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

# Initialize agents
optimization_agent = PortfolioOptimizationAgent()
signal_generator = SignalGenerator()
ti = TechnicalIndicators()

MARKET_DATA_API = "http://localhost:8000"


# ─────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────
def sanitise(obj):
    """Recursively replace all float NaN/Inf values with None so JSON serialisation never fails."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float('inf') or obj == float('-inf')) else obj
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    # numpy scalar
    try:
        import numpy as _np
        if isinstance(obj, _np.floating):
            f = float(obj)
            return None if (f != f or f == float('inf') or f == float('-inf')) else f
        if isinstance(obj, _np.integer):
            return int(obj)
    except Exception:
        pass
    return obj


def fetch_prices(symbol: str, period: str = "1y") -> pd.Series:
    """Fetch closing prices directly from Yahoo Finance.
    Drops the incomplete last candle that yfinance returns for the current trading day
    (its Open/High/Low/Close are NaN until market close).
    """
    try:
        df = yf.Ticker(symbol).history(period=period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance error for {symbol}: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data returned for {symbol}")

    # Drop any rows where Close is NaN (incomplete intraday candle)
    df = df.dropna(subset=['Close'])

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No valid Close prices for {symbol}")

    return df['Close']


# ─────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Alpha Signal Agent API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/signal/{symbol}",
            "/backtest/{symbol}",
            "/indicators/{symbol}",
            "/optimize/max-sharpe",
            "/optimize/min-variance",
            "/optimize/target-return",
            "/optimize/with-signals",
            "/efficient-frontier",
        ]
    }


# ─────────────────────────────────────────────
#  Signal Generation
# ─────────────────────────────────────────────
@app.get("/signal/{symbol}")
def get_signal(symbol: str, portfolio_value: float = 100000):
    """
    Generate trading signal for a symbol using technical indicators.
    Returns recommendation, confidence, signals, position sizing.
    """
    try:
        prices = fetch_prices(symbol, "1y")

        signal = signal_generator.generate_signal(prices)

        # Position sizing based on confidence
        confidence_abs = abs(signal['confidence'])
        position_size_pct = max(0.0, min(25.0, confidence_abs * 0.25))
        actual_position_usd = portfolio_value * position_size_pct / 100.0
        recommended_shares = (
            int(actual_position_usd / signal['current_price'])
            if signal['current_price'] > 0 else 0
        )

        # Volatility / risk
        returns = prices.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252))
        risk_adjusted = volatility < 0.35

        # Clean indicator values (replace NaN with None for JSON)
        clean_indicators = {}
        for k, v in signal.get('indicator_values', {}).items():
            clean_indicators[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

        raw = {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": float(signal['current_price']),
            "timestamp": signal['timestamp'],
            "recommendation": signal['recommendation'],
            "action": signal['action'],
            "confidence": float(signal['confidence']),
            "signals": {k: float(v) for k, v in signal['signals'].items()},
            "explanations": signal['explanations'],
            "indicator_values": clean_indicators,
            "position_sizing": {
                "portfolio_value": portfolio_value,
                "position_size_pct": position_size_pct,
                "actual_position_usd": actual_position_usd,
                "recommended_shares": recommended_shares,
            },
            "risk_metrics": {
                "volatility": volatility,
                "risk_adjusted": risk_adjusted,
            }
        }
        return sanitise(raw)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Backtest
# ─────────────────────────────────────────────
@app.get("/backtest/{symbol}")
def get_backtest(symbol: str,
                 initial_capital: float = 10000,
                 period: str = "1y"):
    """
    Run backtest of the technical-analysis strategy for a symbol.
    """
    try:
        prices = fetch_prices(symbol, period)

        result = signal_generator.backtest_strategy(prices, initial_capital)

        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])

        result['symbol'] = symbol.upper()
        result['period'] = period

        # Serialize dates to strings
        for trade in result.get('trades', []):
            d = trade.get('date')
            if d is not None and hasattr(d, 'strftime'):
                trade['date'] = d.strftime('%Y-%m-%d')
            elif d is not None:
                trade['date'] = str(d)

        for point in result.get('equity_curve', []):
            d = point.get('date')
            if d is not None and hasattr(d, 'strftime'):
                point['date'] = d.strftime('%Y-%m-%d')
            elif d is not None:
                point['date'] = str(d)

        return sanitise(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Raw Indicators
# ─────────────────────────────────────────────
@app.get("/indicators/{symbol}")
def get_indicators(symbol: str, period: str = "6mo"):
    """
    Return raw technical indicator values for a symbol.
    """
    try:
        prices = fetch_prices(symbol, period)

        macd_data = ti.macd(prices)
        bb_data   = ti.bollinger_bands(prices)
        rsi_val   = float(ti.rsi(prices).iloc[-1])

        rsi_interp = (
            "Oversold (Buy)"   if rsi_val < 30 else
            "Overbought (Sell)" if rsi_val > 70 else
            "Neutral"
        )
        macd_sig = "Bullish" if float(macd_data['histogram'].iloc[-1]) > 0 else "Bearish"

        def safe(series):
            v = series.iloc[-1]
            return None if (v is None or np.isnan(v)) else float(v)

        return {
            "symbol": symbol.upper(),
            "period": period,
            "current_price": float(prices.iloc[-1]),
            "moving_averages": {
                "sma_20":  safe(ti.sma(prices, 20)),
                "sma_50":  safe(ti.sma(prices, 50)),
                "sma_200": safe(ti.sma(prices, 200)),
                "ema_12":  safe(ti.ema(prices, 12)),
                "ema_26":  safe(ti.ema(prices, 26)),
            },
            "rsi": {
                "value": rsi_val,
                "interpretation": rsi_interp,
            },
            "macd": {
                "macd_line":   safe(macd_data['macd']),
                "signal_line": safe(macd_data['signal']),
                "histogram":   safe(macd_data['histogram']),
                "signal":      macd_sig,
            },
            "bollinger_bands": {
                "upper":     safe(bb_data['upper']),
                "middle":    safe(bb_data['middle']),
                "lower":     safe(bb_data['lower']),
                "percent_b": safe(bb_data['percent_b']),
                "bandwidth": safe(bb_data['bandwidth']),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Portfolio Optimization endpoints
# ─────────────────────────────────────────────
@app.post("/optimize/max-sharpe")
def optimize_max_sharpe(request: OptimizeRequest):
    """Optimize portfolio to maximize Sharpe ratio."""
    try:
        returns_df = optimization_agent.fetch_returns_data(
            request.symbols, request.period)
        if returns_df is None:
            raise HTTPException(status_code=400,
                                detail="Could not fetch returns data")

        result = optimization_agent.optimize_max_sharpe(
            returns_df,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        result.update(symbols=request.symbols,
                      optimization_type='max_sharpe',
                      period=request.period,
                      status='success')
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/min-variance")
def optimize_min_variance(request: OptimizeRequest):
    """Optimize portfolio to minimize variance."""
    try:
        returns_df = optimization_agent.fetch_returns_data(
            request.symbols, request.period)
        if returns_df is None:
            raise HTTPException(status_code=400,
                                detail="Could not fetch returns data")

        result = optimization_agent.optimize_min_variance(
            returns_df,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        result.update(symbols=request.symbols,
                      optimization_type='min_variance',
                      period=request.period,
                      status='success')
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/target-return")
def optimize_target_return(request: TargetReturnRequest):
    """Optimize portfolio for a target return."""
    try:
        returns_df = optimization_agent.fetch_returns_data(
            request.symbols, request.period)
        if returns_df is None:
            raise HTTPException(status_code=400,
                                detail="Could not fetch returns data")

        result = optimization_agent.optimize_target_return(
            returns_df,
            target_return=request.target_return,
            max_weight=request.max_weight,
            min_weight=request.min_weight
        )
        result.update(symbols=request.symbols,
                      optimization_type='target_return',
                      period=request.period,
                      status='success')
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/with-signals")
def optimize_with_signals(request: SignalOptimizeRequest):
    """Optimize portfolio incorporating alpha signals."""
    try:
        result = optimization_agent.optimize_with_signals(
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


@app.post("/efficient-frontier")
def generate_efficient_frontier(
    request: OptimizeRequest,
    num_portfolios: int = Query(50, description="Number of portfolios")
):
    """Generate efficient frontier."""
    try:
        returns_df = optimization_agent.fetch_returns_data(
            request.symbols, request.period)
        if returns_df is None:
            raise HTTPException(status_code=400,
                                detail="Could not fetch returns data")

        frontier_df = optimization_agent.generate_efficient_frontier(
            returns_df,
            num_portfolios=num_portfolios,
            max_weight=request.max_weight
        )
        return {
            'status': 'success',
            'symbols': request.symbols,
            'num_portfolios': len(frontier_df),
            'frontier': frontier_df.to_dict(orient='records')
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimize/{optimization_type}")
def quick_optimize(
    optimization_type: str,
    symbols: str = Query(..., description="Comma-separated symbols"),
    max_weight: float = Query(0.4),
    period: str = Query("1y")
):
    """Quick GET optimization endpoint."""
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(',')]
        returns_df = optimization_agent.fetch_returns_data(symbols_list, period)
        if returns_df is None:
            raise HTTPException(status_code=400,
                                detail="Could not fetch returns data")

        if optimization_type == 'max-sharpe':
            result = optimization_agent.optimize_max_sharpe(
                returns_df, max_weight=max_weight)
        elif optimization_type == 'min-variance':
            result = optimization_agent.optimize_min_variance(
                returns_df, max_weight=max_weight)
        else:
            raise HTTPException(status_code=400,
                                detail=f"Invalid type: {optimization_type}")

        result.update(symbols=symbols_list,
                      optimization_type=optimization_type,
                      period=period,
                      status='success')
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Alpha Signal Agent API on port 8002...")
    print("Swagger docs: http://localhost:8002/docs")
    print("\nEndpoints:")
    print("  GET  /signal/{symbol}")
    print("  GET  /backtest/{symbol}")
    print("  GET  /indicators/{symbol}")
    print("  POST /optimize/max-sharpe")
    print("  POST /optimize/min-variance")
    print("  POST /optimize/target-return")
    print("  POST /optimize/with-signals")
    print("  POST /efficient-frontier")
    uvicorn.run(app, host="0.0.0.0", port=8002)