"""
Portfolio Optimization Agent — Week 5
agents/portfolio_optimization/agent.py

Implements Modern Portfolio Theory (Markowitz Mean-Variance Optimization).

Strategies:
  1. Maximum Sharpe Ratio
  2. Minimum Variance
  3. Target Return
  4. Efficient Frontier (N optimal portfolios)
  5. Signal-Adjusted (integrates Week 4 alpha signals)
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Optional: Import SignalGenerator for signal-adjusted optimization.
# Falls back gracefully if the alpha_signal module is not on the path.
# ---------------------------------------------------------------------------
try:
    from agents.alpha_signal.signal_generator import SignalGenerator
    _SIGNAL_GEN_AVAILABLE = True
except ImportError:
    _SIGNAL_GEN_AVAILABLE = False


class PortfolioOptimizationAgent:
    """
    Portfolio Optimization Agent — Week 5
    Implements Modern Portfolio Theory (Markowitz Mean-Variance)

    Strategies:
      1. Maximum Sharpe Ratio
      2. Minimum Variance
      3. Target Return
      4. Efficient Frontier (50 optimal portfolios)
      5. Signal-Adjusted (integrates Week 4 alpha signals)
    """

    def __init__(
        self,
        market_data_api: str = "http://localhost:8000",
        risk_agent_api:  str = "http://localhost:8001",
        alpha_agent_api: str = "http://localhost:8002",
        risk_free_rate:  float = 0.04,
    ):
        self.market_data_api = market_data_api
        self.risk_agent_api  = risk_agent_api
        self.alpha_agent_api = alpha_agent_api
        self.risk_free_rate  = risk_free_rate   # 4% annual
        self._sig_gen        = SignalGenerator() if _SIGNAL_GEN_AVAILABLE else None

    # ── Data Fetching ────────────────────────────────────────────────────────

    def fetch_returns_data(
        self,
        symbols: List[str],
        period: str = "1y",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily returns for all symbols.

        Tries the Market Data REST API first; falls back to direct yfinance.
        Returns a DataFrame with columns = symbols and rows = dates, or None
        if data cannot be obtained.
        """
        returns_dict: Dict[str, pd.Series] = {}

        for sym in symbols:
            fetched = False

            # Try Market Data API
            try:
                r = requests.get(
                    f"{self.market_data_api}/historical/{sym}",
                    params={"period": period},
                    timeout=5,
                )
                if r.status_code == 200:
                    df = pd.DataFrame(r.json())
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")
                    returns_dict[sym] = df["Close"].pct_change().dropna()
                    fetched = True
            except Exception:
                pass

            # Fallback: direct yfinance
            if not fetched:
                try:
                    df = yf.Ticker(sym).history(period=period)
                    returns_dict[sym] = df["Close"].pct_change().dropna()
                except Exception as exc:
                    print(f"  Could not fetch {sym}: {exc}")
                    return None

        df_returns = pd.DataFrame(returns_dict).dropna()
        print(
            f"  ✓ Returns fetched: {len(df_returns)} days × {len(symbols)} symbols"
        )
        return df_returns

    # ── Core Portfolio Metrics ───────────────────────────────────────────────

    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute annualised portfolio return, volatility, and Sharpe ratio.

        Formulae:
            R_p    = w^T · μ · 252
            σ²_p   = w^T · Σ_annual · w
            Sharpe = (R_p - R_f) / σ_p
        """
        port_return = float(np.sum(weights * mean_returns) * 252)
        port_var    = float(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        port_vol    = float(np.sqrt(port_var))
        sharpe      = (
            (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0
        )
        return port_return, port_vol, sharpe

    def _negative_sharpe(
        self,
        weights: np.ndarray,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective: minimise −Sharpe  ≡  maximise Sharpe."""
        _, _, sharpe = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        return -sharpe

    def _portfolio_variance(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """Objective: minimise annualised portfolio variance."""
        return float(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    def _build_constraints_bounds(
        self,
        n: int,
        max_w: float,
        min_w: float,
        extra: Optional[List] = None,
    ) -> Tuple[List, tuple]:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        if extra:
            constraints.extend(extra)
        bounds = tuple((min_w, max_w) for _ in range(n))
        return constraints, bounds

    # ── Strategy 1: Maximum Sharpe ──────────────────────────────────────────

    def optimize_max_sharpe(
        self,
        returns_df: pd.DataFrame,
        max_weight: float = 0.4,
        min_weight: float = 0.0,
    ) -> Dict:
        """
        Find weights that MAXIMISE the Sharpe ratio.

        Solver: SciPy SLSQP
        Constraints:
          Σ w_i = 1
          min_weight ≤ w_i ≤ max_weight
        """
        n            = len(returns_df.columns)
        mean_ret     = returns_df.mean().values
        cov          = returns_df.cov().values
        init_w       = np.full(n, 1.0 / n)
        cons, bounds = self._build_constraints_bounds(n, max_weight, min_weight)

        result = minimize(
            self._negative_sharpe,
            init_w,
            args=(mean_ret, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        w = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(w, mean_ret, cov)

        return {
            "weights":               dict(zip(returns_df.columns, w)),
            "expected_return":       ret,
            "volatility":            vol,
            "sharpe_ratio":          sharpe,
            "optimization_success":  result.success,
            "strategy":              "Maximum Sharpe Ratio",
        }

    # ── Strategy 2: Minimum Variance ────────────────────────────────────────

    def optimize_min_variance(
        self,
        returns_df: pd.DataFrame,
        max_weight: float = 0.4,
        min_weight: float = 0.0,
    ) -> Dict:
        """
        Find weights that MINIMISE portfolio variance.
        Suitable for risk-averse / conservative investors.
        """
        n            = len(returns_df.columns)
        mean_ret     = returns_df.mean().values
        cov          = returns_df.cov().values
        init_w       = np.full(n, 1.0 / n)
        cons, bounds = self._build_constraints_bounds(n, max_weight, min_weight)

        result = minimize(
            self._portfolio_variance,
            init_w,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        w = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(w, mean_ret, cov)

        return {
            "weights":               dict(zip(returns_df.columns, w)),
            "expected_return":       ret,
            "volatility":            vol,
            "sharpe_ratio":          sharpe,
            "optimization_success":  result.success,
            "strategy":              "Minimum Variance",
        }

    # ── Strategy 3: Target Return ────────────────────────────────────────────

    def optimize_target_return(
        self,
        returns_df: pd.DataFrame,
        target_return: float,
        max_weight: float = 0.4,
        min_weight: float = 0.0,
    ) -> Dict:
        """
        Minimise variance subject to expected return ≥ target_return.

        Additional constraint:
          w^T · μ · 252 ≥ target_return
        """
        n        = len(returns_df.columns)
        mean_ret = returns_df.mean().values
        cov      = returns_df.cov().values
        init_w   = np.full(n, 1.0 / n)

        extra = [
            {
                "type": "ineq",
                "fun": lambda w: np.sum(w * mean_ret) * 252 - target_return,
            }
        ]
        cons, bounds = self._build_constraints_bounds(n, max_weight, min_weight, extra)

        result = minimize(
            self._portfolio_variance,
            init_w,
            args=(cov,),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        w = result.x
        ret, vol, sharpe = self.calculate_portfolio_metrics(w, mean_ret, cov)

        return {
            "weights":               dict(zip(returns_df.columns, w)),
            "expected_return":       ret,
            "volatility":            vol,
            "sharpe_ratio":          sharpe,
            "target_return":         target_return,
            "optimization_success":  result.success,
            "strategy":              f"Target Return {target_return:.1%}",
        }

    # ── Strategy 4: Efficient Frontier ──────────────────────────────────────

    def generate_efficient_frontier(
        self,
        returns_df: pd.DataFrame,
        num_portfolios: int = 50,
        max_weight: float = 0.4,
    ) -> pd.DataFrame:
        """
        Compute `num_portfolios` optimal portfolios that span the range of
        achievable annualised returns — the Markowitz Efficient Frontier.

        Returns a DataFrame with columns: return, volatility, sharpe,
        plus one column per symbol containing its weight.
        """
        mean_ret    = returns_df.mean().values
        min_ret     = mean_ret.min() * 252
        max_ret     = mean_ret.max() * 252
        target_rets = np.linspace(min_ret, max_ret, num_portfolios)
        rows: List[Dict] = []

        print(
            f"  Building efficient frontier ({num_portfolios} portfolios) ...",
            end="",
            flush=True,
        )
        for i, tr in enumerate(target_rets):
            try:
                res = self.optimize_target_return(returns_df, tr, max_weight)
                if res["optimization_success"]:
                    row = {
                        "return":     res["expected_return"],
                        "volatility": res["volatility"],
                        "sharpe":     res["sharpe_ratio"],
                    }
                    row.update(res["weights"])
                    rows.append(row)
            except Exception:
                pass
            if i % 10 == 9:
                print(".", end="", flush=True)

        print(f" done ({len(rows)} points)")
        return pd.DataFrame(rows)

    # ── Strategy 5: Signal-Adjusted ─────────────────────────────────────────

    def optimize_with_signals(
        self,
        symbols: List[str],
        portfolio_value: float = 100_000,
        max_weight: float = 0.4,
    ) -> Dict:
        """
        Optimize portfolio incorporating Week 4 alpha signals.

        Algorithm:
          1. Compute base Max-Sharpe weights.
          2. For each symbol fetch signal confidence from the Alpha Signal API
             (or generate directly via SignalGenerator as a fallback).
          3. Adjust base weights by ±20 % scaled by confidence.
          4. Re-normalise so weights sum to 1.
          5. Compute adjusted portfolio metrics.
        """
        returns_df = self.fetch_returns_data(symbols)
        if returns_df is None:
            return {"status": "error", "message": "Could not fetch returns"}

        # ── Collect signals ───────────────────────────────────────────────
        signals: Dict[str, float] = {}
        for sym in symbols:
            fetched = False
            try:
                r = requests.get(
                    f"{self.alpha_agent_api}/signal/{sym}",
                    params={"portfolio_value": portfolio_value},
                    timeout=5,
                )
                if r.status_code == 200:
                    sd = r.json()
                    signals[sym] = sd.get("confidence", 0) / 100
                    fetched = True
            except Exception:
                pass

            if not fetched:
                # Direct fallback via SignalGenerator
                if self._sig_gen is not None:
                    try:
                        df_sym   = yf.Ticker(sym).history(period="6mo")
                        sg       = self._sig_gen.generate_signal(df_sym["Close"])
                        signals[sym] = sg["confidence"] / 100
                    except Exception:
                        signals[sym] = 0.0
                else:
                    signals[sym] = 0.0

        # ── Base optimisation ─────────────────────────────────────────────
        base = self.optimize_max_sharpe(returns_df, max_weight)

        # ── Adjust weights ────────────────────────────────────────────────
        adj_w: Dict[str, float] = {}
        for sym in symbols:
            bw       = base["weights"][sym]
            adj      = signals.get(sym, 0) * 0.2 * bw   # ±20 % adjustment
            adj_w[sym] = max(0.0, min(max_weight, bw + adj))

        total  = sum(adj_w.values())
        adj_w  = {k: v / total for k, v in adj_w.items()}

        mean_ret       = returns_df.mean().values
        cov            = returns_df.cov().values
        w_arr          = np.array([adj_w[s] for s in symbols])
        ret, vol, sharpe = self.calculate_portfolio_metrics(w_arr, mean_ret, cov)

        return {
            "status":            "success",
            "strategy":          "Signal-Adjusted Max Sharpe",
            "base_weights":      base["weights"],
            "adjusted_weights":  adj_w,
            "signals":           signals,
            "expected_return":   ret,
            "volatility":        vol,
            "sharpe_ratio":      sharpe,
            "portfolio_value":   portfolio_value,
            "allocation_usd":    {s: adj_w[s] * portfolio_value for s in symbols},
        }