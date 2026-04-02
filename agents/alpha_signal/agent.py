"""
Portfolio Optimization Agent
Week 5: Modern Portfolio Theory (Markowitz)
Optimal asset allocation using mean-variance optimization
"""

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizationAgent:
    """
    Portfolio Optimization Agent
    Implements Modern Portfolio Theory (Markowitz)
    Finds optimal weights to maximize Sharpe ratio or minimize variance
    """
    
    def __init__(self,
                 market_data_api: str = "http://localhost:8000",
                 risk_agent_api: str = "http://localhost:8001",
                 alpha_agent_api: str = "http://localhost:8002"):
        """
        Initialize Portfolio Optimization Agent
        
        Args:
            market_data_api: Market Data Agent URL
            risk_agent_api: Risk Management Agent URL
            alpha_agent_api: Alpha Signal Agent URL
        """
        self.market_data_api = market_data_api
        self.risk_agent_api = risk_agent_api
        self.alpha_agent_api = alpha_agent_api
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
    
    def fetch_returns_data(self, symbols: List[str], period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch historical returns for multiple symbols
        
        Args:
            symbols: List of stock symbols
            period: Time period
        
        Returns:
            DataFrame with returns for each symbol (columns = symbols)
        """
        returns_dict = {}
        
        for symbol in symbols:
            try:
                response = requests.get(
                    f"{self.market_data_api}/historical/{symbol}",
                    params={"period": period},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    
                    # Calculate returns
                    returns = df['Close'].pct_change().dropna()
                    returns_dict[symbol] = returns
                else:
                    print(f"Error fetching {symbol}: {response.status_code}")
                    return None
                    
            except Exception as e:
                print(f"Exception fetching {symbol}: {e}")
                return None
        
        # Combine into single DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Drop rows with any NaN (align dates)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                    mean_returns: np.ndarray,
                                    cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Args:
            weights: Portfolio weights (must sum to 1)
            mean_returns: Expected returns for each asset
            cov_matrix: Covariance matrix of returns
        
        Returns:
            (portfolio_return, portfolio_volatility, sharpe_ratio)
        """
        # Annualized portfolio return
        portfolio_return = np.sum(weights * mean_returns) * 252
        
        # Annualized portfolio volatility
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix * 252, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(self, weights: np.ndarray,
                       mean_returns: np.ndarray,
                       cov_matrix: np.ndarray) -> float:
        """
        Negative Sharpe ratio (for minimization)
        
        Args:
            weights: Portfolio weights
            mean_returns: Expected returns
            cov_matrix: Covariance matrix
        
        Returns:
            Negative Sharpe ratio
        """
        _, _, sharpe = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        return -sharpe
    
    def portfolio_variance(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        Portfolio variance (for minimum variance optimization)
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
        
        Returns:
            Portfolio variance
        """
        return np.dot(weights.T, np.dot(cov_matrix * 252, weights))
    
    def optimize_max_sharpe(self, returns_df: pd.DataFrame,
                           max_weight: float = 0.4,
                           min_weight: float = 0.0) -> Dict:
        """
        Optimize portfolio to maximize Sharpe ratio
        
        Args:
            returns_df: DataFrame of returns
            max_weight: Maximum weight per asset (default 40%)
            min_weight: Minimum weight per asset (default 0%)
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns_df.columns)
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Initial guess (equal weights)
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between min_weight and max_weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Optimize
        result = minimize(
            self.negative_sharpe,
            init_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, mean_returns, cov_matrix
        )
        
        return {
            'weights': dict(zip(returns_df.columns, optimal_weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'optimization_success': result.success
        }
    
    def optimize_min_variance(self, returns_df: pd.DataFrame,
                             max_weight: float = 0.4,
                             min_weight: float = 0.0) -> Dict:
        """
        Optimize portfolio to minimize variance (risk)
        
        Args:
            returns_df: DataFrame of returns
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns_df.columns)
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Initial guess (equal weights)
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Optimize
        result = minimize(
            self.portfolio_variance,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, mean_returns, cov_matrix
        )
        
        return {
            'weights': dict(zip(returns_df.columns, optimal_weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'optimization_success': result.success
        }
    
    def optimize_target_return(self, returns_df: pd.DataFrame,
                               target_return: float,
                               max_weight: float = 0.4,
                               min_weight: float = 0.0) -> Dict:
        """
        Optimize portfolio to minimize variance given target return
        
        Args:
            returns_df: DataFrame of returns
            target_return: Target annualized return (e.g., 0.15 for 15%)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
        
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns_df.columns)
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Initial guess
        init_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Constraints: weights sum to 1 AND expected return >= target
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: np.sum(w * mean_returns) * 252 - target_return}
        ]
        
        # Bounds
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Optimize
        result = minimize(
            self.portfolio_variance,
            init_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            optimal_weights, mean_returns, cov_matrix
        )
        
        return {
            'weights': dict(zip(returns_df.columns, optimal_weights)),
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'target_return': target_return,
            'optimization_success': result.success
        }
    
    def generate_efficient_frontier(self, returns_df: pd.DataFrame,
                                    num_portfolios: int = 50,
                                    max_weight: float = 0.4) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios
        
        Args:
            returns_df: DataFrame of returns
            num_portfolios: Number of portfolios to generate
            max_weight: Maximum weight per asset
        
        Returns:
            DataFrame with returns, volatilities, Sharpe ratios, and weights
        """
        # Calculate mean returns and covariance matrix
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        
        # Get min and max possible returns
        min_return = np.min(mean_returns) * 252
        max_return = np.max(mean_returns) * 252
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        results = []
        
        for target_return in target_returns:
            try:
                result = self.optimize_target_return(
                    returns_df, 
                    target_return,
                    max_weight=max_weight
                )
                
                if result['optimization_success']:
                    row = {
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe': result['sharpe_ratio']
                    }
                    # Add weights
                    row.update(result['weights'])
                    results.append(row)
                    
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def optimize_with_signals(self, symbols: List[str],
                              portfolio_value: float = 100000,
                              max_weight: float = 0.4) -> Dict:
        """
        Optimize portfolio incorporating alpha signals
        Adjusts weights based on signal confidence
        
        Args:
            symbols: List of stock symbols
            portfolio_value: Total portfolio value
            max_weight: Maximum weight per asset
        
        Returns:
            Dictionary with optimized portfolio
        """
        # Fetch returns data
        returns_df = self.fetch_returns_data(symbols)
        if returns_df is None:
            return {'status': 'error', 'message': 'Could not fetch returns data'}
        
        # Get alpha signals for each symbol
        signals = {}
        for symbol in symbols:
            try:
                response = requests.get(
                    f"{self.alpha_agent_api}/signal/{symbol}",
                    params={"portfolio_value": portfolio_value},
                    timeout=10
                )
                
                if response.status_code == 200:
                    signal_data = response.json()
                    if signal_data['status'] == 'success':
                        signals[symbol] = signal_data['confidence'] / 100  # Normalize to [-1, 1]
                else:
                    signals[symbol] = 0  # Neutral if signal not available
                    
            except:
                signals[symbol] = 0
        
        # Optimize for max Sharpe
        base_result = self.optimize_max_sharpe(returns_df, max_weight=max_weight)
        
        # Adjust weights based on signals
        adjusted_weights = {}
        base_weights = base_result['weights']
        
        # Calculate adjustment factor
        for symbol in symbols:
            base_w = base_weights[symbol]
            signal = signals.get(symbol, 0)
            
            # Increase weight for positive signals, decrease for negative
            # Adjustment range: -20% to +20% of base weight
            adjustment = signal * 0.2 * base_w
            adjusted_weights[symbol] = max(0, min(max_weight, base_w + adjustment))
        
        # Normalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        # Recalculate metrics with adjusted weights
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values
        weights_array = np.array([adjusted_weights[s] for s in symbols])
        
        port_return, port_vol, sharpe = self.calculate_portfolio_metrics(
            weights_array, mean_returns, cov_matrix
        )
        
        return {
            'status': 'success',
            'base_weights': base_weights,
            'adjusted_weights': adjusted_weights,
            'signals': signals,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'portfolio_value': portfolio_value,
            'allocation_usd': {s: adjusted_weights[s] * portfolio_value for s in symbols}
        }


# Test the agent
if __name__ == "__main__":
    print("Testing Portfolio Optimization Agent\n" + "="*60)
    
    # Initialize agent
    agent = PortfolioOptimizationAgent()
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print(f"\n1. Fetching returns data for {symbols}...")
    returns_df = agent.fetch_returns_data(symbols, period="1y")
    
    if returns_df is not None:
        print(f"✓ Returns data fetched: {len(returns_df)} days")
        print(f"  Symbols: {list(returns_df.columns)}")
        
        # Test max Sharpe optimization
        print(f"\n2. Optimizing for Maximum Sharpe Ratio...")
        max_sharpe = agent.optimize_max_sharpe(returns_df, max_weight=0.4)
        
        print(f"✓ Optimization Complete")
        print(f"  Expected Return: {max_sharpe['expected_return']*100:.2f}%")
        print(f"  Volatility: {max_sharpe['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
        print(f"\n  Optimal Weights:")
        for symbol, weight in max_sharpe['weights'].items():
            print(f"    {symbol}: {weight*100:.2f}%")
        
        # Test min variance optimization
        print(f"\n3. Optimizing for Minimum Variance...")
        min_var = agent.optimize_min_variance(returns_df, max_weight=0.4)
        
        print(f"✓ Optimization Complete")
        print(f"  Expected Return: {min_var['expected_return']*100:.2f}%")
        print(f"  Volatility: {min_var['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {min_var['sharpe_ratio']:.3f}")
        print(f"\n  Optimal Weights:")
        for symbol, weight in min_var['weights'].items():
            print(f"    {symbol}: {weight*100:.2f}%")
        
        print(f"\n{'='*60}")
        print("✓ All tests completed!")
    else:
        print("✗ Could not fetch returns data")