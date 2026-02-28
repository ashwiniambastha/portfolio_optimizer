import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class RiskManagementAgent:
    """
    Risk Management Agent - Calculates portfolio risk metrics
    Integrates with Market Data Agent from Week 2
    """
    
    def __init__(self, market_data_api: str = "http://localhost:8000"):
        self.market_data_api = market_data_api
        self.risk_limits = {
            'var_95': 0.05,  # Max 5% VaR at 95% confidence
            'var_99': 0.10,  # Max 10% VaR at 99% confidence
            'volatility': 0.30,  # Max 30% annual volatility
            'max_drawdown': 0.20,  # Max 20% drawdown
            'min_sharpe': 1.0,  # Minimum Sharpe ratio
        }
    
    # ==================== DATA FETCHING ====================
    
    def fetch_historical_returns(self, symbol: str, period: str = "1y") -> pd.Series:
        """
        Fetch historical returns from Market Data Agent
        """
        try:
            response = requests.get(
                f"{self.market_data_api}/historical/{symbol}",
                params={"period": period}
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                # Handle different date formats
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                elif 'index' in df.columns:
                    df['Date'] = pd.to_datetime(df['index'])
                    df = df.set_index('Date')
                
                # Calculate daily returns
                returns = df['Close'].pct_change().dropna()
                
                print(f"✓ Fetched {len(returns)} days of returns for {symbol}")
                return returns
            else:
                print(f"✗ Error fetching data for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None
    
    def fetch_current_price(self, symbol: str) -> float:
        """
        Fetch current price from Market Data Agent
        """
        try:
            response = requests.get(f"{self.market_data_api}/price/{symbol}")
            if response.status_code == 200:
                data = response.json()
                return data['price']
            return None
        except Exception as e:
            print(f"✗ Error fetching price: {str(e)}")
            return None
    
    # ==================== VALUE AT RISK (VaR) ====================
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95, 
                     portfolio_value: float = 100000) -> Dict:
        """
        Calculate Value at Risk using Historical Simulation
        
        VaR answers: "What is the maximum loss we can expect with X% confidence?"
        
        Args:
            returns: Series of historical returns
            confidence: Confidence level (0.95 = 95%, 0.99 = 99%)
            portfolio_value: Current portfolio value in dollars
        
        Returns:
            Dictionary with VaR in percentage and dollar terms
        """
        if returns is None or len(returns) == 0:
            return None
        
        # VaR is the percentile of the loss distribution
        var_percentile = (1 - confidence) * 100
        var_return = np.percentile(returns, var_percentile)
        
        # Convert to positive loss
        var_pct = abs(var_return)
        var_dollar = var_pct * portfolio_value
        
        return {
            'confidence': confidence,
            'var_pct': var_pct,
            'var_dollar': var_dollar,
            'interpretation': f"We are {confidence*100:.0f}% confident we won't lose more than ${var_dollar:,.0f} ({var_pct:.2%}) in one day"
        }
    
    # ==================== CONDITIONAL VaR (CVaR) ====================
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95,
                      portfolio_value: float = 100000) -> Dict:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        CVaR answers: "If losses exceed VaR, what is the average loss?"
        
        Args:
            returns: Series of historical returns
            confidence: Confidence level
            portfolio_value: Current portfolio value
        
        Returns:
            Dictionary with CVaR metrics
        """
        if returns is None or len(returns) == 0:
            return None
        
        # First calculate VaR
        var_result = self.calculate_var(returns, confidence, portfolio_value)
        var_return = -var_result['var_pct']  # Negative for calculation
        
        # CVaR is the average of all returns worse than VaR
        tail_losses = returns[returns <= var_return]
        cvar_return = tail_losses.mean()
        
        cvar_pct = abs(cvar_return)
        cvar_dollar = cvar_pct * portfolio_value
        
        return {
            'confidence': confidence,
            'cvar_pct': cvar_pct,
            'cvar_dollar': cvar_dollar,
            'num_tail_events': len(tail_losses),
            'interpretation': f"If losses exceed VaR, average loss is ${cvar_dollar:,.0f} ({cvar_pct:.2%})"
        }
    
    # ==================== VOLATILITY ====================
    
    def calculate_volatility(self, returns: pd.Series, window: int = None) -> Dict:
        """
        Calculate volatility (standard deviation of returns)
        
        Args:
            returns: Series of historical returns
            window: Rolling window (None = full period, 30 = 30-day rolling)
        
        Returns:
            Dictionary with volatility metrics
        """
        if returns is None or len(returns) == 0:
            return None
        
        if window is None:
            # Full period volatility
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            return {
                'daily_volatility': daily_vol,
                'annual_volatility': annual_vol,
                'interpretation': f"Annual volatility: {annual_vol:.2%}"
            }
        else:
            # Rolling volatility
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            return {
                'rolling_volatility': rolling_vol,
                'current_volatility': rolling_vol.iloc[-1],
                'avg_volatility': rolling_vol.mean(),
                'max_volatility': rolling_vol.max(),
                'min_volatility': rolling_vol.min(),
                'window_days': window
            }
    
    # ==================== MAXIMUM DRAWDOWN ====================
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict:
        """
        Calculate Maximum Drawdown - largest peak to trough decline
        
        Max Drawdown answers: "What was the worst loss from a previous high?"
        
        Args:
            returns: Series of historical returns
        
        Returns:
            Dictionary with drawdown metrics
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown from peak
        drawdown = (cum_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find the peak before max drawdown
        peak_date = running_max[:max_dd_date].idxmax()
        
        # Calculate recovery time
        recovery_date = None
        if max_dd_date < drawdown.index[-1]:
            recovery = drawdown[max_dd_date:]
            recovery_points = recovery[recovery >= 0]
            if len(recovery_points) > 0:
                recovery_date = recovery_points.index[0]
        
        # Calculate days
        if isinstance(max_dd_date, (int, np.integer)):
            drawdown_days = abs(max_dd_date - peak_date)
        else:
            drawdown_days = (max_dd_date - peak_date).days
        if recovery_date:
            if isinstance(recovery_date, (int, np.integer)):
                recovery_days = abs(recovery_date - max_dd_date)
            else:
                recovery_days = (recovery_date - max_dd_date).days
        else:
            recovery_days = None
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd),
            'peak_date': str(peak_date) if isinstance(peak_date, (int, np.integer)) else peak_date.strftime('%Y-%m-%d'),
            'trough_date': str(max_dd_date) if isinstance(max_dd_date, (int, np.integer)) else max_dd_date.strftime('%Y-%m-%d'),
            'recovery_date': str(recovery_date) if recovery_date and isinstance(recovery_date, (int, np.integer)) else (recovery_date.strftime('%Y-%m-%d') if recovery_date else 'Not yet recovered'),
            'drawdown_days': drawdown_days,
            'recovery_days': recovery_days if recovery_days else 'Ongoing',
           'interpretation': f"Worst decline: {abs(max_dd):.2%} from {peak_date} to {trough_date}"
        }
    
    # ==================== SHARPE RATIO ====================
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.04) -> Dict:
        """
        Calculate Sharpe Ratio - risk-adjusted return metric
        
        Sharpe Ratio = (Return - Risk-Free Rate) / Volatility
        
        Interpretation:
        > 3.0: Exceptional
        2.0-3.0: Very Good
        1.0-2.0: Good
        0.5-1.0: Acceptable
        < 0.5: Poor
        < 0: Losing money
        
        Args:
            returns: Series of historical returns
            risk_free_rate: Annual risk-free rate (default 4% = 0.04)
        
        Returns:
            Dictionary with Sharpe ratio and interpretation
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate annualized return
        total_return = (1 + returns).prod() - 1
        n_days = len(returns)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate annualized volatility
        annualized_vol = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        
        # Interpretation
        if sharpe > 3.0:
            interpretation = "Exceptional"
        elif sharpe > 2.0:
            interpretation = "Very Good"
        elif sharpe > 1.0:
            interpretation = "Good"
        elif sharpe > 0.5:
            interpretation = "Acceptable"
        elif sharpe > 0:
            interpretation = "Poor"
        else:
            interpretation = "Losing Money"
        
        return {
            'sharpe_ratio': sharpe,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'risk_free_rate': risk_free_rate,
            'rating': interpretation,
            'interpretation': f"Sharpe Ratio: {sharpe:.2f} ({interpretation}) - Earning {sharpe:.2f} units of return per unit of risk"
        }
    
    # ==================== BETA CALCULATION ====================
    
    def calculate_beta(self, asset_returns: pd.Series, market_symbol: str = "SPY") -> Dict:
        """
        Calculate Beta - measures asset's volatility relative to market
        
        Beta > 1: More volatile than market (aggressive)
        Beta = 1: Moves with market
        Beta < 1: Less volatile than market (defensive)
        Beta < 0: Moves opposite to market
        
        Args:
            asset_returns: Returns of the asset
            market_symbol: Market benchmark (default: SPY for S&P 500)
        
        Returns:
            Dictionary with beta and interpretation
        """
        if asset_returns is None or len(asset_returns) == 0:
            return None
        
        # Fetch market returns
        print(f"  Fetching market returns ({market_symbol})...")
        market_returns = self.fetch_historical_returns(market_symbol, period="1y")
        
        if market_returns is None:
            return {'beta': None, 'error': 'Could not fetch market data'}
        
        # Align dates (only use common dates)
        combined = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(combined) < 20:
            return {'beta': None, 'error': 'Insufficient overlapping data'}
        
        # Calculate beta using covariance method
        # Beta = Cov(asset, market) / Var(market)
        covariance = combined['asset'].cov(combined['market'])
        market_variance = combined['market'].var()
        
        beta = covariance / market_variance
        
        # Calculate R-squared (correlation squared)
        correlation = combined['asset'].corr(combined['market'])
        r_squared = correlation ** 2
        
        # Interpretation
        if beta > 1.5:
            interpretation = "Highly Aggressive (Very volatile)"
        elif beta > 1.2:
            interpretation = "Aggressive (More volatile than market)"
        elif beta > 0.8:
            interpretation = "Moderate (Similar to market)"
        elif beta > 0.5:
            interpretation = "Defensive (Less volatile than market)"
        elif beta > 0:
            interpretation = "Very Defensive (Low volatility)"
        else:
            interpretation = "Inverse (Moves opposite to market)"
        
        return {
            'beta': beta,
            'r_squared': r_squared,
            'correlation': correlation,
            'market_benchmark': market_symbol,
            'interpretation': interpretation,
            'explanation': f"Beta: {beta:.2f} - {interpretation}. R²: {r_squared:.2%} (explained variance)"
        }
    
    # ==================== STRESS TESTING ====================
    
    def stress_test(self, returns: pd.Series, portfolio_value: float = 100000,
                   scenarios: Dict[str, float] = None) -> Dict:
        """
        Perform stress testing on portfolio
        
        Tests how portfolio performs under extreme market conditions
        
        Args:
            returns: Series of historical returns
            portfolio_value: Current portfolio value
            scenarios: Dict of scenario names and market shocks
                      e.g., {"Market Crash -20%": -0.20}
        
        Returns:
            Dictionary with stress test results
        """
        if returns is None or len(returns) == 0:
            return None
        
        # Default stress scenarios
        if scenarios is None:
            scenarios = {
                "Moderate Decline -5%": -0.05,
                "Correction -10%": -0.10,
                "Bear Market -20%": -0.20,
                "Severe Crash -30%": -0.30,
                "2008 Crisis -50%": -0.50,
                "Black Monday -20%": -0.20,
                "COVID Crash -35%": -0.35,
                "Flash Crash -10%": -0.10,
            }
        
        results = {}
        avg_daily_return = returns.mean()
        
        for scenario_name, shock in scenarios.items():
            # Calculate portfolio value after shock
            shocked_value = portfolio_value * (1 + shock)
            loss = portfolio_value - shocked_value
            
            # Estimate recovery time
            if avg_daily_return > 0:
                days_to_recover = abs(shock) / avg_daily_return
                years_to_recover = days_to_recover / 252
            else:
                days_to_recover = float('inf')
                years_to_recover = float('inf')
            
            results[scenario_name] = {
                'shock_pct': shock,
                'initial_value': portfolio_value,
                'shocked_value': shocked_value,
                'loss_amount': loss,
                'loss_pct': shock,
                'days_to_recover': int(days_to_recover) if days_to_recover != float('inf') else None,
                'years_to_recover': round(years_to_recover, 1) if years_to_recover != float('inf') else None
            }
        
        return results
    
    # ==================== CORRELATION MATRIX ====================
    
    def calculate_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Calculate correlation matrix between multiple assets
        
        Args:
            symbols: List of stock symbols
            period: Historical period
        
        Returns:
            Correlation matrix as DataFrame
        """
        returns_dict = {}
        
        print(f"\nFetching returns for correlation analysis...")
        for symbol in symbols:
            returns = self.fetch_historical_returns(symbol, period)
            if returns is not None:
                returns_dict[symbol] = returns
        
        if len(returns_dict) == 0:
            return None
        
        # Create DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        return corr_matrix
    
    # ==================== COMPREHENSIVE RISK ASSESSMENT ====================
    
    def assess_risk(self, symbol: str, portfolio_value: float = 100000) -> Dict:
        """
        Comprehensive risk assessment for a single asset
        
        Returns:
            Dictionary with all risk metrics
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE RISK ASSESSMENT: {symbol}")
        print(f"Portfolio Value: ${portfolio_value:,.0f}")
        print(f"{'='*70}")
        
        # Fetch returns
        returns = self.fetch_historical_returns(symbol, period="1y")
        
        if returns is None:
            return {'error': 'Could not fetch data'}
        
        print(f"\nCalculating risk metrics...")
        
        # Calculate all metrics
        var_95 = self.calculate_var(returns, 0.95, portfolio_value)
        var_99 = self.calculate_var(returns, 0.99, portfolio_value)
        cvar_95 = self.calculate_cvar(returns, 0.95, portfolio_value)
        cvar_99 = self.calculate_cvar(returns, 0.99, portfolio_value)
        volatility = self.calculate_volatility(returns)
        max_dd = self.calculate_max_drawdown(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        beta = self.calculate_beta(returns)
        stress_results = self.stress_test(returns, portfolio_value)
        
        # Check against limits and generate alerts
        alerts = []
        
        if var_95 and var_95['var_pct'] > self.risk_limits['var_95']:
            alerts.append(f"⚠️  VaR(95%) exceeds limit: {var_95['var_pct']:.2%} > {self.risk_limits['var_95']:.2%}")
        
        if var_99 and var_99['var_pct'] > self.risk_limits['var_99']:
            alerts.append(f"⚠️  VaR(99%) exceeds limit: {var_99['var_pct']:.2%} > {self.risk_limits['var_99']:.2%}")
        
        if volatility and volatility['annual_volatility'] > self.risk_limits['volatility']:
            alerts.append(f"⚠️  Volatility exceeds limit: {volatility['annual_volatility']:.2%} > {self.risk_limits['volatility']:.2%}")
        
        if max_dd and max_dd['max_drawdown'] > self.risk_limits['max_drawdown']:
            alerts.append(f"⚠️  Max Drawdown exceeds limit: {max_dd['max_drawdown']:.2%} > {self.risk_limits['max_drawdown']:.2%}")
        
        if sharpe and sharpe['sharpe_ratio'] < self.risk_limits['min_sharpe']:
            alerts.append(f"⚠️  Sharpe Ratio below minimum: {sharpe['sharpe_ratio']:.2f} < {self.risk_limits['min_sharpe']:.2f}")
        
        # Compile results
        assessment = {
            'symbol': symbol,
            'portfolio_value': portfolio_value,
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(returns),
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'beta': beta,
            'stress_test': stress_results,
            'alerts': alerts,
            'risk_status': 'ALERT' if len(alerts) > 0 else 'OK'
        }
        
        return assessment
    
    def print_risk_report(self, assessment: Dict):
        """
        Print formatted risk report
        """
        if 'error' in assessment:
            print(f"\n❌ Error: {assessment['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"RISK REPORT: {assessment['symbol']}")
        print(f"Date: {assessment['assessment_date']}")
        print(f"Portfolio Value: ${assessment['portfolio_value']:,.0f}")
        print(f"Data Points: {assessment['data_points']} trading days")
        print(f"{'='*70}")
        
        # VaR
        print(f"\n📊 VALUE AT RISK (VaR)")
        if assessment['var_95']:
            print(f"  95% Confidence: {assessment['var_95']['var_pct']:.2%} (${assessment['var_95']['var_dollar']:,.0f})")
        if assessment['var_99']:
            print(f"  99% Confidence: {assessment['var_99']['var_pct']:.2%} (${assessment['var_99']['var_dollar']:,.0f})")
        
        # CVaR
        print(f"\n📉 CONDITIONAL VaR (Expected Shortfall)")
        if assessment['cvar_95']:
            print(f"  95% Confidence: {assessment['cvar_95']['cvar_pct']:.2%} (${assessment['cvar_95']['cvar_dollar']:,.0f})")
            print(f"  Tail Events: {assessment['cvar_95']['num_tail_events']}")
        
        # Volatility
        print(f"\n📈 VOLATILITY")
        if assessment['volatility']:
            print(f"  Annual Volatility: {assessment['volatility']['annual_volatility']:.2%}")
            print(f"  Daily Volatility: {assessment['volatility']['daily_volatility']:.2%}")
        
        # Max Drawdown
        print(f"\n⬇️  MAXIMUM DRAWDOWN")
        if assessment['max_drawdown']:
            print(f"  Worst Decline: {assessment['max_drawdown']['max_drawdown']:.2%}")
            print(f"  Peak Date: {assessment['max_drawdown']['peak_date']}")
            print(f"  Trough Date: {assessment['max_drawdown']['trough_date']}")
            print(f"  Drawdown Period: {assessment['max_drawdown']['drawdown_days']} days")
            print(f"  Recovery: {assessment['max_drawdown']['recovery_date']}")
        
        # Sharpe Ratio
        print(f"\n⚡ SHARPE RATIO (Risk-Adjusted Return)")
        if assessment['sharpe_ratio']:
            print(f"  Sharpe Ratio: {assessment['sharpe_ratio']['sharpe_ratio']:.2f}")
            print(f"  Rating: {assessment['sharpe_ratio']['rating']}")
            print(f"  Annual Return: {assessment['sharpe_ratio']['annualized_return']:.2%}")
        
        # Beta
        print(f"\n📊 BETA (Market Sensitivity)")
        if assessment['beta'] and 'beta' in assessment['beta'] and assessment['beta']['beta'] is not None:
            print(f"  Beta: {assessment['beta']['beta']:.2f}")
            print(f"  Interpretation: {assessment['beta']['interpretation']}")
            print(f"  R-Squared: {assessment['beta']['r_squared']:.2%}")
        
        # Stress Test
        print(f"\n💥 STRESS TEST SCENARIOS")
        if assessment['stress_test']:
            for scenario, result in list(assessment['stress_test'].items())[:5]:  # Show top 5
                print(f"\n  {scenario}:")
                print(f"    Loss: ${result['loss_amount']:,.0f} ({result['loss_pct']:.2%})")
                if result['years_to_recover']:
                    print(f"    Recovery: ~{result['years_to_recover']} years")
        
        # Alerts
        print(f"\n{'='*70}")
        if assessment['alerts']:
            print(f"🚨 RISK ALERTS ({len(assessment['alerts'])})")
            for alert in assessment['alerts']:
                print(f"  {alert}")
            print(f"\nRisk Status: ⚠️  ALERT")
        else:
            print(f"✅ Risk Status: OK - All metrics within limits")
        print(f"{'='*70}\n")


# Demo/Test
if __name__ == "__main__":
    # Initialize Risk Agent
    risk_agent = RiskManagementAgent()
    
    # Test with Apple stock
    symbol = "AAPL"
    portfolio_value = 100000  # $100k portfolio
    
    print("\n" + "="*70)
    print("RISK MANAGEMENT AGENT - DEMO")
    print("="*70)
    
    # Comprehensive risk assessment
    assessment = risk_agent.assess_risk(symbol, portfolio_value)
    
    # Print formatted report
    risk_agent.print_risk_report(assessment)
    
    # Test correlation matrix
    print("\n" + "="*70)
    print("CORRELATION MATRIX")
    print("="*70)
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    corr_matrix = risk_agent.calculate_correlation_matrix(symbols)
    
    if corr_matrix is not None:
        print("\n", corr_matrix.round(2))
        print("\nInterpretation:")
        print("  1.0 = Perfect positive correlation")
        print("  0.0 = No correlation")
        print(" -1.0 = Perfect negative correlation")