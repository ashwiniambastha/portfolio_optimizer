"""
Signal Generator Module
Week 4: Alpha Signal Agent
Combines multiple technical indicators to generate trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

try:
    from week4_indicators import TechnicalIndicators
except:
    from indicators import TechnicalIndicators


class SignalGenerator:
    """
    Generates trading signals by combining multiple technical indicators
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        
        # Signal weights (must sum to 1.0)
        self.weights = {
            'ma_crossover': 0.25,
            'rsi': 0.25,
            'macd': 0.30,
            'bollinger': 0.20
        }
    
    def _ma_crossover_signal(self, prices: pd.Series) -> Tuple[int, str]:
        """
        Moving Average Crossover Signal
        
        Returns:
            (signal, explanation)
            signal: 1 (buy), 0 (hold), -1 (sell)
        """
        sma_20 = self.indicators.sma(prices, 20)
        sma_50 = self.indicators.sma(prices, 50)
        
        current_price = prices.iloc[-1]
        sma_20_val = sma_20.iloc[-1]
        sma_50_val = sma_50.iloc[-1]
        
        if pd.isna(sma_20_val) or pd.isna(sma_50_val):
            return 0, "Insufficient data"
        
        # Strong buy: Price > SMA20 > SMA50
        if current_price > sma_20_val > sma_50_val:
            return 1, f"Bullish: Price ({current_price:.2f}) > SMA20 ({sma_20_val:.2f}) > SMA50 ({sma_50_val:.2f})"
        
        # Strong sell: Price < SMA20 < SMA50
        elif current_price < sma_20_val < sma_50_val:
            return -1, f"Bearish: Price ({current_price:.2f}) < SMA20 ({sma_20_val:.2f}) < SMA50 ({sma_50_val:.2f})"
        
        # Hold: Mixed signals
        else:
            return 0, "Mixed moving average signals"
    
    def _rsi_signal(self, prices: pd.Series) -> Tuple[int, str]:
        """
        RSI Signal
        
        Returns:
            (signal, explanation)
        """
        rsi = self.indicators.rsi(prices, 14)
        rsi_val = rsi.iloc[-1]
        
        if pd.isna(rsi_val):
            return 0, "Insufficient data"
        
        # Oversold - Buy signal
        if rsi_val < 30:
            return 1, f"Oversold: RSI = {rsi_val:.1f} < 30"
        
        # Overbought - Sell signal
        elif rsi_val > 70:
            return -1, f"Overbought: RSI = {rsi_val:.1f} > 70"
        
        # Slightly bullish
        elif 30 <= rsi_val < 50:
            return 0.5, f"Slightly oversold: RSI = {rsi_val:.1f}"
        
        # Slightly bearish
        elif 50 < rsi_val <= 70:
            return -0.5, f"Slightly overbought: RSI = {rsi_val:.1f}"
        
        # Neutral
        else:
            return 0, f"Neutral: RSI = {rsi_val:.1f}"
    
    def _macd_signal(self, prices: pd.Series) -> Tuple[int, str]:
        """
        MACD Signal
        
        Returns:
            (signal, explanation)
        """
        macd_data = self.indicators.macd(prices)
        
        macd_val = macd_data['macd'].iloc[-1]
        signal_val = macd_data['signal'].iloc[-1]
        histogram = macd_data['histogram'].iloc[-1]
        
        if pd.isna(macd_val) or pd.isna(signal_val):
            return 0, "Insufficient data"
        
        # Bullish crossover
        if macd_val > signal_val and histogram > 0:
            strength = "Strong" if abs(histogram) > 0.5 else "Weak"
            return 1, f"{strength} bullish: MACD ({macd_val:.3f}) > Signal ({signal_val:.3f})"
        
        # Bearish crossover
        elif macd_val < signal_val and histogram < 0:
            strength = "Strong" if abs(histogram) > 0.5 else "Weak"
            return -1, f"{strength} bearish: MACD ({macd_val:.3f}) < Signal ({signal_val:.3f})"
        
        # Neutral
        else:
            return 0, f"Neutral: MACD ≈ Signal"
    
    def _bollinger_signal(self, prices: pd.Series) -> Tuple[int, str]:
        """
        Bollinger Bands Signal
        
        Returns:
            (signal, explanation)
        """
        bb = self.indicators.bollinger_bands(prices, 20, 2.0)
        
        current_price = prices.iloc[-1]
        upper = bb['upper'].iloc[-1]
        middle = bb['middle'].iloc[-1]
        lower = bb['lower'].iloc[-1]
        percent_b = bb['percent_b'].iloc[-1]
        
        if pd.isna(upper) or pd.isna(lower):
            return 0, "Insufficient data"
        
        # Price below lower band - Oversold
        if current_price < lower:
            deviation = ((current_price - lower) / lower) * 100
            return 1, f"Oversold: Price {deviation:.1f}% below lower band"
        
        # Price above upper band - Overbought
        elif current_price > upper:
            deviation = ((current_price - upper) / upper) * 100
            return -1, f"Overbought: Price {deviation:.1f}% above upper band"
        
        # Price near lower band (within 5%)
        elif percent_b < 0.2:
            return 0.5, f"Near lower band: %B = {percent_b:.2f}"
        
        # Price near upper band (within 5%)
        elif percent_b > 0.8:
            return -0.5, f"Near upper band: %B = {percent_b:.2f}"
        
        # Price in middle - Neutral
        else:
            return 0, f"Mid-range: %B = {percent_b:.2f}"
    
    def generate_signal(self, prices: pd.Series) -> Dict:
        """
        Generate comprehensive trading signal
        
        Args:
            prices: Series of closing prices
        
        Returns:
            Dictionary with signals, confidence, and recommendation
        """
        # Get individual signals
        ma_signal, ma_explain = self._ma_crossover_signal(prices)
        rsi_signal, rsi_explain = self._rsi_signal(prices)
        macd_signal, macd_explain = self._macd_signal(prices)
        bb_signal, bb_explain = self._bollinger_signal(prices)
        
        # Store signals
        signals = {
            'ma_crossover': ma_signal,
            'rsi': rsi_signal,
            'macd': macd_signal,
            'bollinger': bb_signal
        }
        
        explanations = {
            'ma_crossover': ma_explain,
            'rsi': rsi_explain,
            'macd': macd_explain,
            'bollinger': bb_explain
        }
        
        # Calculate weighted confidence score (-100 to +100)
        confidence = sum(
            signals[key] * self.weights[key] 
            for key in self.weights.keys()
        ) * 100
        
        # Determine recommendation
        if confidence > 60:
            recommendation = "STRONG BUY"
            action = "BUY"
        elif confidence > 20:
            recommendation = "BUY"
            action = "BUY"
        elif confidence > -20:
            recommendation = "HOLD"
            action = "HOLD"
        elif confidence > -60:
            recommendation = "SELL"
            action = "SELL"
        else:
            recommendation = "STRONG SELL"
            action = "SELL"
        
        # Calculate indicator values
        rsi_val = self.indicators.rsi(prices, 14).iloc[-1]
        macd_data = self.indicators.macd(prices)
        bb_data = self.indicators.bollinger_bands(prices)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_price': prices.iloc[-1],
            'recommendation': recommendation,
            'action': action,
            'confidence': confidence,
            'signals': signals,
            'explanations': explanations,
            'indicator_values': {
                'rsi': rsi_val if not pd.isna(rsi_val) else None,
                'macd': macd_data['macd'].iloc[-1] if not pd.isna(macd_data['macd'].iloc[-1]) else None,
                'macd_signal': macd_data['signal'].iloc[-1] if not pd.isna(macd_data['signal'].iloc[-1]) else None,
                'macd_histogram': macd_data['histogram'].iloc[-1] if not pd.isna(macd_data['histogram'].iloc[-1]) else None,
                'bb_upper': bb_data['upper'].iloc[-1] if not pd.isna(bb_data['upper'].iloc[-1]) else None,
                'bb_middle': bb_data['middle'].iloc[-1] if not pd.isna(bb_data['middle'].iloc[-1]) else None,
                'bb_lower': bb_data['lower'].iloc[-1] if not pd.isna(bb_data['lower'].iloc[-1]) else None,
                'bb_percent_b': bb_data['percent_b'].iloc[-1] if not pd.isna(bb_data['percent_b'].iloc[-1]) else None,
            }
        }
    
    def backtest_strategy(self, prices: pd.Series, initial_capital: float = 10000) -> Dict:
        """
        Simple backtest of the strategy
        
        Args:
            prices: Series of closing prices
            initial_capital: Starting capital
        
        Returns:
            Dictionary with backtest results
        """
        capital = initial_capital
        position = 0  # Number of shares
        trades = []
        equity_curve = []
        
        # Need at least 50 days for indicators
        if len(prices) < 50:
            return {'error': 'Insufficient data for backtest'}
        
        # Backtest day by day
        for i in range(50, len(prices)):
            # Get signal using data up to current day
            current_prices = prices.iloc[:i+1]
            signal_data = self.generate_signal(current_prices)
            
            current_price = prices.iloc[i]
            action = signal_data['action']
            
            # Execute trades
            if action == "BUY" and position == 0 and capital > current_price:
                # Buy as many shares as possible
                shares_to_buy = int(capital / current_price)
                cost = shares_to_buy * current_price
                
                if shares_to_buy > 0:
                    position = shares_to_buy
                    capital -= cost
                    trades.append({
                        'date': prices.index[i],
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'value': cost
                    })
            
            elif action == "SELL" and position > 0:
                # Sell all shares
                proceeds = position * current_price
                capital += proceeds
                
                trades.append({
                    'date': prices.index[i],
                    'action': 'SELL',
                    'shares': position,
                    'price': current_price,
                    'value': proceeds
                })
                
                position = 0
            
            # Track equity
            portfolio_value = capital + (position * current_price)
            equity_curve.append({
                'date': prices.index[i],
                'value': portfolio_value
            })
        
        # Final portfolio value
        final_price = prices.iloc[-1]
        final_value = capital + (position * final_price)
        
        # Calculate metrics
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        num_trades = len(trades)
        
        # Buy and hold comparison
        buy_hold_shares = initial_capital / prices.iloc[50]
        buy_hold_value = buy_hold_shares * final_price
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'num_trades': num_trades,
            'trades': trades,
            'equity_curve': equity_curve,
            'buy_hold_return_pct': buy_hold_return,
            'outperformance': total_return - buy_hold_return
        }


# Test the signal generator
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Simulate price data with trend
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(200) * 2),
        index=dates
    )
    
    # Test signal generation
    signal_gen = SignalGenerator()
    
    print("Testing Signal Generator\n" + "="*60)
    
    # Generate signal
    signal = signal_gen.generate_signal(prices)
    
    print(f"\nCurrent Price: ${signal['current_price']:.2f}")
    print(f"Recommendation: {signal['recommendation']}")
    print(f"Confidence: {signal['confidence']:.1f}%")
    print(f"Action: {signal['action']}")
    
    print(f"\nIndividual Signals:")
    for indicator, value in signal['signals'].items():
        print(f"  {indicator}: {value:+.2f} - {signal['explanations'][indicator]}")
    
    print(f"\nIndicator Values:")
    for indicator, value in signal['indicator_values'].items():
        if value is not None:
            print(f"  {indicator}: {value:.4f}")
    
    # Test backtest
    print(f"\n{'='*60}")
    print("Testing Backtest\n")
    
    backtest = signal_gen.backtest_strategy(prices, initial_capital=10000)
    
    if 'error' not in backtest:
        print(f"Initial Capital: ${backtest['initial_capital']:,.2f}")
        print(f"Final Value: ${backtest['final_value']:,.2f}")
        print(f"Total Return: {backtest['total_return_pct']:.2f}%")
        print(f"Number of Trades: {backtest['num_trades']}")
        print(f"Buy & Hold Return: {backtest['buy_hold_return_pct']:.2f}%")
        print(f"Outperformance: {backtest['outperformance']:+.2f}%")
    
    print("\n✓ Signal generator tested successfully!")