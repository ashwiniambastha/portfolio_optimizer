"""
Technical Indicators Module
Week 4: Alpha Signal Agent
Calculates SMA, EMA, RSI, MACD, Bollinger Bands
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class TechnicalIndicators:
    """
    Technical Indicators Calculator
    All methods are static for easy reuse
    """
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average
        
        Formula: SMA_n(t) = (P_t + P_{t-1} + ... + P_{t-n+1}) / n
        
        Args:
            prices: Series of closing prices
            period: Number of periods (e.g., 20, 50, 200)
        
        Returns:
            Series of SMA values
        """
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average
        
        Formula: EMA_t = α × P_t + (1 - α) × EMA_{t-1}
        where α = 2 / (period + 1)
        
        Args:
            prices: Series of closing prices
            period: Number of periods
        
        Returns:
            Series of EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Formula: RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            prices: Series of closing prices
            period: Number of periods (default 14)
        
        Returns:
            Series of RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        
        Components:
        - MACD Line = EMA(fast) - EMA(slow)
        - Signal Line = EMA(signal) of MACD Line
        - Histogram = MACD Line - Signal Line
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
        
        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, 
                       num_std: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        
        Formula:
        - Middle Band = SMA(period)
        - Upper Band = SMA(period) + num_std × σ
        - Lower Band = SMA(period) - num_std × σ
        
        Args:
            prices: Series of closing prices
            period: Number of periods (default 20)
            num_std: Number of standard deviations (default 2)
        
        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'bandwidth'
        """
        # Middle band (SMA)
        middle = prices.rolling(window=period).mean()
        
        # Standard deviation
        std = prices.rolling(window=period).std()
        
        # Upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        
        # Bandwidth (volatility measure)
        bandwidth = (upper - lower) / middle
        
        # %B (position within bands)
        percent_b = (prices - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, 
                             close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        
        Formula: %K = (Close - Low_n) / (High_n - Low_n) × 100
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Number of periods (default 14)
        
        Returns:
            Dictionary with '%K' and '%D' (smoothed %K)
        """
        # Calculate %K
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()
        
        k_percent = 100 * (close - low_min) / (high_max - low_min)
        
        # Calculate %D (3-period SMA of %K)
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Average True Range (volatility measure)
        
        Formula: ATR = EMA(True Range, period)
        where True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Number of periods (default 14)
        
        Returns:
            Series of ATR values
        """
        # Previous close
        prev_close = close.shift(1)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        # True Range is maximum of three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is EMA of True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr


# Test the indicators
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Simulate price data with trend
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 2),
        index=dates
    )
    
    high = prices + np.random.rand(100) * 2
    low = prices - np.random.rand(100) * 2
    
    # Test all indicators
    indicators = TechnicalIndicators()
    
    print("Testing Technical Indicators\n" + "="*50)
    
    # SMA
    sma_20 = indicators.sma(prices, 20)
    print(f"\nSMA(20) - Last 5 values:")
    print(sma_20.tail())
    
    # EMA
    ema_12 = indicators.ema(prices, 12)
    print(f"\nEMA(12) - Last 5 values:")
    print(ema_12.tail())
    
    # RSI
    rsi = indicators.rsi(prices, 14)
    print(f"\nRSI(14) - Last 5 values:")
    print(rsi.tail())
    print(f"Current RSI: {rsi.iloc[-1]:.2f}")
    
    # MACD
    macd_data = indicators.macd(prices)
    print(f"\nMACD - Last value:")
    print(f"  MACD Line: {macd_data['macd'].iloc[-1]:.4f}")
    print(f"  Signal Line: {macd_data['signal'].iloc[-1]:.4f}")
    print(f"  Histogram: {macd_data['histogram'].iloc[-1]:.4f}")
    
    # Bollinger Bands
    bb = indicators.bollinger_bands(prices)
    print(f"\nBollinger Bands - Last value:")
    print(f"  Upper: {bb['upper'].iloc[-1]:.2f}")
    print(f"  Middle: {bb['middle'].iloc[-1]:.2f}")
    print(f"  Lower: {bb['lower'].iloc[-1]:.2f}")
    print(f"  Price: {prices.iloc[-1]:.2f}")
    
    # ATR
    atr = indicators.atr(high, low, prices)
    print(f"\nATR(14) - Last value: {atr.iloc[-1]:.2f}")
    
    print("\n✓ All indicators calculated successfully!")