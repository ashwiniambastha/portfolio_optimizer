import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataValidator:
    """
    Validates and cleans market data
    """
    
    @staticmethod
    def validate_realtime_data(data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate real-time data point
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required_fields = ['symbol', 'timestamp', 'price', 'volume']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing or null field: {field}")
        
        # Validate price is positive
        if 'price' in data and data['price'] is not None:
            if data['price'] <= 0:
                errors.append(f"Invalid price: {data['price']}")
        
        # Validate volume is non-negative
        if 'volume' in data and data['volume'] is not None:
            if data['volume'] < 0:
                errors.append(f"Invalid volume: {data['volume']}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean historical price data
        """
        # Remove rows with missing closing prices
        df = df.dropna(subset=['Close'])
        
        # Remove outliers (prices that change >50% in one day)
        df['price_change'] = df['Close'].pct_change()
        df = df[abs(df['price_change']) < 0.5]
        
        # Forward fill any remaining NaNs
        df = df.fillna(method='ffill')
        
        return df
    
    @staticmethod
    def calculate_data_quality_score(df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-100)
        """
        total_points = 100
        
        # Penalize for missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        total_points -= missing_pct * 30
        
        # Penalize for duplicate timestamps
        duplicate_pct = df.index.duplicated().sum() / len(df)
        total_points -= duplicate_pct * 20
        
        # Check for reasonable price movements
        if 'Close' in df.columns:
            daily_returns = df['Close'].pct_change().dropna()
            extreme_moves = (abs(daily_returns) > 0.2).sum()
            total_points -= (extreme_moves / len(daily_returns)) * 20
        
        return max(0, total_points)


# Demo
if __name__ == "__main__":
    # Test validation
    validator = DataValidator()
    
    # Good data
    good_data = {
        'symbol': 'AAPL',
        'timestamp': '2025-02-12T10:30:00',
        'price': 175.23,
        'volume': 1000000
    }
    
    is_valid, errors = validator.validate_realtime_data(good_data)
    print(f"Good data valid: {is_valid}, errors: {errors}")
    
    # Bad data
    bad_data = {
        'symbol': 'AAPL',
        'timestamp': None,
        'price': -10,
        'volume': 1000000
    }
    
    is_valid, errors = validator.validate_realtime_data(bad_data)
    print(f"Bad data valid: {is_valid}, errors: {errors}")