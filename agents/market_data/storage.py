import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict
import threading

class MarketDataStorage:
    """
    Thread-safe SQLite storage for market data
    """
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.local = threading.local()  # Thread-local storage
        self._create_tables()
    
    def _get_conn(self):
        """Get thread-local database connection"""
        if not hasattr(self.local, 'conn'):
            # Create new connection for this thread
            self.local.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False  # Allow multi-threading
            )
        return self.local.conn
    
    def _create_tables(self):
        """Create database tables"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Real-time prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS realtime_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                market_cap REAL,
                pe_ratio REAL
            )
        """)
        
        # Historical prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume INTEGER,
                UNIQUE(symbol, date)
            )
        """)
        
        conn.commit()
    
    def save_realtime_data(self, data: Dict):
        """Save real-time data point (thread-safe)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO realtime_prices 
                (symbol, timestamp, price, open, high, low, volume, market_cap, pe_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['symbol'],
                data['timestamp'],
                data['price'],
                data.get('open'),
                data.get('high'),
                data.get('low'),
                data.get('volume'),
                data.get('market_cap'),
                data.get('pe_ratio')
            ))
            conn.commit()
            print(f"✓ Saved {data['symbol']} to database")
        except Exception as e:
            print(f"✗ Error saving data: {e}")
            conn.rollback()
    
    def save_historical_data(self, symbol: str, df: pd.DataFrame):
        """Save historical data (thread-safe)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            for date, row in df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO historical_prices
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            
            conn.commit()
            print(f"✓ Saved {len(df)} historical records for {symbol}")
        except Exception as e:
            print(f"✗ Error saving historical data: {e}")
            conn.rollback()
    
    def get_latest_prices(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get latest prices for symbols (thread-safe)"""
        conn = self._get_conn()
        
        query = """
            SELECT symbol, price, timestamp
            FROM realtime_prices
            WHERE id IN (
                SELECT MAX(id)
                FROM realtime_prices
                GROUP BY symbol
            )
        """
        
        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            query += f" AND symbol IN ({placeholders})"
            df = pd.read_sql_query(query, conn, params=symbols)
        else:
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_historical_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get historical data for a symbol (thread-safe)"""
        conn = self._get_conn()
        
        query = """
            SELECT date, open, high, low, close, volume
            FROM historical_prices
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        return df
    
    def close(self):
        """Close database connection"""
        if hasattr(self.local, 'conn'):
            self.local.conn.close()


# Test the storage
if __name__ == "__main__":
    storage = MarketDataStorage()
    
    # Test data
    test_data = {
        'symbol': 'AAPL',
        'timestamp': datetime.now().isoformat(),
        'price': 175.23,
        'open': 174.50,
        'high': 176.00,
        'low': 173.80,
        'volume': 50000000,
        'market_cap': 2800000000000,
        'pe_ratio': 28.5
    }
    
    print("Testing thread-safe storage...")
    storage.save_realtime_data(test_data)
    
    # Retrieve
    latest = storage.get_latest_prices(['AAPL'])
    print("\nLatest prices:")
    print(latest)