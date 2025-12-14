from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta


class DataSource(ABC):
    """Abstract base class for stock data sources"""
    
    @abstractmethod
    def fetch_history(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock price history from the data source
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
        """
        pass


class YFinanceDataSource(DataSource):
    """YFinance data source implementation"""
    
    def fetch_history(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock price history from YFinance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
        """
        import yfinance as yf
        
        df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), 
                        auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Date','Open','High','Low','Close','Adj Close','Volume'])
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']
        
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        keep = [c for c in ['Date','Open','High','Low','Close','Adj Close','Volume'] if c in df.columns]
        return df[keep]

