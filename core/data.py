import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from core.data_source import DataSource, YFinanceDataSource


class DataFetcher:
    """Data fetcher for stock price history"""
    
    def __init__(self, data_source: Optional[DataSource] = None) -> None:
        """
        Initialize DataFetcher
        
        Args:
            data_source: Data source implementation (default: YFinanceDataSource)
        """
        self.data_source = data_source if data_source is not None else YFinanceDataSource()

    def get_history(
        self,
        ticker: str,
        period_days: int | None = 365,
        *,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch stock price history for the specified period
        
        Args:
            ticker: Stock ticker symbol
            period_days: Number of days to fetch
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
        """
        if end_date is None:
            end = datetime.now()
        else:
            end = end_date
        if start_date is None:
            days = period_days if period_days is not None else 365
            start = end - timedelta(days=days)
        else:
            start = start_date
        
        start_ts = pd.to_datetime(start.date())
        end_ts = pd.to_datetime(end.date())

        # API에서 직접 데이터 가져오기
        result = self.data_source.fetch_history(ticker, start_ts, end_ts)

        return result[
            (result['Date'] >= start_ts) & (result['Date'] <= end_ts)
        ].reset_index(drop=True)
    
    def get_mansfield_rs(
        self,
        df_asset: pd.DataFrame,
        df_benchmark: pd.DataFrame,
        window: int = 52,
    ) -> pd.Series:
        """
        Mansfield Relative Strength 계산
        자산의 벤치마크 대비 상대 강도를 측정
        
        Args:
            df_asset: 자산의 OHLCV 데이터프레임
            df_benchmark: 벤치마크의 OHLCV 데이터프레임
            window: 이동평균 기간 (기본: 52주)
            
        Returns:
            Mansfield RS 시리즈 (0 이상: 벤치마크 대비 강함, 0 미만: 약함)
        """
        a = df_asset[['Date', 'Close']].rename(columns={'Close': 'A'})
        b = df_benchmark[['Date', 'Close']].rename(columns={'Close': 'B'})
        m = pd.merge(a, b, on='Date', how='inner').set_index('Date')
        rs = (m['A'] / m['B']).astype(float)
        base = rs.rolling(window=window, min_periods=1).mean()
        mansfield = rs / base - 1.0
        return mansfield

