from __future__ import annotations

from datetime import date as date_type, datetime, timedelta
from typing import List, Sequence

import pandas as pd

from core.data import DataFetcher


class DataProvider:
    """백테스트/실거래 공통 데이터 제공 인터페이스."""

    def get_history(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def get_price(self, ticker: str, date: datetime) -> float | None:
        raise NotImplementedError

    def get_trading_days(self, start: datetime, end: datetime) -> List[datetime]:
        """거래일 캘린더 반환."""
        raise NotImplementedError


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date_type):
        return datetime.combine(value, datetime.min.time())
    raise TypeError(f"Unsupported date type: {type(value)!r}")


class BacktestDataProvider(DataProvider):
    """DataFetcher 를 이용하는 기본 백테스트용 DataProvider."""

    def __init__(
        self,
        universe: Sequence[str],
        benchmark: str | None = None,
        data_fetcher: DataFetcher | None = None,
    ) -> None:
        self.universe: tuple[str, ...] = tuple(universe)
        if not self.universe and not benchmark:
            raise ValueError("universe 또는 benchmark 중 하나는 최소 1개 이상이어야 합니다.")
        self.benchmark: str | None = benchmark
        self.data_fetcher: DataFetcher = data_fetcher or DataFetcher()
        self._history_cache: dict[str, pd.DataFrame] = {}

    def _ensure_history(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """요청 구간을 충분히 포함하는 히스토리를 캐시에서 보장."""
        cached = self._history_cache.get(ticker)
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end)

        fetch_start = start_dt
        fetch_end = end_dt
        if cached is not None and not cached.empty and "Date" in cached.columns:
            cached_min = pd.to_datetime(cached["Date"].min())
            cached_max = pd.to_datetime(cached["Date"].max())
            if cached_min <= start_dt and cached_max >= end_dt:
                return cached
            fetch_start = min(fetch_start, cached_min)
            fetch_end = max(fetch_end, cached_max)
        else:
            cached = None

        period_days = max(365, int((fetch_end - fetch_start).days) + 30)
        df = self.data_fetcher.get_history(
            ticker,
            period_days=period_days,
            start_date=fetch_start,
            end_date=fetch_end,
        )
        if "Date" in df.columns:
            df = df.copy()
            df["Date"] = pd.to_datetime(df["Date"])

        if cached is not None:
            df = (
                pd.concat([cached, df], ignore_index=True)
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
                .reset_index(drop=True)
            )

        self._history_cache[ticker] = df
        return df

    def get_history(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end)
        df = self._ensure_history(ticker, start_dt, end_dt)
        if "Date" not in df.columns or df.empty:
            return df
        mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt + timedelta(days=1))
        return df.loc[mask].reset_index(drop=True)

    def get_price(self, ticker: str, date: datetime) -> float | None:
        """해당 날짜의 종가를 반환 (없으면 None)."""
        date_dt = _to_datetime(date)
        df = self._ensure_history(ticker, date_dt - timedelta(days=5), date_dt)
        if "Date" not in df.columns or "Close" not in df.columns or df.empty:
            return None
        day = pd.to_datetime(date_dt.date())
        row = df.loc[df["Date"] == day]
        if row.empty:
            return None
        return float(row["Close"].iloc[0])

    def get_trading_days(self, start: datetime, end: datetime) -> List[datetime]:
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end)
        base_ticker = self.benchmark or (self.universe[0] if self.universe else None)
        if not base_ticker:
            raise RuntimeError("거래일 생성을 위한 기준 티커가 필요합니다.")
        df = self.get_history(base_ticker, start_dt, end_dt)
        if "Date" not in df.columns or df.empty:
            return []
        # FutureWarning 회피: DatetimeProperties.to_pydatetime() 사용을 피하고,
        # 개별 Timestamp 를 파이썬 datetime 으로 변환
        dt_series = pd.to_datetime(df["Date"])
        dates = sorted([ts.to_pydatetime() for ts in dt_series])
        return dates


