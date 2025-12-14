from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from core.portfolio import Trade as PortfolioTrade


@dataclass(frozen=True)
class Order:
    """전략이 생성하는 주문 정보."""

    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: float
    cost_pct: float = 0.0
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if self.side not in ("BUY", "SELL"):
            raise ValueError("side must be 'BUY' or 'SELL'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class Fill:
    """체결 결과."""

    date: datetime
    ticker: str
    side: str
    price: float
    quantity: float
    cost_pct: float


@dataclass
class SnapshotRecord:
    """백테스트 중 하루 단위 포트폴리오 자산 스냅샷."""

    date: datetime
    equity: float


TradeRecord = PortfolioTrade


@dataclass
class BacktestResult:
    """백테스트 결과 데이터 구조."""

    equity_curve: pd.Series
    benchmark_equity_curve: pd.Series | None
    trades: List[TradeRecord]
    latest_prices: Optional[Dict[str, float]] = None


