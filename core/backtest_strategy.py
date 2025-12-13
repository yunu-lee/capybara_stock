from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence

from core.portfolio import Portfolio

from .backtest_data import DataProvider
from .backtest_types import Fill, Order


class BaseStrategy:
    """백테스트/실거래에서 공통으로 사용하는 전략 베이스 클래스."""

    def __init__(
        self,
        universe: Sequence[str],
        data_provider: DataProvider,
        **params: object,
    ) -> None:
        self.universe: tuple[str, ...] = tuple(universe)
        self.data_provider = data_provider
        self.params = params

    def on_start(self, portfolio: Portfolio) -> None:  # pragma: no cover - 기본 구현
        """백테스트 시작 시 1회 호출."""

    def on_end(self, portfolio: Portfolio) -> None:  # pragma: no cover - 기본 구현
        """백테스트 종료 시 1회 호출."""

    def on_bar(self, date: datetime, portfolio: Portfolio) -> List[Order]:  # pragma: no cover - 추상 역할
        """각 거래일마다 호출되어 주문 리스트를 반환한다."""
        raise NotImplementedError


class BaseExecutionEngine:
    """주문을 실제 체결로 변환하는 베이스 클래스."""

    def execute_orders(
        self,
        orders: Iterable[Order],
        date: datetime,
        data_provider: DataProvider,
        portfolio: Portfolio,
    ) -> List[Fill]:  # pragma: no cover - 기본 구현
        raise NotImplementedError


class BacktestExecutionEngine(BaseExecutionEngine):
    """단순한 종가 체결을 지원하는 백테스트용 체결 엔진.

    기본 구현은 해당 날짜의 종가로 전량 체결한다고 가정한다.
    """

    def execute_orders(
        self,
        orders: Iterable[Order],
        date: datetime,
        data_provider: DataProvider,
        portfolio: Portfolio,
    ) -> List[Fill]:
        fills: List[Fill] = []
        for order in orders:
            price = data_provider.get_price(order.ticker, date)
            if price is None:
                # 해당 일자 가격이 없으면 체결 불가
                continue

            if order.side == "BUY":
                portfolio.buy_stock(
                    ticker=order.ticker,
                    date=date,
                    price=price,
                    quantity=order.quantity,
                    cost_pct=order.cost_pct,
                    note=order.note,
                )
            else:
                portfolio.sell_stock(
                    ticker=order.ticker,
                    date=date,
                    price=price,
                    quantity=order.quantity,
                    cost_pct=order.cost_pct,
                    note=order.note,
                )

            fills.append(
                Fill(
                    date=date,
                    ticker=order.ticker,
                    side=order.side,
                    price=price,
                    quantity=order.quantity,
                    cost_pct=order.cost_pct,
                )
            )
        return fills


