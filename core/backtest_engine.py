from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from core.portfolio import Portfolio

from .backtest_data import DataProvider
from .backtest_strategy import BaseExecutionEngine, BaseStrategy
from .backtest_types import BacktestResult, SnapshotRecord, TradeRecord
from .backtest_report import BacktestReport  # re-export 용도


class BacktestEngine:
    """전체 백테스트 시뮬레이션을 관리하는 엔진."""

    def __init__(
        self,
        universe,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        data_provider: DataProvider,
        execution_engine: BaseExecutionEngine,
        start_date: datetime,
        end_date: datetime,
        benchmark: str | None = None,
    ) -> None:
        self.universe = tuple(universe)
        self.strategy = strategy
        self.portfolio = portfolio
        self.data_provider = data_provider
        self.execution_engine = execution_engine
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self._snapshots: list[SnapshotRecord] = []

    def _compute_equity(self, date: datetime) -> Tuple[float, Dict[str, float]]:
        """현재 포트폴리오의 총 자산가치(모든 통화 합)와 종목별 가격을 계산."""
        prices: Dict[str, float] = {}
        for ticker, position in self.portfolio.positions.items():
            price = self.data_provider.get_price(ticker, date)
            if price is not None:
                position.last_price = price
            else:
                price = position.last_price
            if price is not None:
                prices[ticker] = price

        snapshot = self.portfolio.snapshot(prices=prices)
        totals = snapshot.get("totals", {})
        equity = float(sum(float(v) for v in totals.values()))
        return equity, prices

    def _annotate_trades_with_equity(self, date: datetime, equity: float) -> None:
        """해당 날짜에 발생한 거래 기록에 총 평가 금액을 기록."""
        self.portfolio.annotate_trades_with_value(date, equity)

    def run(self) -> BacktestResult:
        dates = self.data_provider.get_trading_days(self.start_date, self.end_date)
        if not dates:
            raise RuntimeError("지정된 기간에 대한 거래일이 없습니다.")

        self.strategy.on_start(self.portfolio)

        latest_prices: Dict[str, float] = {}
        for current_date in dates:
            orders = self.strategy.on_bar(current_date, self.portfolio)
            if orders is None:
                orders = []
            self.execution_engine.execute_orders(
                orders=orders,
                date=current_date,
                data_provider=self.data_provider,
                portfolio=self.portfolio,
            )
            equity, prices = self._compute_equity(current_date)
            self._snapshots.append(SnapshotRecord(date=current_date, equity=equity))
            self._annotate_trades_with_equity(current_date, equity)
            latest_prices = prices.copy()

        self.strategy.on_end(self.portfolio)

        equity_curve = pd.Series(
            data=[s.equity for s in self._snapshots],
            index=[s.date for s in self._snapshots],
            name="equity",
        )

        benchmark_curve: pd.Series | None = None
        if self.benchmark:
            start = dates[0]
            end = dates[-1]
            bench_df = self.data_provider.get_history(self.benchmark, start, end)
            if not bench_df.empty and "Date" in bench_df.columns and "Close" in bench_df.columns:
                bench_df = bench_df.sort_values("Date")
                closes = bench_df["Close"].astype(float)
                if not closes.empty and closes.iloc[0] != 0:
                    normalized = closes / float(closes.iloc[0])
                    # FutureWarning 회피: DatetimeProperties.to_pydatetime() 사용을 피하고 개별 Timestamp 변환
                    dt_series = pd.to_datetime(bench_df["Date"])
                    normalized.index = [ts.to_pydatetime() for ts in dt_series]
                    benchmark_curve = normalized

        trades: list[TradeRecord] = list(self.portfolio.get_trade_history())
        return BacktestResult(
            equity_curve=equity_curve,
            benchmark_equity_curve=benchmark_curve,
            trades=trades,
            latest_prices=latest_prices or None,
        )


