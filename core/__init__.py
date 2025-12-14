"""core 패키지
리팩토링된 주식 데이터 및 보조지표 모듈
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "Stock",
    "TechnicalIndicators",
    "DataFetcher",
    "DataSource",
    "YFinanceDataSource",
    "ChartRenderer",
    "ChartConfig",
    "TradeSignal",
    "Portfolio",
    "Position",
    "Trade",
    "CashEvent",
    # backtest: types
    "Order",
    "Fill",
    "SnapshotRecord",
    "TradeRecord",
    "BacktestResult",
    # backtest: data
    "DataProvider",
    "BacktestDataProvider",
    # backtest: strategy/execution
    "BaseStrategy",
    "BaseExecutionEngine",
    "BacktestExecutionEngine",
    # backtest: engine/report
    "BacktestEngine",
    "BacktestReport",
]

if TYPE_CHECKING:  # pragma: no cover
    from core.stock import Stock
    from core.indicators import TechnicalIndicators
    from core.data import DataFetcher
    from core.data_source import DataSource, YFinanceDataSource
    from core.chart import ChartRenderer, ChartConfig, TradeSignal
    from core.portfolio import CashEvent, Portfolio, Position, Trade
    from core.backtest_types import BacktestResult, Fill, Order, SnapshotRecord, TradeRecord
    from core.backtest_data import BacktestDataProvider, DataProvider
    from core.backtest_engine import BacktestEngine
    from core.backtest_report import BacktestReport
    from core.backtest_strategy import BaseExecutionEngine, BaseStrategy


def __getattr__(name: str) -> Any:
    if name == "Stock":
        from .stock import Stock
        return Stock
    if name == "TechnicalIndicators":
        from .indicators import TechnicalIndicators
        return TechnicalIndicators
    if name == "DataFetcher":
        from .data import DataFetcher
        return DataFetcher
    if name == "DataSource":
        from .data_source import DataSource
        return DataSource
    if name == "YFinanceDataSource":
        from .data_source import YFinanceDataSource
        return YFinanceDataSource
    if name == "ChartRenderer":
        from .chart import ChartRenderer
        return ChartRenderer
    if name == "ChartConfig":
        from .chart import ChartConfig
        return ChartConfig
    if name == "TradeSignal":
        from .chart import TradeSignal
        return TradeSignal
    if name == "Portfolio":
        from .portfolio import Portfolio
        return Portfolio
    if name == "Position":
        from .portfolio import Position
        return Position
    if name == "Trade":
        from .portfolio import Trade
        return Trade
    if name == "CashEvent":
        from .portfolio import CashEvent
        return CashEvent
    if name in {"Order", "Fill", "SnapshotRecord", "TradeRecord", "BacktestResult"}:
        from .backtest_types import BacktestResult, Fill, Order, SnapshotRecord, TradeRecord

        return {
            "Order": Order,
            "Fill": Fill,
            "SnapshotRecord": SnapshotRecord,
            "TradeRecord": TradeRecord,
            "BacktestResult": BacktestResult,
        }[name]
    if name in {"DataProvider", "BacktestDataProvider"}:
        from .backtest_data import BacktestDataProvider, DataProvider

        return {"DataProvider": DataProvider, "BacktestDataProvider": BacktestDataProvider}[name]
    if name in {"BaseStrategy", "BaseExecutionEngine", "BacktestExecutionEngine"}:
        from .backtest_strategy import BaseExecutionEngine, BaseStrategy, BacktestExecutionEngine

        return {
            "BaseStrategy": BaseStrategy,
            "BaseExecutionEngine": BaseExecutionEngine,
            "BacktestExecutionEngine": BacktestExecutionEngine,
        }[name]
    if name in {"BacktestEngine", "BacktestReport"}:
        if name == "BacktestEngine":
            from .backtest_engine import BacktestEngine

            return BacktestEngine
        from .backtest_report import BacktestReport

        return BacktestReport
    raise AttributeError(f"module 'core' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(__all__)
