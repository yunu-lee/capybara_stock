from __future__ import annotations

import base64
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

import pandas as pd
from pykrx import stock as krx_stock

from core import (
    BacktestDataProvider,
    BacktestEngine,
    BacktestExecutionEngine,
    BacktestReport,
    Portfolio,
    Stock,
    TechnicalIndicators,
)
from core.backtest_report import HtmlReportConfig, build_html_report
from core.chart import ChartConfig, ChartRenderer, TradeSignal
from core.backtest_strategy import BaseStrategy
from core.backtest_types import BacktestResult, Order, TradeRecord
from core.backtest_data import BacktestDataProvider


NEW_HIGH_BACKTEST_CONFIG: Dict[str, Any] = {
    "universe": None,
    "benchmark": "069500.KS",
    "start_date": "2025-04-01",
    "end_date": None,
    "initial_cash": {"KRW": 100_000_000},
    "allocation_pct": 0.05,
    "stop_loss_pct": 0.08,
    "ma_window": 20,
    "high_lookback": 252,
    "cost_pct": 0.1,
    "debug": False,
    "rank_weight_cap": 0.0,
    "rank_weight_rs": 0.0,
    "fresh_new_high_window": 30,
    "fresh_new_high_weight": 1.0,
    "short_term_penalty_window": 10,
    "short_term_penalty_threshold": 0.12,
    "short_term_penalty_weight": 0.0,
}


DEFAULT_KOREA_UNIVERSE: List[str] = [
    "005930.KS",
    "000660.KS",
    "207940.KS",
    "035420.KS",
    "051910.KS",
    "005380.KS",
    "068270.KS",
    "035720.KS",
    "066570.KS",
    "028260.KS",
    "096770.KS",
    "034730.KS",
    "055550.KS",
    "105560.KS",
    "003550.KS",
    "251270.KS",
    "259960.KQ",
    "086520.KQ",
    "035900.KQ",
    "357780.KS",
]


def _fetch_index_members(index_code: str, suffix: str, lookback_days: int = 21) -> List[str]:
    """최근 거래일 기준 해당 지수 구성 종목 리스트를 반환."""
    today = datetime.now()
    for days_back in range(lookback_days):
        target = today - timedelta(days=days_back)
        date_str = target.strftime("%Y%m%d")
        try:
            tickers = krx_stock.get_index_portfolio_deposit_file(index_code, date_str)
        except Exception:
            continue
        if tickers:
            return [f"{code}.{suffix}" for code in tickers]
    return []


def build_korea_universe() -> List[str]:
    """KOSPI200 + KOSDAQ150 유니버스를 구성."""
    kospi200 = _fetch_index_members("1028", "KS")
    kosdaq150 = _fetch_index_members("2203", "KQ")
    merged = sorted({*kospi200, *kosdaq150})
    if not merged:
        raise RuntimeError("Failed to fetch KRX index members. Please provide a universe manually.")
    return merged


class NewHighBreakoutStrategy(BaseStrategy):
    """52주 신고가 & 시가총액 상위 종목 매수 전략."""

    def __init__(
        self,
        universe: Sequence[str],
        data_provider: BacktestDataProvider,
        *,
        allocation_pct: float = 0.10,
        stop_loss_pct: float = 0.08,
        ma_window: int = 20,
        high_lookback: int = 252,
        cost_pct: float = 0.1,
        cash_currency: str = "KRW",
        debug: bool = False,
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None,
        rank_weight_cap: float = 0.4,
        rank_weight_rs: float = 0.4,
        fresh_new_high_window: int = 30,
        fresh_new_high_weight: float = 0.2,
        short_term_penalty_window: int = 10,
        short_term_penalty_threshold: float = 0.12,
        short_term_penalty_weight: float = 0.2,
        max_new_positions: int = 20,
    ) -> None:
        super().__init__(universe, data_provider, debug=debug)
        if not universe:
            raise ValueError("universe must not be empty")
        self.allocation_pct = float(allocation_pct)
        self.stop_loss_pct = float(stop_loss_pct)
        self.ma_window = int(ma_window)
        self.high_lookback = int(high_lookback)
        self.cost_pct = float(cost_pct)
        self.cash_currency = cash_currency
        self.debug = bool(debug)
        self.benchmark = getattr(data_provider, "benchmark", None)
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.max_new_positions = max(1, int(max_new_positions))
        approx_calendar_days = int(self.high_lookback * 365 / 252) + 60
        self._history_margin_days = max(approx_calendar_days, self.ma_window * 6, 365)
        self._indicator_cache: Dict[str, pd.DataFrame] = {}
        self._rs_lookback_days = max(90, self.ma_window * 6)
        self._benchmark_history: Optional[pd.DataFrame] = None
        self.rank_weight_cap = float(rank_weight_cap)
        self.rank_weight_rs = float(rank_weight_rs)
        self.fresh_new_high_window = max(1, int(fresh_new_high_window))
        self.fresh_new_high_weight = float(fresh_new_high_weight)
        self.short_term_penalty_window = max(1, int(short_term_penalty_window))
        self.short_term_penalty_threshold = float(short_term_penalty_threshold)
        self.short_term_penalty_weight = float(short_term_penalty_weight)

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][NEW_HIGH] {message}")

    @staticmethod
    def _normalize_value_map(values: Dict[str, float]) -> Dict[str, float]:
        usable = [v for v in values.values() if v is not None]
        if not usable:
            return {k: 0.0 for k in values}
        v_min = min(usable)
        v_max = max(usable)
        if math.isclose(v_max, v_min):
            return {k: 0.5 for k in values}
        return {
            key: (float(value) - v_min) / (v_max - v_min) if value is not None else 0.0
            for key, value in values.items()
        }

    def _compute_rs_score(self, ticker: str, date: datetime) -> float:
        if not self.benchmark:
            return 0.0
        lookback = self._rs_lookback_days
        start = date - timedelta(days=lookback)
        asset_df = self.data_provider.get_history(ticker, start, date)
        bench_df = self.data_provider.get_history(self.benchmark, start, date)
        if (
            asset_df.empty
            or bench_df.empty
            or "Date" not in asset_df.columns
            or "Close" not in asset_df.columns
            or "Close" not in bench_df.columns
        ):
            return 0.0
        asset = asset_df[["Date", "Close"]].dropna()
        bench = bench_df[["Date", "Close"]].dropna()
        merged = pd.merge(asset, bench, on="Date", how="inner", suffixes=("_asset", "_bench"))
        if len(merged) < 5:
            return 0.0
        merged = merged.sort_values("Date")
        asset_start = float(merged["Close_asset"].iloc[0])
        bench_start = float(merged["Close_bench"].iloc[0])
        asset_end = float(merged["Close_asset"].iloc[-1])
        bench_end = float(merged["Close_bench"].iloc[-1])
        if asset_start <= 0 or bench_start <= 0:
            return 0.0
        asset_return = asset_end / asset_start - 1.0
        bench_return = bench_end / bench_start - 1.0
        return asset_return - bench_return

    def _compute_freshness_score(self, ticker: str, day_key: datetime) -> float:
        frame = self._indicator_cache.get(ticker)
        if frame is None or frame.empty:
            return 0.0
        if day_key not in frame.index:
            subset = frame.loc[:day_key]
            if subset.empty:
                return 0.0
            row = subset.iloc[-1]
        else:
            row = frame.loc[day_key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
        count = float(row.get("recent_new_high_count", 0.0) or 0.0)
        window = float(self.fresh_new_high_window)
        freshness = 1.0 - min(count, window) / window
        return max(0.0, min(1.0, freshness))

    def _compute_short_term_penalty(self, ticker: str, day_key: datetime) -> float:
        frame = self._indicator_cache.get(ticker)
        if frame is None or frame.empty:
            return 0.0
        if day_key not in frame.index:
            subset = frame.loc[:day_key]
            if subset.empty:
                return 0.0
            current_row = subset.iloc[-1]
            current_date = current_row.name
        else:
            current_row = frame.loc[day_key]
            if isinstance(current_row, pd.DataFrame):
                current_row = current_row.iloc[-1]
            current_date = day_key
        prev_cutoff = current_date - timedelta(days=self.short_term_penalty_window)
        prev_slice = frame.loc[:prev_cutoff]
        if prev_slice.empty:
            return 0.0
        prev_row = prev_slice.iloc[-1]
        price_today = float(current_row.get("Close", float("nan")))
        price_prev = float(prev_row.get("Close", float("nan")))
        if math.isnan(price_today) or math.isnan(price_prev) or price_prev <= 0:
            return 0.0
        ret = price_today / price_prev - 1.0
        excess = ret - self.short_term_penalty_threshold
        if excess <= 0:
            return 0.0
        penalty = min(1.0, excess / max(0.0001, self.short_term_penalty_threshold))
        return penalty

    def _benchmark_allows_buy(self, date: datetime) -> bool:
        if not self.benchmark or self._benchmark_history is None:
            return True
        bench_df = self._benchmark_history
        if bench_df.empty or "Date" not in bench_df.columns or "Close" not in bench_df.columns:
            return True
        bench_df = bench_df.copy()
        bench_df["Date"] = pd.to_datetime(bench_df["Date"])
        bench_df = bench_df.sort_values("Date")
        bench_df = bench_df[bench_df["Date"] <= date]
        if bench_df.empty:
            return True
        closes = bench_df["Close"].astype(float)
        if len(closes) < 60:
            return True
        ma60 = closes.rolling(window=60).mean().iloc[-1]
        last_price = closes.iloc[-1]
        if pd.isna(ma60) or pd.isna(last_price):
            return True
        return last_price >= ma60

    def on_start(self, portfolio: Portfolio) -> None:
        super().on_start(portfolio)
        if self.backtest_start is None or self.backtest_end is None:
            raise ValueError("backtest_start/end must be provided")
        self._indicator_cache.clear()
        history_start = self.backtest_start - timedelta(days=self._history_margin_days)
        for ticker in self.universe:
            history = self.data_provider.get_history(ticker, history_start, self.backtest_end)
            indicators = self._prepare_indicator_frame(ticker, history)
            self._indicator_cache[ticker] = indicators
            if self.debug:
                self._debug(
                    f"prepared {ticker} rows={len(indicators)} start={history_start:%Y-%m-%d} end={self.backtest_end:%Y-%m-%d}"
                )
        if self.benchmark:
            bench_history_start = self.backtest_start - timedelta(days=self.short_term_penalty_window * 4)
            self._benchmark_history = self.data_provider.get_history(self.benchmark, bench_history_start, self.backtest_end)
        else:
            self._benchmark_history = None

    def _prepare_indicator_frame(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()
        data = (
            df.copy()
            .dropna(subset=["Date", "Close"])
            .assign(Date=lambda x: pd.to_datetime(x["Date"]))
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )
        ti = TechnicalIndicators(data)
        sma = ti.sma(self.ma_window)
        rolling_max = ti.rolling_max(self.high_lookback)
        prev_max = rolling_max.shift(1)
        new_high = ti.new_high_signal(self.high_lookback)
        ready = (~sma.isna()) & (~prev_max.isna())
        recent_new_high_count = (
            new_high.astype(int)
            .rolling(window=self.fresh_new_high_window, min_periods=1)
            .sum()
        )
        frame = pd.DataFrame(
            {
                "Date": data["Date"],
                "Close": data["Close"].astype(float),
                "sma": sma,
                "new_high": new_high.fillna(False).astype(bool),
                "ready": ready.fillna(False),
                "recent_new_high_count": recent_new_high_count.fillna(0.0),
            }
        )
        frame["date"] = frame["Date"].dt.normalize()
        frame = frame.set_index("date")
        return frame[["Close", "sma", "new_high", "ready", "recent_new_high_count"]]

    def _get_indicator_row(self, ticker: str, day_key: datetime) -> Optional[pd.Series]:
        frame = self._indicator_cache.get(ticker)
        if frame is None or frame.empty:
            return None
        if day_key not in frame.index:
            return None
        row = frame.loc[day_key]
        if isinstance(row, pd.DataFrame):  # pragma: no cover - guard for duplicate index
            row = row.iloc[-1]
        return row

    def _rank_candidates(self, tickers: List[str], date: datetime) -> List[str]:
        if not tickers:
            return []
        date_str = date.strftime("%Y%m%d")
        cap_map_raw = Stock.fetch_market_caps(tickers, date=date_str)
        cap_values: Dict[str, float] = {
            ticker: cap_map_raw.get(Stock.normalize_code(ticker), 0.0) for ticker in tickers
        }
        cap_norm = self._normalize_value_map(cap_values)

        rs_values: Dict[str, float] = {ticker: self._compute_rs_score(ticker, date) for ticker in tickers}
        rs_norm = self._normalize_value_map(rs_values)

        scores = []
        for ticker in tickers:
            base_score = (
                self.rank_weight_cap * cap_norm.get(ticker, 0.0)
                + self.rank_weight_rs * rs_norm.get(ticker, 0.0)
            )
            freshness = self._compute_freshness_score(ticker, date)
            penalty = self._compute_short_term_penalty(ticker, date)
            score = (
                base_score
                + self.fresh_new_high_weight * freshness
                - self.short_term_penalty_weight * penalty
            )
            scores.append((ticker, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [ticker for ticker, _ in scores]

    def _compute_portfolio_value(self, portfolio: Portfolio, prices: Dict[str, float]) -> float:
        snapshot = portfolio.snapshot(prices=prices)
        totals = snapshot.get("totals", {})
        return float(totals.get(self.cash_currency, 0.0))

    def on_bar(self, date: datetime, portfolio: Portfolio) -> List[Order]:
        orders: List[Order] = []
        day_key = datetime(date.year, date.month, date.day)

        indicator_rows: Dict[str, pd.Series] = {}
        current_prices: Dict[str, float] = {}
        for ticker in self.universe:
            row = self._get_indicator_row(ticker, day_key)
            if row is None:
                continue
            price = float(row.get("Close", float("nan")))
            if pd.isna(price):
                continue
            indicator_rows[ticker] = row
            current_prices[ticker] = price

        # ensure we have prices for held positions to compute equity
        for ticker, position in portfolio.positions.items():
            if ticker in current_prices or position.quantity <= 0:
                continue
            price = self.data_provider.get_price(ticker, date)
            if price is not None:
                current_prices[ticker] = float(price)

        # Sell logic (stop-loss or MA breakdown)
        for ticker, position in portfolio.positions.items():
            qty = float(position.quantity)
            if qty <= 0:
                continue
            row = indicator_rows.get(ticker)
            if row is None:
                continue
            price = float(row.get("Close", float("nan")))
            sma_value = row.get("sma")
            if pd.isna(price) or pd.isna(sma_value):
                continue
            stop_price = position.avg_price * (1 - self.stop_loss_pct)
            hit_stop = price <= stop_price and position.avg_price > 0
            below_ma = price < float(sma_value)
            if hit_stop or below_ma:
                note = "stop_loss" if hit_stop else "ma_break"
                orders.append(
                    Order(
                        ticker=ticker,
                        side="SELL",
                        quantity=qty,
                        cost_pct=self.cost_pct,
                        note=f"{note} exit",
                    )
                )

        candidates = [
            ticker
            for ticker, row in indicator_rows.items()
            if bool(row.get("ready", False)) and bool(row.get("new_high", False))
        ]
        if not candidates:
            return orders

        ranked = self._rank_candidates(candidates, day_key)
        ranked = ranked[: self.max_new_positions]

        portfolio_value = self._compute_portfolio_value(portfolio, current_prices)
        target_amount = portfolio_value * self.allocation_pct
        cash_balance = float(portfolio.get_cash_balances().get(self.cash_currency, 0.0))
        available_cash = cash_balance

        if target_amount <= 0 or available_cash <= 0 or not self._benchmark_allows_buy(date):
            return orders

        for ticker in ranked:
            if available_cash <= 0:
                break
            position = portfolio.positions.get(ticker)
            if position and position.quantity > 0:
                continue
            row = indicator_rows.get(ticker)
            if row is None:
                continue
            price = float(row.get("Close", float("nan")))
            if pd.isna(price) or price <= 0:
                continue
            invest_amount = min(target_amount, available_cash)
            effective_price = price * (1 + self.cost_pct / 100.0)
            quantity = int(invest_amount // effective_price)
            if quantity <= 0:
                continue
            orders.append(
                Order(
                    ticker=ticker,
                    side="BUY",
                    quantity=quantity,
                    cost_pct=self.cost_pct,
                    note="52w_new_high_entry",
                )
            )
            available_cash -= quantity * effective_price

        return orders


def run_new_high_breakout_backtest(
    config_override: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, Any],
    "pd.DataFrame",
    List[TradeRecord],
    Sequence[str],
    BacktestDataProvider,
    datetime,
    datetime,
    Dict[str, Any],
]:
    cfg: Dict[str, Any] = {**NEW_HIGH_BACKTEST_CONFIG, **(config_override or {})}

    universe: Sequence[str] = cfg.get("universe") or build_korea_universe()
    if not universe:
        raise RuntimeError("Universe is empty. Provide tickers or ensure KRX data is available.")
    benchmark: str = cfg.get("benchmark", "069500.KS")

    start_cfg = cfg.get("start_date", "2018-01-01")
    end_cfg = cfg.get("end_date")

    if isinstance(start_cfg, str):
        start_date = datetime.fromisoformat(start_cfg)
    elif isinstance(start_cfg, datetime):
        start_date = start_cfg
    else:
        raise TypeError("start_date must be str or datetime")

    if end_cfg is None:
        end_date = datetime.now()
    elif isinstance(end_cfg, str):
        end_date = datetime.fromisoformat(end_cfg)
    elif isinstance(end_cfg, datetime):
        end_date = end_cfg
    else:
        raise TypeError("end_date must be None, str, or datetime")

    data_provider = BacktestDataProvider(universe=universe, benchmark=benchmark)

    initial_cash = cfg.get("initial_cash", {"KRW": 100_000_000})
    cash_currency = cfg.get("cash_currency")
    if not cash_currency:
        if isinstance(initial_cash, dict) and initial_cash:
            cash_currency = next(iter(initial_cash.keys()))
        else:
            cash_currency = "KRW"

    portfolio = Portfolio(initial_cash=initial_cash)
    for ticker in universe:
        portfolio.add_stock(ticker, cash_currency)

    strategy = NewHighBreakoutStrategy(
        universe=universe,
        data_provider=data_provider,
        allocation_pct=float(cfg.get("allocation_pct", 0.10)),
        stop_loss_pct=float(cfg.get("stop_loss_pct", 0.08)),
        ma_window=int(cfg.get("ma_window", 20)),
        high_lookback=int(cfg.get("high_lookback", 252)),
        cost_pct=float(cfg.get("cost_pct", 0.1)),
        cash_currency=cash_currency,
        debug=bool(cfg.get("debug", False)),
        backtest_start=start_date,
        backtest_end=end_date,
        rank_weight_cap=float(cfg.get("rank_weight_cap", 0.4)),
        rank_weight_rs=float(cfg.get("rank_weight_rs", 0.4)),
        fresh_new_high_window=int(cfg.get("fresh_new_high_window", 30)),
        fresh_new_high_weight=float(cfg.get("fresh_new_high_weight", 0.2)),
        short_term_penalty_window=int(cfg.get("short_term_penalty_window", 10)),
        short_term_penalty_threshold=float(cfg.get("short_term_penalty_threshold", 0.12)),
        short_term_penalty_weight=float(cfg.get("short_term_penalty_weight", 0.2)),
        max_new_positions=int(cfg.get("max_new_positions", 20)),
    )

    execution_engine = BacktestExecutionEngine()
    engine = BacktestEngine(
        universe=universe,
        strategy=strategy,
        portfolio=portfolio,
        data_provider=data_provider,
        execution_engine=execution_engine,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
    )

    result: BacktestResult = engine.run()
    report = BacktestReport(result)
    metrics: Dict[str, Any] = report.compute_metrics()
    df_equity: pd.DataFrame = report.to_dataframe()

    print("=== New High Breakout Backtest ===")
    print(f"Period: {metrics['start']} ~ {metrics['end']}")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    if metrics.get("benchmark_total_return") is not None:
        print(f"Benchmark Return: {metrics['benchmark_total_return'] * 100:.2f}%")
    print(f"MDD: {metrics['mdd'] * 100:.2f}%")
    print(f"Trades: {metrics['num_trades']}")

    return (
        metrics,
        df_equity,
        result.trades,
        universe,
        data_provider,
        start_date,
        end_date,
        cfg,
    )


def format_new_high_backtest_message(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("52주 신고가 브레이크아웃 백테스트")
    start = metrics.get("start")
    end = metrics.get("end")
    start_str = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
    end_str = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)
    lines.append(f"기간: {start_str} ~ {end_str}")
    lines.append(f"총 수익률: {metrics.get('total_return', 0.0) * 100:.2f}%")
    bench = metrics.get("benchmark_total_return")
    if bench is not None:
        lines.append(f"벤치마크: {bench * 100:.2f}%")
    lines.append(f"MDD: {metrics.get('mdd', 0.0) * 100:.2f}%")
    lines.append(f"거래 횟수: {metrics.get('num_trades', 0)}")
    lines.append(
        f"설정: 배분 {cfg.get('allocation_pct', 0.1) * 100:.0f}%, 손절 {cfg.get('stop_loss_pct', 0.08) * 100:.0f}%, 20MA 이탈"
    )
    return "\n".join(lines)


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _optimize_chart_image(path: str, max_width: int = 900, quality: int = 82) -> str:
    if Image is None:
        return path
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / float(img.width)
                new_height = int(img.height * ratio)
                img = img.resize((max_width, max(new_height, 1)), Image.LANCZOS)
            optimized_path = os.path.splitext(path)[0] + ".jpg"
            img.save(
                optimized_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
        return optimized_path
    except Exception as exc:  # pragma: no cover - optimization best effort
        print(f"[WARN] Failed to optimize chart image {path}: {exc}")
        return path


def _build_trade_signals_for_ticker(ticker: str, trades: List[TradeRecord]) -> List[TradeSignal]:
    signals: List[TradeSignal] = []
    for trade in trades:
        if getattr(trade, "ticker", None) != ticker:
            continue
        action = getattr(trade, "action", "").upper()
        if action not in {"BUY", "SELL"}:
            continue
        trade_date = getattr(trade, "date", None)
        trade_price = getattr(trade, "price", None)
        if not isinstance(trade_date, datetime) or trade_price is None:
            continue
        signal_type = "buy" if action == "BUY" else "sell"
        signals.append(
            TradeSignal(
                date=trade_date,
                price=float(trade_price),
                type=signal_type,
            )
        )
    return signals


def _format_price(value: float, currency: str) -> str:
    symbol = "₩" if currency.upper() == "KRW" else ""
    if abs(value) >= 1000:
        return f"{symbol}{value:,.0f}"
    return f"{symbol}{value:,.2f}"


def _format_quantity(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:,.0f}"
    return f"{value:.4f}"


def _format_trade_pair_line(
    *,
    buy_trade: TradeRecord,
    sell_trade: Optional[TradeRecord],
    quantity: float,
    latest_price: Optional[float] = None,
) -> str:
    currency = getattr(buy_trade, "currency", "KRW") or "KRW"
    buy_date = buy_trade.date.strftime("%Y/%m/%d") if buy_trade.date else "N/A"
    buy_price = float(getattr(buy_trade, "price", 0.0))
    qty_str = _format_quantity(quantity)
    buy_price_str = _format_price(buy_price, currency)

    buy_total = buy_price * quantity
    buy_total_str = _format_price(buy_total, currency)

    if sell_trade is None:
        if latest_price is None:
            return f"{buy_date} 매수 {qty_str}주 @ {buy_price_str} (총액 {buy_total_str}) (보유 중)"
        latest_price_str = _format_price(latest_price, currency)
        current_value = latest_price * quantity
        current_total_str = _format_price(current_value, currency)
        pnl_pct = ((latest_price - buy_price) / buy_price * 100.0) if buy_price else 0.0
        pnl_str = f"{pnl_pct:+.2f}%"
        return (
            f"{buy_date} 매수 {qty_str}주 @ {buy_price_str} (총액 {buy_total_str}) "
            f"(보유 중, 현재가 {latest_price_str}, 평가액 {current_total_str}, 수익률 {pnl_str})"
        )

    sell_date = sell_trade.date.strftime("%Y/%m/%d") if sell_trade.date else "N/A"
    sell_price = float(getattr(sell_trade, "price", 0.0))
    sell_price_str = _format_price(sell_price, currency)
    sell_total = sell_price * quantity
    sell_total_str = _format_price(sell_total, currency)
    pnl_pct = ((sell_price - buy_price) / buy_price * 100.0) if buy_price else 0.0
    pnl_str = f"{pnl_pct:+.2f}%"
    return (
        f"{buy_date} 매수 {qty_str}주 @ {buy_price_str} (총액 {buy_total_str}) → "
        f"{sell_date} 매도 @ {sell_price_str} (총액 {sell_total_str}) | 수익률 {pnl_str}"
    )


@lru_cache(maxsize=256)
def _resolve_ticker_name(ticker: str) -> str:
    try:
        stock_obj = Stock(ticker)
        return stock_obj.name or ticker
    except Exception:
        return ticker


@lru_cache(maxsize=256)
def _resolve_market_label(ticker: str) -> str:
    if not ticker:
        return "기타"
    normalized = ticker.upper()
    if normalized.endswith(".KS"):
        return "KOSPI"
    if normalized.endswith(".KQ"):
        return "KOSDAQ"
    return "기타"


def _get_latest_price(
    data_provider: BacktestDataProvider,
    ticker: str,
    as_of: datetime,
    lookback_days: int = 10,
) -> Optional[float]:
    for offset in range(lookback_days + 1):
        target = as_of - timedelta(days=offset)
        price = data_provider.get_price(ticker, target)
        if price is not None:
            return float(price)
    return None


def _render_trade_summary_html(
    *,
    trades: List[TradeRecord],
    chart_paths: Dict[str, str],
    latest_price_map: Dict[str, Optional[float]],
    html_output_dir: str,
) -> str:
    base_dir = os.path.abspath(html_output_dir)
    grouped: Dict[str, List[TradeRecord]] = defaultdict(list)
    for trade in trades:
        ticker = getattr(trade, "ticker", None)
        if not ticker:
            continue
        grouped[ticker].append(trade)

    ticker_blocks: List[str] = []

    for ticker in sorted(grouped.keys()):
        ticker_trades = grouped[ticker]
        ticker_trades.sort(key=lambda t: t.date)
        open_lots: List[Dict[str, Any]] = []
        entries: List[str] = []
        latest_price = latest_price_map.get(ticker)
        for trade in ticker_trades:
            action = getattr(trade, "action", "").upper()
            if action == "BUY":
                open_lots.append({"trade": trade, "remaining": float(trade.quantity)})
                continue
            if action != "SELL":
                continue
            qty_to_match = float(trade.quantity)
            while qty_to_match > 1e-8 and open_lots:
                lot = open_lots[0]
                available = float(lot["remaining"])
                match_qty = min(available, qty_to_match)
                entries.append(
                    _format_trade_pair_line(
                        buy_trade=lot["trade"],
                        sell_trade=trade,
                        quantity=match_qty,
                    )
                )
                lot["remaining"] = available - match_qty
                qty_to_match -= match_qty
                if lot["remaining"] <= 1e-8:
                    open_lots.pop(0)
        for lot in open_lots:
            remaining = float(lot["remaining"])
            if remaining <= 1e-8:
                continue
            entries.append(
                _format_trade_pair_line(
                    buy_trade=lot["trade"],
                    sell_trade=None,
                    quantity=remaining,
                    latest_price=latest_price,
                )
            )

        if not entries:
            continue

        stock_name = _resolve_ticker_name(ticker)
        market_label = _resolve_market_label(ticker)
        ticker_blocks.append(f"<h3>{stock_name} ({ticker}) [{market_label}]</h3>")
        ticker_blocks.append(f"<p><em>시장: {market_label}</em></p>")
        ticker_blocks.append("<ul>")
        for entry in entries:
            ticker_blocks.append(f"  <li>{entry}</li>")
        ticker_blocks.append("</ul>")
        chart_path = chart_paths.get(ticker)
        if chart_path and os.path.exists(chart_path):
            try:
                with open(chart_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("ascii")
                if chart_path.lower().endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                else:
                    mime = "image/png"
                data_uri = f"data:{mime};base64,{encoded}"
                ticker_blocks.append(
                    f'<div class="ticker-chart"><img src="{data_uri}" alt="{ticker} ({market_label}) chart" '
                    f'style="width:100%;max-width:900px;height:auto;" /></div>'
                )
            except OSError as exc:
                print(f"[WARN] Failed to embed chart for {ticker}: {exc}")

    if not ticker_blocks:
        return ""

    html_parts: List[str] = ["<h2>거래 요약</h2>"]
    html_parts.extend(ticker_blocks)
    return "\n".join(html_parts)


def _compute_trade_stats_and_returns(
    trades: List[TradeRecord],
) -> Tuple[str, List[Tuple[datetime, float]]]:
    if not trades:
        return "", []
    buy_queues: Dict[str, deque[Dict[str, float]]] = defaultdict(deque)
    success_returns_24: List[float] = []
    failure_returns_24: List[float] = []
    success_returns_0: List[float] = []
    failure_returns_0: List[float] = []
    sell_returns_by_date: Dict[datetime, List[float]] = defaultdict(list)

    def _unit_cost(trade: TradeRecord) -> float:
        qty = float(getattr(trade, "quantity", 0.0))
        total = float(getattr(trade, "total", 0.0))
        price = float(getattr(trade, "price", 0.0))
        if qty > 0 and total:
            return abs(total) / qty
        return price

    sorted_trades = sorted(trades, key=lambda t: getattr(t, "date", datetime.min))
    for trade in sorted_trades:
        action = getattr(trade, "action", "").upper()
        qty = float(getattr(trade, "quantity", 0.0))
        if qty <= 0:
            continue
        ticker = getattr(trade, "ticker", None)
        if not ticker:
            continue
        if action == "BUY":
            cost = _unit_cost(trade)
            if cost > 0:
                buy_queues[ticker].append(
                    {
                        "remaining": qty,
                        "unit_cost": cost,
                    }
                )
            continue
        if action != "SELL":
            continue
        queue = buy_queues.get(ticker)
        if not queue:
            continue
        remaining = qty
        sell_total = float(getattr(trade, "total", 0.0))
        sell_price = float(getattr(trade, "price", 0.0))
        sell_unit = sell_total / qty if qty > 0 and sell_total else sell_price
        sell_date = getattr(trade, "date", None) or datetime.min
        while remaining > 1e-9 and queue:
            lot = queue[0]
            lot_remaining = float(lot["remaining"])
            match_qty = min(lot_remaining, remaining)
            unit_cost = float(lot["unit_cost"])
            if unit_cost > 0:
                pnl_pct = (sell_unit - unit_cost) / unit_cost
                if pnl_pct >= 0.24:
                    success_returns_24.append(pnl_pct)
                else:
                    failure_returns_24.append(pnl_pct)
                if pnl_pct > 0:
                    success_returns_0.append(pnl_pct)
                else:
                    failure_returns_0.append(pnl_pct)
                sell_returns_by_date[sell_date].append(pnl_pct)
            lot_remaining -= match_qty
            remaining -= match_qty
            if lot_remaining <= 1e-9:
                queue.popleft()
            else:
                lot["remaining"] = lot_remaining
        if not queue:
            buy_queues.pop(ticker, None)
    def _summarize_stats(success_list: List[float], failure_list: List[float]) -> Optional[Dict[str, float]]:
        total = len(success_list) + len(failure_list)
        if total == 0:
            return None
        success_ratio = len(success_list) / total
        failure_ratio = len(failure_list) / total
        success_avg = sum(success_list) / len(success_list) if success_list else 0.0
        failure_avg = sum(failure_list) / len(failure_list) if failure_list else 0.0
        loss_mag = abs(failure_avg)
        if loss_mag <= 1e-9:
            reward_risk = float("inf") if success_avg > 0 else 0.0
        else:
            reward_risk = success_avg / loss_mag
        if math.isfinite(reward_risk):
            tpi = success_ratio * (1.0 + reward_risk)
        else:
            tpi = float("inf")
        return {
            "success_ratio": success_ratio,
            "failure_ratio": failure_ratio,
            "success_avg": success_avg,
            "failure_avg": failure_avg,
            "success_count": len(success_list),
            "failure_count": len(failure_list),
            "reward_risk": reward_risk,
            "tpi": tpi,
        }

    def _render_stats_section(title: str, stats: Dict[str, float]) -> str:
        return f"""
      <div class="stat-subtitle">{title}</div>
      <div class="grid">
        <div class="card">
          <div class="card-label">승률</div>
          <div class="card-value">{stats['success_ratio'] * 100:.2f}% ({stats['success_count']})</div>
        </div>
        <div class="card">
          <div class="card-label">패배율</div>
          <div class="card-value">{stats['failure_ratio'] * 100:.2f}% ({stats['failure_count']})</div>
        </div>
        <div class="card">
          <div class="card-label">평균 수익률</div>
          <div class="card-value">{stats['success_avg'] * 100:+.2f}%</div>
        </div>
        <div class="card">
          <div class="card-label">평균 손실률</div>
          <div class="card-value">{stats['failure_avg'] * 100:+.2f}%</div>
        </div>
        <div class="card">
          <div class="card-label">TPI</div>
          <div class="card-value">{("∞" if math.isinf(stats['tpi']) else f"{stats['tpi']:.2f}x")}</div>
          <div class="card-desc">승률 × (1 + 손익비)</div>
        </div>
      </div>
    """

    stats_html = ""
    stats_sections: List[str] = []
    stats_24 = _summarize_stats(success_returns_24, failure_returns_24)
    stats_0 = _summarize_stats(success_returns_0, failure_returns_0)
    if stats_24:
        stats_sections.append(_render_stats_section("≥ 24% 기준", stats_24))
    if stats_0:
        stats_sections.append(_render_stats_section("> 0% 기준", stats_0))
    if stats_sections:
        sections_html = "\n".join(stats_sections)
        stats_html = f"""
      <h2>거래 통계</h2>
      {sections_html}
      <p class="stats-note">TPI = 승률 × (1 + 손익비). 손익비는 평균 수익률 ÷ |평균 손실률|이며, 값이 높을수록 위험 대비 성과가 우수합니다.</p>
    """
    sell_return_series = sorted(
        (date, sum(values) / len(values)) for date, values in sell_returns_by_date.items()
    )
    return stats_html, sell_return_series


def _render_new_high_ticker_chart(
    *,
    ticker: str,
    data_provider: BacktestDataProvider,
    start_date: datetime,
    end_date: datetime,
    trades: List[TradeRecord],
    ma_window: int,
    high_lookback: int,
) -> str:
    history_margin_days = max(365, high_lookback + ma_window * 3)
    history_start = start_date - timedelta(days=history_margin_days)
    df_price = data_provider.get_history(ticker, history_start, end_date)
    if df_price.empty or "Date" not in df_price.columns:
        raise RuntimeError(f"No price data available for {ticker}")

    df_price = df_price.copy()
    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_render = df_price[df_price["Date"] >= start_date].reset_index(drop=True)
    if df_render.empty:
        df_render = df_price.tail(250).reset_index(drop=True)
    df_indexed = df_price.set_index("Date")
    closes = df_indexed["Close"].astype(float)

    sma_series = closes.rolling(window=ma_window).mean().rename(f"sma_{ma_window}")
    indicators_data = {f"sma_{ma_window}": sma_series}

    trade_signals = _build_trade_signals_for_ticker(ticker, trades)
    sma_windows = [int(ma_window)]

    config = ChartConfig(
        show_trade_signals=True,
        show_sma=True,
        show_volume=True,
        sma_windows=sma_windows,
        dpi=140,
        figsize=(9, 5),
    )
    renderer = ChartRenderer(config=config)

    safe_ticker = ticker.replace(".", "_")
    chart_path = os.path.join("output", f"new_high_{safe_ticker}.png")
    _ensure_output_dir(chart_path)
    out_path = renderer.render(
        df=df_render,
        indicators_data=indicators_data,
        trade_signals=trade_signals,
        display_days=None,
        title=f"{ticker} 52주 신고가 백테스트",
        save_path=chart_path,
    )
    optimized_path = _optimize_chart_image(os.path.abspath(out_path))
    return optimized_path


def send_new_high_backtest_report(
    metrics: Dict[str, Any],
    df_equity: "pd.DataFrame",
    trades: List[TradeRecord],
    universe: Sequence[str],
    data_provider: BacktestDataProvider,
    start_date: datetime,
    end_date: datetime,
    cfg: Dict[str, Any],
    *,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> str:
    equity_series = df_equity.get("equity")
    if equity_series is None:
        print("[WARN] df_equity 에 'equity' 컬럼이 없어 HTML 리포트를 건너뜁니다.")
        return ""

    equity_curve = equity_series.dropna()
    if equity_curve.empty:
        print("[WARN] Equity curve 가 비어 있어 HTML 리포트를 건너뜁니다.")
        return ""

    benchmark_curve = None
    if "benchmark" in df_equity.columns:
        bench_series = df_equity["benchmark"].dropna()
        if not bench_series.empty:
            benchmark_curve = bench_series

    latest_price_map: Dict[str, Optional[float]] = {}

    result = BacktestResult(
        equity_curve=equity_curve,
        benchmark_equity_curve=benchmark_curve,
        trades=trades,
        latest_prices=latest_price_map,
    )
    adv_metrics = BacktestReport(result).compute_advanced_metrics()

    allocation_pct = cfg.get("allocation_pct", 0.10) * 100
    stop_loss_pct = cfg.get("stop_loss_pct", 0.08) * 100
    ma_window = cfg.get("ma_window", 20)
    high_lookback = cfg.get("high_lookback", 252)
    rank_weight_cap = float(cfg.get("rank_weight_cap", 0.4))
    rank_weight_rs = float(cfg.get("rank_weight_rs", 0.4))
    fresh_window = int(cfg.get("fresh_new_high_window", 30))
    fresh_weight = float(cfg.get("fresh_new_high_weight", 0.2))
    penalty_window = int(cfg.get("short_term_penalty_window", 10))
    penalty_threshold = float(cfg.get("short_term_penalty_threshold", 0.12))
    penalty_weight = float(cfg.get("short_term_penalty_weight", 0.2))
    rs_lookback_days = max(90, ma_window * 6)
    max_positions = int(cfg.get("max_new_positions", 20))
    cost_pct = float(cfg.get("cost_pct", 0.1))

    strategy_description = f"""
      <p><strong>진입 조건</strong><br>
      - {high_lookback}거래일 신고가를 돌파하고 {ma_window}일 이동평균이 계산된 종목(`ready`)만 후보로 사용합니다.<br>
      - 벤치마크가 60일 이동평균 위에 있을 때만 신규 매수가 허용되며, 하루 신규 편입은 최대 {max_positions}종목입니다.
      </p>
      <p><strong>랭킹 로직</strong><br>
      - 점수 = {rank_weight_cap:.2f}·시가총액 스코어 + {rank_weight_rs:.2f}·상대강도 + {fresh_weight:.2f}·신선도 − {penalty_weight:.2f}·단기 과열 패널티<br>
      - 상대강도: 최근 {rs_lookback_days}일 동안 벤치마크 대비 초과 수익률을 사용합니다.<br>
      - 신선도: 최근 {fresh_window}일 이내 신고가 횟수가 적을수록 가산점이 붙습니다.<br>
      - 단기 과열: 최근 {penalty_window}일 수익률이 {penalty_threshold * 100:.0f}%를 넘으면 패널티를 적용합니다.
      </p>
      <p><strong>리스크 관리</strong><br>
      - 신규 진입 시 포트폴리오 자산의 {allocation_pct:.0f}% 한도로 매수하며, 손절 {stop_loss_pct:.0f}% 또는 {ma_window}일 이동평균 이탈 시 전량 매도합니다.<br>
      - 매수·매도에는 거래 비용 {cost_pct:.2f}%를 반영합니다.
      </p>
    """
    html_config = HtmlReportConfig(title="52주 신고가 백테스트 리포트")
    chart_paths: Dict[str, str] = {}
    seen_tickers: set[str] = set()
    for trade in trades:
        ticker = getattr(trade, "ticker", None)
        if not ticker or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        if on_progress:
            denom = max(1, len(trades))
            on_progress("종목별 차트 생성 중...", min(0.9, len(seen_tickers) / denom))
        try:
            chart_path = _render_new_high_ticker_chart(
                ticker=ticker,
                data_provider=data_provider,
                start_date=start_date,
                end_date=end_date,
                trades=trades,
                ma_window=ma_window,
                high_lookback=high_lookback,
            )
        except Exception as exc:
            print(f"[WARN] Failed to render chart for {ticker}: {exc}")
            continue
        chart_paths[ticker] = chart_path

    for ticker in seen_tickers:
        latest_price_map[ticker] = _get_latest_price(data_provider, ticker, end_date)

    stats_html, sell_return_series = _compute_trade_stats_and_returns(trades)

    trade_summary_html = _render_trade_summary_html(
        trades=trades,
        chart_paths=chart_paths,
        latest_price_map=latest_price_map,
        html_output_dir=html_config.output_dir,
    )
    summary_sections = ""
    if stats_html:
        summary_sections += stats_html
    if trade_summary_html:
        summary_sections += f"\n{trade_summary_html}"

    try:
        html_path = build_html_report(
            result,
            metrics=adv_metrics,
            config=html_config,
            strategy_description=strategy_description,
            per_ticker_charts=None,
            sell_return_series=sell_return_series or None,
            additional_sections=summary_sections or None,
        )
    except Exception as exc:
        error_message = f"[ERROR] Failed to build HTML report: {exc}"
        print(error_message)
        raise

    message = format_new_high_backtest_message(metrics, cfg)
    print("52주 신고가 백테스트 결과:")
    print(message)

    equity_chart_path = os.path.abspath(
        os.path.join(html_config.output_dir, html_config.equity_chart_filename)
    )
    if os.path.exists(equity_chart_path):
        print(f"Equity chart saved: {equity_chart_path}")
    else:
        print(f"[WARN] Equity chart not found: {equity_chart_path}")

    print(f"HTML report saved: {html_path}")
    if on_progress:
        on_progress("리포트 생성 완료", 1.0)
    return html_path


def run_new_high_backtest_and_notify(
    config_override: Optional[Dict[str, Any]] = None,
) -> int:
    (
        metrics,
        df_equity,
        trades,
        universe,
        data_provider,
        start_date,
        end_date,
        cfg,
    ) = run_new_high_breakout_backtest(config_override=config_override)

    send_new_high_backtest_report(
        metrics=metrics,
        df_equity=df_equity,
        trades=trades,
        universe=universe,
        data_provider=data_provider,
        start_date=start_date,
        end_date=end_date,
        cfg=cfg,
    )
    return 0
  
