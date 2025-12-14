from __future__ import annotations

import html
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import pandas as pd

from core import (
    BacktestDataProvider,
    BacktestEngine,
    BacktestExecutionEngine,
    BacktestReport,
    Portfolio,
)
from core.backtest_data import DataProvider
from core.backtest_report import HtmlReportConfig, build_html_report
from core.backtest_strategy import BaseStrategy
from core.backtest_types import BacktestResult, Order, TradeRecord
from core.chart import ChartConfig, ChartRenderer, TradeSignal
from core.portfolio import Portfolio


# SMA 백테스트 설정 (쉽게 수정 가능하도록 JSON 유사 형태로 정의)
SMA_BACKTEST_CONFIG: Dict[str, Any] = {
    # 백테스트 대상 종목
    "universe": ['252670.KS', '122630.KS'],
    # 비교 벤치마크
    "benchmark": "069500.KS",
    # 기간
    # 문자열은 ISO 형식(YYYY-MM-DD)으로 작성, end_date 를 None 으로 두면 오늘 날짜까지
    "start_date": "2015-01-01",
    "end_date": None,
    # 초기 자본
    "initial_cash": {"KRW": 10_000_000},
    # 전략 파라미터
    "short_window": 5,
    "long_window": 120,
    "risk_per_trade": 0.5,
    # 디버깅 옵션 (필요 시 사용)
    "debug": True,
}


class SmaCrossStrategy(BaseStrategy):
    """
    단순 이동평균(SMA) 크로스 전략

    - 단기 SMA가 장기 SMA를 상향 돌파하면 (골든크로스) 매수
    - 단기 SMA가 장기 SMA를 하향 돌파하면 (데드크로스) 전량 매도
    """

    def __init__(
        self,
        universe: Sequence[str],
        data_provider: DataProvider,
        short_window: int = 20,
        long_window: int = 60,
        risk_per_trade: float = 0.5,  # 보유 현금의 몇 %를 1종목에 투자할지
        cash_currency: str = "KRW",
        debug: bool = False,
        backtest_start: Optional[datetime] = None,
        backtest_end: Optional[datetime] = None,
    ) -> None:
        super().__init__(universe, data_provider, debug=debug)
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.risk_per_trade = risk_per_trade
        self.cash_currency = cash_currency
        self.debug = bool(debug)
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self._indicator_cache: Dict[str, pd.DataFrame] = {}
        self._history_margin_days = long_window * 3

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[DEBUG][SMA] {message}")

    def on_start(self, portfolio: Portfolio) -> None:
        super().on_start(portfolio)
        if self.backtest_start is None or self.backtest_end is None:
            raise ValueError("backtest_start and backtest_end must be provided for SMA strategy")

        self._indicator_cache.clear()
        history_start = self.backtest_start - timedelta(days=self._history_margin_days)
        for ticker in self.universe:
            history = self.data_provider.get_history(ticker, history_start, self.backtest_end)
            precomputed = self._prepare_indicator_frame(history)
            self._indicator_cache[ticker] = precomputed
            if self.debug:
                self._debug(
                    f"Prepared indicators for {ticker}: rows={len(precomputed)} "
                    f"history_start={history_start:%Y-%m-%d} history_end={self.backtest_end:%Y-%m-%d}"
                )

    def _prepare_indicator_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "Date" not in df.columns or "Close" not in df.columns:
            return pd.DataFrame()

        data = (
            df.copy()
            .dropna(subset=["Date", "Close"])
            .assign(Date=lambda x: pd.to_datetime(x["Date"]))
            .sort_values("Date")
            .drop_duplicates(subset=["Date"], keep="last")
            .reset_index(drop=True)
        )
        data["Close"] = data["Close"].astype(float)
        data["date"] = data["Date"].dt.normalize()
        data = data.set_index("date")

        closes = data["Close"]
        short_sma = closes.rolling(window=self.short_window).mean()
        long_sma = closes.rolling(window=self.long_window).mean()
        diff = short_sma - long_sma
        diff_prev = diff.shift(1)
        long_prev = long_sma.shift(1)
        tol_base = (
            pd.concat([long_sma.abs(), long_prev.abs()], axis=1)
            .max(axis=1)
            .fillna(1.0)
        )
        tol = tol_base * 1e-6 + 1e-9

        data["short_sma"] = short_sma
        data["long_sma"] = long_sma
        data["ready"] = long_sma.notna()
        data["golden_cross"] = ((diff_prev <= tol) & (diff > tol)).fillna(False)
        data["dead_cross"] = ((diff_prev >= -tol) & (diff < -tol)).fillna(False)
        return data

    def on_bar(self, date: datetime, portfolio: Portfolio) -> List[Order]:
        orders: List[Order] = []

        for ticker in self.universe:
            indicator_df = self._indicator_cache.get(ticker)
            if indicator_df is None or indicator_df.empty:
                self._debug(f"{ticker} has no precomputed data.")
                continue

            day_key = datetime(date.year, date.month, date.day)
            if day_key not in indicator_df.index:
                self._debug(f"{ticker} missing data on {date.date()}")
                continue

            row = indicator_df.loc[day_key]
            if not bool(row.get("ready", False)):
                self._debug(f"{ticker} insufficient data on {date.date()}")
                continue

            golden_cross = bool(row.get("golden_cross", False))
            dead_cross = bool(row.get("dead_cross", False))

            position = portfolio.positions.get(ticker)
            qty = 0.0 if position is None else float(position.quantity)

            if golden_cross:
                self._debug(f"GOLDEN CROSS {ticker} on {date.date()}")
                # 매수: 현재 현금의 risk_per_trade 만큼 이 종목에 투자
                if qty > 0:
                    self._debug(
                        f"Skip BUY {ticker} on {date.date()} - already holding qty={qty}"
                    )
                    continue

                cash_balance = float(
                    portfolio.get_cash_balances().get(self.cash_currency, 0.0)
                )
                if cash_balance <= 0:
                    self._debug(
                        f"Skip BUY {ticker} on {date.date()} - "
                        f"no {self.cash_currency} cash (cash={cash_balance})"
                    )
                    continue

                invest_amount = cash_balance * self.risk_per_trade
                price = row.get("Close")
                if price is None or pd.isna(price):
                    price = self.data_provider.get_price(ticker, date)
                if price is None or price <= 0:
                    self._debug(
                        f"Skip BUY {ticker} on {date.date()} - invalid price={price}"
                    )
                    continue

                cost_pct = 0.1
                effective_price = price * (1 + cost_pct / 100.0)
                quantity = int(invest_amount // effective_price)
                if quantity <= 0:
                    self._debug(
                        f"Skip BUY {ticker} on {date.date()} - "
                        f"invest_amount={invest_amount:.2f} < price={price:.2f}"
                    )
                    continue

                self._debug(
                    f"BUY {ticker} on {date.date()} - "
                    f"quantity={quantity} price={price:.2f} invest={invest_amount:.2f}"
                )

                orders.append(
                    Order(
                        ticker=ticker,
                        side="BUY",
                        quantity=quantity,
                        cost_pct=cost_pct,  # 예: 수수료 0.1%
                        note="SMA golden cross",
                    )
                )

            elif dead_cross and qty > 0:
                self._debug(f"DEAD CROSS {ticker} on {date.date()} qty={qty}")
                # 전량 매도
                orders.append(
                    Order(
                        ticker=ticker,
                        side="SELL",
                        quantity=qty,
                        cost_pct=0.1,
                        note="SMA dead cross",
                    )
                )

        return orders


def run_sma_crossover_backtest(
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
    """기본 SMA 크로스 전략 백테스트를 실행하고 결과를 반환하는 헬퍼 함수.

    반환값:
        metrics: BacktestReport 의 성과 지표 딕셔너리
        df_equity: 포트폴리오/벤치마크 자산 곡선 DataFrame
        trades: 포트폴리오 거래 기록 리스트
    """
    # lazy import to avoid hard dependency on pandas at import time type checking
    import pandas as pd  # type: ignore

    # 구성 가능한 설정에서 조회
    cfg: Dict[str, Any] = {**SMA_BACKTEST_CONFIG, **(config_override or {})}

    universe: Sequence[str] = cfg.get("universe", ["122630.KS", "069500.KS"])
    benchmark: str = cfg.get("benchmark", "069500.KS")

    # 날짜 파싱 (문자열 또는 None 허용)
    start_cfg = cfg.get("start_date", "2024-01-01")
    end_cfg = cfg.get("end_date")

    if isinstance(start_cfg, str):
        start_date = datetime.fromisoformat(start_cfg)
    elif isinstance(start_cfg, datetime):
        start_date = start_cfg
    else:
        raise TypeError("start_date must be a date string or datetime")

    if end_cfg is None:
        end_date = datetime.now()
    elif isinstance(end_cfg, str):
        end_date = datetime.fromisoformat(end_cfg)
    elif isinstance(end_cfg, datetime):
        end_date = end_cfg
    else:
        raise TypeError("end_date must be None, a date string, or datetime")

    # 데이터 공급자
    data_provider = BacktestDataProvider(universe=universe, benchmark=benchmark)

    # 초기 자본
    initial_cash = cfg.get("initial_cash", {"KRW": 10_000_000})
    cash_currency = cfg.get("cash_currency")
    if not cash_currency:
        if isinstance(initial_cash, dict) and initial_cash:
            cash_currency = next(iter(initial_cash.keys()))
        else:
            cash_currency = "KRW"
    portfolio = Portfolio(initial_cash=initial_cash)
    for ticker in universe:
        portfolio.add_stock(ticker, cash_currency)

    # 전략 및 체결 엔진
    strategy = SmaCrossStrategy(
        universe=universe,
        data_provider=data_provider,
        short_window=int(cfg.get("short_window", 20)),
        long_window=int(cfg.get("long_window", 60)),
        risk_per_trade=float(cfg.get("risk_per_trade", 0.5)),
        cash_currency=cash_currency,
        debug=bool(cfg.get("debug", False)),
        backtest_start=start_date,
        backtest_end=end_date,
    )
    execution_engine = BacktestExecutionEngine()

    # 백테스트 엔진
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

    # 요약 출력
    print("=== SMA Cross Backtest Summary ===")
    print(f"Period: {metrics['start']} ~ {metrics['end']}")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    if metrics["benchmark_total_return"] is not None:
        print(f"Benchmark Return: {metrics['benchmark_total_return'] * 100:.2f}%")
        print(f"Excess Return: {metrics['excess_return'] * 100:.2f}%")
    print(f"MDD: {metrics['mdd'] * 100:.2f}%")
    print(f"#Trades: {metrics['num_trades']}")

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


def _format_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(value)


def _format_pct(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_int(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "0"
    return f"{int(value):,}"


def format_sma_backtest_message(metrics: Dict[str, Any], trades: List[TradeRecord]) -> str:
    lines: List[str] = []

    lines.append("SMA Crossover Backtest Summary")
    start = metrics.get("start")
    end = metrics.get("end")
    if start is not None and hasattr(start, "strftime"):
        start_str = start.strftime("%Y-%m-%d")
    else:
        start_str = str(start)
    if end is not None and hasattr(end, "strftime"):
        end_str = end.strftime("%Y-%m-%d")
    else:
        end_str = str(end)

    lines.append(f"Period: {start_str} ~ {end_str}")
    lines.append(f"Total Return: {_format_pct(metrics.get('total_return'))}")

    benchmark_total_return = metrics.get("benchmark_total_return")
    excess_return = metrics.get("excess_return")
    if benchmark_total_return is not None:
        lines.append(f"Benchmark Return: {_format_pct(benchmark_total_return)}")
    if excess_return is not None:
        lines.append(f"Excess Return: {_format_pct(excess_return)}")

    lines.append(f"MDD: {_format_pct(metrics.get('mdd'))}")
    lines.append(f"#Trades: {metrics.get('num_trades', len(trades))}")

    return "\n".join(lines)


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def build_trade_signals_for_ticker(ticker: str, trades: List[TradeRecord]) -> List[TradeSignal]:
    """백테스트 TradeRecord 를 Chart용 TradeSignal 리스트로 변환."""
    signals: List[TradeSignal] = []
    for t in trades:
        if t.ticker != ticker:
            continue
        action = getattr(t, "action", "").upper()
        if action not in ("BUY", "SELL"):
            continue
        signal_type = "buy" if action == "BUY" else "sell"
        date = getattr(t, "date", None)
        price = getattr(t, "price", None)
        if not isinstance(date, datetime) or price is None:
            continue
        signals.append(TradeSignal(date=date, price=float(price), type=signal_type))
    return signals


def render_ticker_chart(
    ticker: str,
    data_provider: BacktestDataProvider,
    start_date: datetime,
    end_date: datetime,
    trades: List[TradeRecord],
    short_window: int,
    long_window: int,
) -> str:
    """지정된 티커에 대해 가격 + 매수/매도 신호 + SMA 라인이 포함된 차트 이미지를 생성하고 경로를 반환."""
    df_price = data_provider.get_history(ticker, start_date, end_date)
    if df_price.empty:
        raise RuntimeError(f"No price data for ticker {ticker}")

    trade_signals = build_trade_signals_for_ticker(ticker, trades)

    # Date 인덱스를 기준으로 SMA 계산 (ChartRenderer 의 indicators_data 포맷에 맞춤)
    df_sma = df_price.copy()
    if "Date" in df_sma.columns:
        df_sma["Date"] = pd.to_datetime(df_sma["Date"])
        df_sma = df_sma.set_index("Date")
    closes = df_sma["Close"].astype(float)

    sma_short = closes.rolling(window=short_window).mean().rename(f"sma_{short_window}")
    sma_long = closes.rolling(window=long_window).mean().rename(f"sma_{long_window}")

    indicators_data = {
        f"sma_{short_window}": sma_short,
        f"sma_{long_window}": sma_long,
    }

    config = ChartConfig(
        show_trade_signals=True,
        show_sma=True,
        show_volume=True,
        sma_windows=[short_window, long_window],
    )
    renderer = ChartRenderer(config=config)

    safe_ticker = ticker.replace(".", "_")
    save_path = os.path.join("output", f"sma_{safe_ticker}.png")
    _ensure_output_dir(save_path)

    out_path = renderer.render(
        df=df_price,
        indicators_data=indicators_data,
        trade_signals=trade_signals,
        display_days=None,
        title=f"{ticker} SMA({short_window}/{long_window}) Backtest",
        save_path=save_path,
    )
    return out_path


def send_sma_backtest_report(
    metrics: Dict[str, Any],
    df_equity: "pd.DataFrame",
    trades: List[TradeRecord],
    universe: Sequence[str],
    data_provider: BacktestDataProvider,
    start_date: datetime,
    end_date: datetime,
    cfg: Dict[str, Any],
) -> None:
    """백테스트 리포트를 생성."""
    if not isinstance(data_provider, BacktestDataProvider):
        raise TypeError("Expected BacktestDataProvider instance for report generation")

    message = format_sma_backtest_message(metrics, trades)
    print("SMA backtest summary:")
    print(message)

    equity_curve = df_equity["equity"].dropna()
    benchmark_curve = None
    if "benchmark" in df_equity.columns:
        bench_series = df_equity["benchmark"].dropna()
        if not bench_series.empty:
            benchmark_curve = bench_series

    if isinstance(universe, (list, tuple)):
        tickers: Sequence[str] = universe
    else:
        tickers = [str(universe)]

    latest_prices: Dict[str, float] = {}
    for ticker in tickers:
        for offset in range(10):
            target_date = end_date - timedelta(days=offset)
            price = data_provider.get_price(ticker, target_date)
            if price is not None:
                latest_prices[ticker] = float(price)
                break

    result = BacktestResult(
        equity_curve=equity_curve,
        benchmark_equity_curve=benchmark_curve,
        trades=trades,
        latest_prices=latest_prices or None,
    )

    adv_metrics = BacktestReport(result).compute_advanced_metrics()
    html_config = HtmlReportConfig(title="SMA 백테스트 리포트")

    if cfg.get("debug"):
        total_return = adv_metrics.get("total_return")
        total_return_pct = adv_metrics.get("total_return_pct")
        print(
            f"[DEBUG] Portfolio total return: "
            f"{total_return:.6f} ({total_return_pct:+.2f}%)"
        )

        eq_series = result.equity_curve
        print("[DEBUG] Equity around all trades:")
        for t in trades:
            d = t.date
            eq_on_date = eq_series.loc[eq_series.index.date == d.date()]
            eq_val = float(eq_on_date.iloc[0]) if not eq_on_date.empty else float("nan")
            print(
                f"  {d.strftime('%Y-%m-%d')} {t.action} {t.ticker} "
                f"qty={t.quantity} price={t.price} total={t.total} -> equity={eq_val:,.2f}"
            )

    per_ticker_charts: Dict[str, str] = {}
    short_window = int(cfg.get("short_window", 20))
    long_window = int(cfg.get("long_window", 60))

    for ticker in tickers:
        try:
            chart_path = render_ticker_chart(
                ticker=ticker,
                data_provider=data_provider,
                start_date=start_date,
                end_date=end_date,
                trades=trades,
                short_window=short_window,
                long_window=long_window,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to render chart for {ticker}: {exc}")
            continue

        per_ticker_charts[ticker] = chart_path

    universe_str = ", ".join(str(t) for t in cfg.get("universe", []))
    benchmark = cfg.get("benchmark", "N/A")
    risk_per_trade = float(cfg.get("risk_per_trade", 0.5)) * 100

    strategy_description = f"""
      <p><strong>전략 개요</strong><br>
      단순 이동평균(SMA) {short_window}일선과 {long_window}일선을 이용한 크로스오버 추세 추종 전략입니다.
      단기 SMA가 장기 SMA를 상향 돌파할 때 매수하고, 하향 돌파할 때 전량 매도하는 것을 기본 원칙으로 합니다.</p>

      <p><strong>대상 종목 및 벤치마크</strong></p>
      <ul>
        <li><strong>대상 종목(universe)</strong>: {universe_str}</li>
        <li><strong>벤치마크</strong>: {benchmark} (단순 Buy &amp; Hold 수익률과 비교)</li>
      </ul>

      <p><strong>매수/매도 규칙</strong></p>
      <ul>
        <li><strong>매수 조건 (골든크로스)</strong>: 전일 기준으로 단기 SMA ≤ 장기 SMA 이고, 오늘 단기 SMA &gt; 장기 SMA 로 바뀐 경우
          <ul>
            <li>해당 종목을 보유하고 있지 않을 때만 진입</li>
            <li>포트폴리오 내 KRW 현금의 약 {risk_per_trade:.1f}% 를 투자 금액으로 사용</li>
          </ul>
        </li>
        <li><strong>매도 조건 (데드크로스)</strong>: 전일 기준으로 단기 SMA ≥ 장기 SMA 이고, 오늘 단기 SMA &lt; 장기 SMA 로 바뀐 경우
          <ul>
            <li>보유 수량 전체를 시장가 기준으로 매도</li>
          </ul>
        </li>
      </ul>

      <p><strong>리스크 관리</strong><br>
      각 종목별 진입 시점마다 현금의 일정 비율({risk_per_trade:.1f}%)만 투자함으로써, 과도한 집중을 방지하고
      여러 종목에 분산 투자할 수 있도록 설계되어 있습니다.</p>
    """

    html_path = build_html_report(
        result,
        metrics=adv_metrics,
        config=html_config,
        strategy_description=strategy_description,
        per_ticker_charts=per_ticker_charts,
    )
    equity_chart_path = os.path.join(html_config.output_dir, html_config.equity_chart_filename)

    print(f"Portfolio equity chart saved: {equity_chart_path}")
    if os.path.exists(equity_chart_path):
        print(f"Equity chart available at: {equity_chart_path}")

    print(f"HTML backtest report saved: {html_path}")


def _collect_sma_sweep_results(
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    extra_config: Optional[Dict[str, Any]] = None,
) -> Tuple["pd.DataFrame", Optional[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    best_return = float("-inf")

    for short_window in short_windows:
        for long_window in long_windows:
            if short_window >= long_window:
                print(
                    f"[WARN] Skip SMA combo short={short_window}, long={long_window} "
                    f"(short must be smaller than long)"
                )
                continue

            cfg_override: Dict[str, Any] = {
                "short_window": int(short_window),
                "long_window": int(long_window),
            }
            if extra_config:
                cfg_override.update(extra_config)

            try:
                (
                    metrics,
                    df_equity,
                    trades,
                    universe,
                    data_provider,
                    start_date,
                    end_date,
                    cfg,
                ) = run_sma_crossover_backtest(cfg_override)
            except Exception as exc:
                print(
                    f"[ERROR] SMA backtest failed for short={short_window}, "
                    f"long={long_window}: {exc}"
                )
                continue

            row = {
                "short_window": short_window,
                "long_window": long_window,
                "total_return": metrics.get("total_return"),
                "benchmark_total_return": metrics.get("benchmark_total_return"),
                "mdd": metrics.get("mdd"),
                "num_trades": metrics.get("num_trades"),
            }
            rows.append(row)

            total_return_value = row["total_return"]
            if total_return_value is not None and total_return_value > best_return:
                best_return = float(total_return_value)
                best_payload = {
                    "metrics": metrics,
                    "df_equity": df_equity,
                    "trades": trades,
                    "universe": universe,
                    "data_provider": data_provider,
                    "start_date": start_date,
                    "end_date": end_date,
                    "cfg": cfg,
                }

    if rows:
        results_df = pd.DataFrame(rows).sort_values(
            by="total_return", ascending=False, na_position="last"
        )
        results_df = results_df.reset_index(drop=True)
    else:
        results_df = pd.DataFrame(
            columns=["short_window", "long_window", "total_return", "mdd", "num_trades"]
        )

    return results_df, best_payload


def _build_sweep_plain_summary(results_df: "pd.DataFrame", limit: int = 10) -> str:
    lines: List[str] = ["[SMA Sweep Summary]"]
    if results_df.empty:
        lines.append("no successful runs")
        return "\n".join(lines)

    head_df = results_df.head(limit)
    for _, row in head_df.iterrows():
        short_window = int(row["short_window"])
        long_window = int(row["long_window"])
        total_return = _format_pct(row.get("total_return"))
        benchmark_return = _format_pct(row.get("benchmark_total_return"))
        mdd_pct = _format_pct(row.get("mdd"))
        trades_value = row.get("num_trades")
        trades = 0 if trades_value is None or pd.isna(trades_value) else int(trades_value)
        lines.append(
            f"short={short_window} long={long_window} "
            f"total={total_return} benchmark={benchmark_return} "
            f"mdd={mdd_pct} trades={trades}"
        )

    best_row = results_df.iloc[0]
    lines.append(
        "best short={short} long={long} total={ret}".format(
            short=int(best_row["short_window"]),
            long=int(best_row["long_window"]),
            ret=_format_pct(best_row.get("total_return")),
        )
    )
    return "\n".join(lines)



def send_sma_sweep_html_report(
    results_df: "pd.DataFrame",
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    extra_config: Optional[Dict[str, Any]] = None,
    *,
    best_payload: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join("output", f"sma_sweep_summary_{timestamp}.html")
    _ensure_output_dir(file_path)

    cfg_items = extra_config or {}
    cfg_lines = "".join(
        f"<li><strong>{html.escape(str(k))}</strong>: {html.escape(str(v))}</li>"
        for k, v in cfg_items.items()
    )

    html_lines = [
        "<html><head><meta charset='utf-8'><title>SMA Sweep Summary</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 16px; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
        "th { background-color: #f2f2f2; }",
        ".best { background-color: #e6ffe6; }",
        "</style></head><body>",
        "<h1>SMA Parameter Sweep Summary</h1>",
        f"<p><strong>Generated:</strong> {timestamp}</p>",
        "<p><strong>Short windows:</strong> "
        + ", ".join(html.escape(str(v)) for v in short_windows)
        + "</p>",
        "<p><strong>Long windows:</strong> "
        + ", ".join(html.escape(str(v)) for v in long_windows)
        + "</p>",
    ]
    if cfg_lines:
        html_lines.append("<p><strong>Extra config</strong></p><ul>")
        html_lines.append(cfg_lines)
        html_lines.append("</ul>")

    if results_df.empty:
        html_lines.append("<p>No successful runs.</p>")
    else:
        columns = [
            ("short_window", "Short"),
            ("long_window", "Long"),
            ("total_return", "Total Return"),
            ("benchmark_total_return", "Benchmark Return"),
            ("mdd", "MDD"),
            ("num_trades", "#Trades"),
        ]
        html_lines.append("<table><thead><tr>")
        for _, title in columns:
            html_lines.append(f"<th>{html.escape(title)}</th>")
        html_lines.append("</tr></thead><tbody>")
        for idx, row in results_df.reset_index(drop=True).iterrows():
            row_class = "best" if idx == 0 else ""
            html_lines.append(f"<tr class='{row_class}'>")
            html_lines.append(f"<td>{int(row['short_window'])}</td>")
            html_lines.append(f"<td>{int(row['long_window'])}</td>")
            html_lines.append(f"<td>{html.escape(_format_pct(row.get('total_return')))}</td>")
            html_lines.append(
                f"<td>{html.escape(_format_pct(row.get('benchmark_total_return')))}</td>"
            )
            html_lines.append(f"<td>{html.escape(_format_pct(row.get('mdd')))}</td>")
            html_lines.append(f"<td>{html.escape(_format_int(row.get('num_trades')))}</td>")
            html_lines.append("</tr>")
        html_lines.append("</tbody></table>")

    if best_payload is not None:
        cfg = best_payload.get("cfg", {})
        short_window = cfg.get("short_window")
        long_window = cfg.get("long_window")
        html_lines.append("<h2>Best Combination</h2>")
        html_lines.append(
            "<p>"
            f"short={html.escape(str(short_window))}, "
            f"long={html.escape(str(long_window))}"
            "</p>"
        )

    html_lines.append("</body></html>")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))

    print(f"SMA sweep HTML summary saved: {file_path}")
    return file_path


def run_sma_parameter_sweep_and_notify(
    short_windows: Sequence[int],
    long_windows: Sequence[int],
    extra_config: Optional[Dict[str, Any]] = None,
    summary_limit: int = 10,
) -> int:
    print('Y:run_sma_parameter_sweep_and_notify')
    short_windows = list(short_windows)
    long_windows = list(long_windows)
    if not short_windows or not long_windows:
        raise ValueError("short_windows and long_windows must contain at least one value")
    print('Y: sweep start')
    results_df, best_payload = _collect_sma_sweep_results(
        short_windows=short_windows,
        long_windows=long_windows,
        extra_config=extra_config,
    )
    print('Y')
    print(results_df)
    print(best_payload)
    summary_message = _build_sweep_plain_summary(results_df, limit=summary_limit)
    print("SMA Sweep Summary:")
    print(summary_message)
    send_sma_sweep_html_report(
        results_df=results_df,
        short_windows=short_windows,
        long_windows=long_windows,
        extra_config=extra_config,
        best_payload=best_payload,
    )

    if best_payload is None:
        print("[WARN] No successful SMA backtest results to report.")
        return 1

    send_sma_backtest_report(**best_payload)
    return 0

def run_sma_backtest_and_notify(
    config_override: Optional[Dict[str, Any]] = None,
) -> int:
    """SMA 백테스트를 실행하고, 리포트/차트를 생성."""
    (
        metrics,
        df_equity,
        trades,
        universe,
        data_provider,
        start_date,
        end_date,
        cfg,
    ) = run_sma_crossover_backtest(config_override=config_override)

    send_sma_backtest_report(
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

