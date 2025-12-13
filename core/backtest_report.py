from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import base64
import math
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import os
import pandas as pd
from collections import deque
from dataclasses import dataclass

from .backtest_types import BacktestResult
from .stock import Stock


def _resolve_ticker_name(ticker: str) -> str:
    try:
        stock_obj = Stock(ticker)
        return stock_obj.name or ticker
    except Exception:
        return ticker


def _resolve_market_label(ticker: str) -> str:
    if not ticker:
        return "기타"
    normalized = ticker.upper()
    if normalized.endswith(".KS"):
        return "KOSPI"
    if normalized.endswith(".KQ"):
        return "KOSDAQ"
    return "기타"


def _format_currency(value: float, currency: Optional[str]) -> str:
    symbol = "₩" if (currency or "").upper() == "KRW" else ""
    if math.isnan(value) or math.isinf(value):
        return "-"
    if abs(value) >= 1000:
        formatted = f"{symbol}{value:,.0f}"
    else:
        formatted = f"{symbol}{value:,.2f}"
    if not symbol and currency:
        formatted = f"{formatted} {currency}"
    return formatted


def _format_price_cell(unit_price: float, quantity: float, currency: Optional[str]) -> str:
    unit_display = _format_currency(unit_price, currency)
    total_display = _format_currency(unit_price * quantity, currency)
    return f"{unit_display}<br><small>총 {total_display}</small>"


def _trade_unit_cost(trade: TradeRecord, fallback_price: float) -> float:
    quantity = float(getattr(trade, "quantity", 0.0))
    total_value = float(getattr(trade, "total", 0.0))
    if quantity > 0 and total_value:
        return abs(total_value) / quantity
    return fallback_price


class BacktestReport:
    """백테스트 결과 리포트 생성기."""

    def __init__(self, result: BacktestResult) -> None:
        self.result = result

    def compute_advanced_metrics(self, risk_free_rate: float = 0.03) -> Dict[str, Any]:
        """백테스트 성과 지표를 상세하게 계산."""
        equity = self.result.equity_curve
        if equity.empty:
            raise RuntimeError("Equity curve 가 비어 있습니다.")

        # 기본 값 및 기간
        equity = equity.astype(float)
        start_value = float(equity.iloc[0])
        end_value = float(equity.iloc[-1])
        if start_value <= 0:
            raise RuntimeError("Equity curve 의 시작 값이 0 이하입니다.")

        dates = pd.to_datetime(equity.index)
        days = (dates[-1] - dates[0]).days
        years = days / 365.25 if days > 0 else 0.0

        # 1) 수익률 지표
        total_return = end_value / start_value - 1.0
        total_return_pct = total_return * 100.0

        cagr = (end_value / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        cagr_pct = cagr * 100.0

        # 2) 일간 수익률 / 변동성
        daily_returns = equity.pct_change().dropna().to_numpy()
        volatility = float(np.std(daily_returns) * np.sqrt(252)) if daily_returns.size > 0 else 0.0
        volatility_pct = volatility * 100.0

        # 3) MDD
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        mdd = float(drawdown.min())
        mdd_pct = mdd * 100.0

        # 4) Sharpe / Calmar / Recovery
        sharpe_ratio = 0.0
        if volatility > 0:
            excess_return = cagr - risk_free_rate
            sharpe_ratio = excess_return / volatility

        calmar_ratio = cagr / abs(mdd) if mdd < 0 else 0.0
        recovery_factor = total_return_pct / abs(mdd_pct) if mdd_pct < 0 else 0.0

        metrics: Dict[str, Any] = {
            "start": equity.index[0],
            "end": equity.index[-1],
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "cagr": cagr,
            "cagr_pct": cagr_pct,
            "volatility": volatility,
            "volatility_pct": volatility_pct,
            "mdd": mdd,
            "mdd_pct": mdd_pct,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "recovery_factor": recovery_factor,
            "num_trades": len(self.result.trades),
        }

        # 5) 벤치마크 대비 성과
        bench_curve = self.result.benchmark_equity_curve
        if bench_curve is not None and not bench_curve.empty:
            bench = bench_curve.astype(float)

            # 인덱스 교집합 기준으로 정렬
            common_index = equity.index.intersection(bench.index)
            if not common_index.empty:
                eq = equity.loc[common_index]
                be = bench.loc[common_index]
            else:
                eq = equity
                be = bench.reindex_like(equity).ffill().dropna()

            if not be.empty and be.iloc[0] > 0:
                b_start = float(be.iloc[0])
                b_end = float(be.iloc[-1])
                b_total_return = b_end / b_start - 1.0
                b_total_return_pct = b_total_return * 100.0

                b_days = (pd.to_datetime(be.index[-1]) - pd.to_datetime(be.index[0])).days
                b_years = b_days / 365.25 if b_days > 0 else 0.0
                b_cagr = (b_end / b_start) ** (1.0 / b_years) - 1.0 if b_years > 0 else 0.0
                b_cagr_pct = b_cagr * 100.0

                b_daily_returns = be.pct_change().dropna().to_numpy()
                b_volatility = float(np.std(b_daily_returns) * np.sqrt(252)) if b_daily_returns.size > 0 else 0.0
                b_volatility_pct = b_volatility * 100.0

                b_running_max = be.cummax()
                b_drawdown = be / b_running_max - 1.0
                b_mdd = float(b_drawdown.min())
                b_mdd_pct = b_mdd * 100.0

                # 초과 수익/IR 등
                excess_return_pct = total_return_pct - b_total_return_pct
                excess_cagr_pct = cagr_pct - b_cagr_pct
                information_ratio = 0.0
                if volatility > 0:
                    information_ratio = excess_cagr_pct / volatility_pct

                metrics.update(
                    {
                        "benchmark_total_return": b_total_return,
                        "benchmark_total_return_pct": b_total_return_pct,
                        "benchmark_cagr": b_cagr,
                        "benchmark_cagr_pct": b_cagr_pct,
                        "benchmark_volatility": b_volatility,
                        "benchmark_volatility_pct": b_volatility_pct,
                        "benchmark_mdd": b_mdd,
                        "benchmark_mdd_pct": b_mdd_pct,
                        "excess_return": total_return - b_total_return,
                        "excess_return_pct": excess_return_pct,
                        "excess_cagr_pct": excess_cagr_pct,
                        "information_ratio": information_ratio,
                    }
                )

        return metrics

    def compute_metrics(self) -> Dict[str, Any]:
        """총 수익률, 초과 수익률, MDD 등의 핵심 지표 계산.

        기존 간단 지표 인터페이스를 유지하면서,
        내부적으로는 `compute_advanced_metrics` 결과를 사용합니다.
        """
        adv = self.compute_advanced_metrics()
        return {
            "start": adv["start"],
            "end": adv["end"],
            "total_return": adv["total_return"],
            "mdd": adv["mdd"],
            "benchmark_total_return": adv.get("benchmark_total_return"),
            "excess_return": adv.get("excess_return"),
            "num_trades": adv["num_trades"],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """자산 곡선과 (있다면) 벤치마크 곡선을 하나의 DataFrame 으로 반환."""
        df = pd.DataFrame({"equity": self.result.equity_curve})
        if self.result.benchmark_equity_curve is not None:
            bench = self.result.benchmark_equity_curve.rename("benchmark")
            df = df.join(bench, how="outer")
        return df


@dataclass
class HtmlReportConfig:
    """HTML 백테스트 리포트 및 포트폴리오 차트 설정."""

    output_dir: str = "output"
    equity_chart_filename: str = "portfolio_equity_chart.png"
    html_filename: str = "backtest_report.html"
    title: str = "백테스트 리포트"
    include_benchmark: bool = True


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _configure_korean_font() -> None:
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "NanumBarunGothic",
        "Batang",
        "Dotum",
        "Gulim",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "DejaVu Sans",
    ]
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            mpl.rcParams["font.family"] = font_name
            mpl.rcParams["axes.unicode_minus"] = False
            return
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.unicode_minus"] = False


def build_equity_chart(
    result: BacktestResult,
    config: HtmlReportConfig,
    sell_returns: Optional[List[Tuple[datetime, float]]] = None,
) -> str:
    """포트폴리오 vs 벤치마크(있다면) 누적 수익률 차트를 생성하고 파일 경로를 반환.

    y축은 초기값 기준 0% 에서 시작하는 누적 수익률(%)입니다.
    """

    equity = result.equity_curve
    if equity.empty:
        raise RuntimeError("Equity curve 가 비어 있습니다.")

    # 정렬 및 누적 수익률(%) 계산 (기준 0%)
    eq = equity.sort_index().astype(float)
    start_val = float(eq.iloc[0])
    if start_val == 0:
        raise RuntimeError("Equity curve 시작 값이 0 입니다.")
    eq_norm = (eq / start_val - 1.0) * 100.0

    bench_norm = None
    if config.include_benchmark and result.benchmark_equity_curve is not None:
        bench = result.benchmark_equity_curve.dropna()
        if not bench.empty and bench.iloc[0] != 0:
            bench = bench.sort_index().astype(float)
            b_start = float(bench.iloc[0])
            bench_norm = (bench / b_start - 1.0) * 100.0

    _ensure_dir(config.output_dir)
    chart_path = os.path.join(config.output_dir, config.equity_chart_filename)

    _configure_korean_font()

    if sell_returns:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 7.5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None

    ax1.plot(eq_norm.index, eq_norm.values, label="Portfolio", color="tab:blue", linewidth=2.0)
    if bench_norm is not None:
        ax1.plot(bench_norm.index, bench_norm.values, label="Benchmark", color="tab:orange", linewidth=1.8)

    ax1.set_title("포트폴리오 vs 벤치마크 누적 수익률")
    ax1.set_ylabel("누적 수익률 (%)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if sell_returns and ax2 is not None:
        sell_returns_sorted = sorted(sell_returns, key=lambda x: x[0])
        dates = [item[0] for item in sell_returns_sorted]
        values = [item[1] * 100.0 for item in sell_returns_sorted]
        colors = ["tab:green" if val >= 0 else "tab:red" for val in values]
        ax2.bar(dates, values, color=colors, alpha=0.8)
        ax2.axhline(0, color="#666666", linewidth=0.8)
        ax2.set_ylabel("매도 수익률 (%)")
        ax2.grid(alpha=0.2, axis="y")

    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    return os.path.abspath(chart_path)


def build_html_report(
    result: BacktestResult,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[HtmlReportConfig] = None,
    strategy_description: Optional[str] = None,
    per_ticker_charts: Optional[Dict[str, str]] = None,
    sell_return_series: Optional[List[Tuple[datetime, float]]] = None,
    additional_sections: Optional[str] = None,
) -> str:
    """포트폴리오 성과 요약 + 차트를 포함한 HTML 리포트 생성."""

    if config is None:
        config = HtmlReportConfig()

    _ensure_dir(config.output_dir)

    # 메트릭 계산
    if metrics is None:
        metrics = BacktestReport(result).compute_advanced_metrics()

    latest_prices: Dict[str, float] = getattr(result, "latest_prices", None) or {}

    # 차트 생성
    start = metrics.get("start")
    end = metrics.get("end")
    start_str = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
    end_str = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)

    total_return = metrics.get("total_return_pct", metrics.get("total_return", 0.0))
    cagr = metrics.get("cagr_pct", metrics.get("cagr", 0.0))
    mdd = metrics.get("mdd_pct", metrics.get("mdd", 0.0))
    volatility = metrics.get("volatility_pct", metrics.get("volatility", 0.0))
    sharpe = metrics.get("sharpe_ratio", 0.0)
    calmar = metrics.get("calmar_ratio", 0.0)

    b_total = metrics.get("benchmark_total_return_pct") or metrics.get("benchmark_total_return")
    excess = metrics.get("excess_return_pct") or metrics.get("excess_return")

    def _fmt_pct(value: Optional[float]) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.2f}%"

    def summarize_trades(trades: List[TradeRecord]) -> Tuple[str, str, List[Tuple[datetime, float]]]:
        buy_queue: deque[Tuple[float, float]] = deque()
        success_returns: List[float] = []
        failure_returns: List[float] = []
        sell_returns_by_date: defaultdict[datetime, List[float]] = defaultdict(list)
        for trade in sorted(trades, key=lambda t: getattr(t, "date", datetime.min)):
            action = getattr(trade, "action", "").upper()
            qty = float(getattr(trade, "quantity", 0.0))
            if qty <= 0:
                continue
            if action == "BUY":
                price = float(getattr(trade, "price", 0.0))
                if price > 0:
                    buy_queue.append((price, qty))
                continue
            if action != "SELL":
                continue
            remaining = qty
            sell_price = float(getattr(trade, "price", 0.0))
            sell_date = getattr(trade, "date", None) or datetime.min
            while remaining > 1e-9 and buy_queue:
                buy_price, buy_remaining = buy_queue[0]
                match_qty = min(buy_remaining, remaining)
                if buy_price > 0:
                    pnl_pct = (sell_price - buy_price) / buy_price
                    if pnl_pct >= 0.24:
                        success_returns.append(pnl_pct)
                    else:
                        failure_returns.append(pnl_pct)
                    sell_returns_by_date[sell_date].append(pnl_pct)
                buy_remaining -= match_qty
                remaining -= match_qty
                if buy_remaining <= 1e-9:
                    buy_queue.popleft()
                else:
                    buy_queue[0] = (buy_price, buy_remaining)
        total = len(success_returns) + len(failure_returns)
        stats_section = ""
        if total > 0:
            success_avg_pct = (sum(success_returns) / len(success_returns)) * 100 if success_returns else 0.0
            failure_avg_pct = (sum(failure_returns) / len(failure_returns)) * 100 if failure_returns else 0.0
            stats_section = f"""

    <h2>거래 통계</h2>
    <div class="grid">
      <div class="card">
        <div class="card-label">성공 거래 비율 (≥ 24%)</div>
        <div class="card-value">{(len(success_returns) / total) * 100:.2f}% ({len(success_returns)})</div>
      </div>
      <div class="card">
        <div class="card-label">실패 거래 비율</div>
        <div class="card-value">{(len(failure_returns) / total) * 100:.2f}% ({len(failure_returns)})</div>
      </div>
      <div class="card">
        <div class="card-label">성공 거래 평균 수익률</div>
        <div class="card-value">{_fmt_pct(success_avg_pct)}</div>
      </div>
      <div class="card">
        <div class="card-label">실패 거래 평균 수익률</div>
        <div class="card-value">{_fmt_pct(failure_avg_pct)}</div>
      </div>
    </div>"""

        trades_section = ""
        if trades:
            sorted_trades = sorted(trades, key=lambda t: getattr(t, "date", None) or datetime.min)
            trade_rows = []
            for t in sorted_trades:
                date_str = t.date.strftime("%Y-%m-%d") if getattr(t, "date", None) else "n/a"
                qty = getattr(t, "quantity", 0.0)
                price = getattr(t, "price", 0.0)
                total_value = getattr(t, "total", 0.0)
                cost_pct = getattr(t, "cost_pct", 0.0)
                portfolio_value = getattr(t, "portfolio_value", None)
                portfolio_value_cell = (
                    f"<td>{portfolio_value:,.2f}</td>" if portfolio_value is not None else "<td>n/a</td>"
                )
                trade_rows.append(
                    f"<tr>"
                    f"<td>{date_str}</td>"
                    f"<td>{getattr(t, 'action', '')}</td>"
                    f"<td>{getattr(t, 'ticker', '')}</td>"
                    f"<td>{qty:,.4f}</td>"
                    f"<td>{price:,.2f}</td>"
                    f"<td>{cost_pct:.2f}%</td>"
                    f"<td>{total_value:,.2f}</td>"
                    f"{portfolio_value_cell}"
                    f"</tr>"
                )
            trades_html = "\n".join(trade_rows)
            trades_section = f"""

    <h2>매수/매도 기록</h2>
    <table class="trades">
      <thead>
        <tr>
          <th>날짜</th>
          <th>액션</th>
          <th>티커</th>
          <th>수량</th>
          <th>가격</th>
          <th>수수료(%)</th>
          <th>거래금액</th>
          <th>총 평가금액</th>
        </tr>
      </thead>
      <tbody>
        {trades_html}
      </tbody>
    </table>"""

        sell_returns = sorted(
            (date, sum(values) / len(values)) for date, values in sell_returns_by_date.items()
        )

        return stats_section, trades_section, sell_returns

    trades_section = ""
    trade_pairs_section = ""
    trades = getattr(result, "trades", None)
    if trades:
        pair_rows: List[str] = []
        buy_queues: Dict[str, deque[Dict[str, Any]]] = {}
        sorted_trades = sorted(trades, key=lambda t: getattr(t, "date", None) or datetime.min)

        for trade in sorted_trades:
            action = getattr(trade, "action", "").upper()
            qty = float(getattr(trade, "quantity", 0.0))
            if qty <= 0:
                continue

            if action == "BUY":
                ticker = getattr(trade, "ticker", "")
                if not ticker:
                    continue
                queue = buy_queues.setdefault(ticker, deque())
                queue.append(
                    {
                        "trade": trade,
                        "remaining": qty,
                        "unit_cost": _trade_unit_cost(trade, float(getattr(trade, "price", 0.0))),
                    }
                )
                continue

            if action != "SELL":
                continue

            ticker = getattr(trade, "ticker", "")
            if not ticker:
                continue
            queue = buy_queues.get(ticker)
            if not queue:
                continue

            remaining = qty
            sell_date = getattr(trade, "date", None)
            sell_price = float(getattr(trade, "price", 0.0))
            sell_total = float(getattr(trade, "total", 0.0))
            sell_currency = getattr(trade, "currency", "")
            sell_unit_net = sell_total / qty if qty > 0 and sell_total else sell_price

            total_buy_cost = 0.0
            first_buy_trade: Optional[TradeRecord] = None
            matched_currency = sell_currency

            while remaining > 1e-9 and queue:
                lot = queue[0]
                lot_remaining = float(lot["remaining"])
                match_qty = min(lot_remaining, remaining)
                if match_qty <= 0:
                    queue.popleft()
                    continue

                lot_trade = lot["trade"]
                lot_unit_cost = float(lot["unit_cost"])
                total_buy_cost += lot_unit_cost * match_qty
                remaining -= match_qty
                lot["remaining"] = lot_remaining - match_qty

                if first_buy_trade is None:
                    first_buy_trade = lot_trade
                    matched_currency = getattr(lot_trade, "currency", sell_currency)

                if lot["remaining"] <= 1e-9:
                    queue.popleft()

            if remaining > 1e-6 or total_buy_cost <= 0:
                continue

            name = _resolve_ticker_name(ticker)
            market_label = _resolve_market_label(ticker)
            buy_date_str = (
                first_buy_trade.date.strftime("%Y-%m-%d")
                if first_buy_trade and getattr(first_buy_trade, "date", None)
                else "n/a"
            )
            sell_date_str = sell_date.strftime("%Y-%m-%d") if sell_date else "n/a"

            buy_unit_avg = total_buy_cost / qty if qty > 0 else 0.0
            pnl_pct = ((sell_total - total_buy_cost) / total_buy_cost * 100.0) if total_buy_cost > 0 else 0.0

            pair_rows.append(
                f"<tr>"
                f"<td>{buy_date_str}</td>"
                f"<td>{name} ({ticker})</td>"
                f"<td>{market_label}</td>"
                f"<td>{qty:,.2f}</td>"
                f"<td>{_format_price_cell(buy_unit_avg, qty, matched_currency)}</td>"
                f"<td>{sell_date_str}</td>"
                f"<td>{_format_price_cell(sell_unit_net, qty, sell_currency or matched_currency)}</td>"
                f"<td>{pnl_pct:+.2f}%</td>"
                f"<td>완료</td>"
                f"</tr>"
            )

        for ticker, queue in buy_queues.items():
            while queue:
                lot = queue.popleft()
                remaining_qty = float(lot["remaining"])
                if remaining_qty <= 0:
                    continue
                trade = lot["trade"]
                name = _resolve_ticker_name(ticker)
                market_label = _resolve_market_label(ticker)
                buy_date_str = (
                    trade.date.strftime("%Y-%m-%d") if getattr(trade, "date", None) else "n/a"
                )
                currency = getattr(trade, "currency", "")
                unit_cost = float(lot["unit_cost"])
                current_price = latest_prices.get(ticker)
                pnl_pct = 0.0
                status_text = "보유 중"
                if unit_cost > 0 and current_price is not None:
                    pnl_pct = (float(current_price) - unit_cost) / unit_cost * 100.0
                    valuation = float(current_price) * remaining_qty
                    status_text = f"보유 중 (평가액 {_format_currency(valuation, currency)})"

                pair_rows.append(
                    f"<tr>"
                    f"<td>{buy_date_str}</td>"
                    f"<td>{name} ({ticker})</td>"
                    f"<td>{market_label}</td>"
                    f"<td>{remaining_qty:,.2f}</td>"
                    f"<td>{_format_price_cell(unit_cost, remaining_qty, currency)}</td>"
                    f"<td>보유 중</td>"
                    f"<td>-</td>"
                    f"<td>{pnl_pct:+.2f}%</td>"
                    f"<td>{status_text}</td>"
                    f"</tr>"
                )

        if pair_rows:
            rows_html = "\n".join(pair_rows)
            trade_pairs_section = f"""

    <h2>매수/매도 기록 (매수-매도 쌍)</h2>
    <table class="trades">
      <thead>
        <tr>
          <th>매수일</th>
          <th>종목</th>
          <th>시장</th>
          <th>수량</th>
          <th>매수단가</th>
          <th>매도일</th>
          <th>매도단가</th>
          <th>수익률</th>
          <th>상태/평가액</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>"""

    equity_chart_path = build_equity_chart(result, config, sell_return_series)
    chart_filename = os.path.basename(equity_chart_path)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>{config.title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
      margin: 0;
      padding: 24px;
      background-color: #f5f5f5;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      padding: 24px 28px 32px 28px;
    }}
    h1 {{
      margin-top: 0;
      color: #111827;
      font-size: 24px;
    }}
    h2 {{
      margin-top: 24px;
      color: #111827;
      font-size: 18px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 4px;
    }}
    .meta {{
      color: #6b7280;
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }}
    .card {{
      background: #f9fafb;
      border-radius: 6px;
      padding: 10px 12px;
    }}
    .card-label {{
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 4px;
    }}
    .card-value {{
      font-size: 16px;
      font-weight: 600;
      color: #111827;
    }}
    .card-desc {{
      margin-top: 2px;
      font-size: 11px;
      color: #4b5563;
    }}
    .stat-subtitle {{
      margin-top: 16px;
      font-size: 14px;
      font-weight: 600;
      color: #111827;
    }}
    .stats-note {{
      margin-top: 8px;
      font-size: 12px;
      color: #4b5563;
    }}
    img.chart {{
      max-width: 100%;
      border-radius: 6px;
      border: 1px solid #e5e7eb;
      margin-top: 8px;
    }}
    .ticker-chart {{
      margin-top: 16px;
    }}
    .ticker-chart-title {{
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 4px;
    }}
    table.trades {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 24px;
      font-size: 13px;
    }}
    table.trades th, table.trades td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: right;
    }}
    table.trades th:first-child,
    table.trades td:first-child {{
      text-align: left;
    }}
    table.trades thead {{
      background-color: #f3f4f6;
    }}
    table.trades tbody tr:nth-child(even) {{
      background-color: #f9fafb;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📊 {config.title}</h1>
    <div class="meta">기간: {start_str} ~ {end_str}</div>"""

    if strategy_description:
        html += f"""

    <h2>전략 설명</h2>
    <div class="strategy">
      {strategy_description}
    </div>"""

    html += f"""
    <h2>포트폴리오 vs 벤치마크 차트</h2>
    <img class="chart" src="{chart_filename}" alt="Portfolio vs Benchmark Equity Curve">

    <h2>요약 지표</h2>
    <div class="grid">
      <div class="card">
        <div class="card-label">총 수익률</div>
        <div class="card-value">{_fmt_pct(total_return)}</div>
        <div class="card-desc">백테스트 기간 누적 수익</div>
      </div>
      <div class="card">
        <div class="card-label">CAGR (연평균 복리 수익률)</div>
        <div class="card-value">{_fmt_pct(cagr)}</div>
        <div class="card-desc">연 환산 복리 성과</div>
      </div>
      <div class="card">
        <div class="card-label">MDD (최대 손실폭)</div>
        <div class="card-value">{_fmt_pct(mdd)}</div>
        <div class="card-desc">최대 낙폭(피크 대비)</div>
      </div>
      <div class="card">
        <div class="card-label">연간 변동성</div>
        <div class="card-value">{_fmt_pct(volatility)}</div>
        <div class="card-desc">일간 수익률 표준편차 연환산</div>
      </div>
      <div class="card">
        <div class="card-label">Sharpe Ratio</div>
        <div class="card-value">{sharpe:.3f}</div>
        <div class="card-desc">변동성 대비 초과 수익</div>
      </div>
      <div class="card">
        <div class="card-label">Calmar Ratio</div>
        <div class="card-value">{calmar:.3f}</div>
        <div class="card-desc">CAGR ÷ |MDD|</div>
      </div>"""

    if b_total is not None:
        html += f"""
      <div class="card">
        <div class="card-label">벤치마크 총 수익률</div>
        <div class="card-value">{_fmt_pct(b_total)}</div>
        <div class="card-desc">동일 기간 벤치마크 성과</div>
      </div>"""
    if excess is not None:
        html += f"""
      <div class="card">
        <div class="card-label">초과 수익률</div>
        <div class="card-value">{_fmt_pct(excess)}</div>
        <div class="card-desc">포트폴리오 − 벤치마크</div>
      </div>"""

    html += f"""
    </div>
    """

    if additional_sections:
        html += f"""

    {additional_sections}
    """

    def summarize_trades(trades: List[TradeRecord]) -> Dict[str, float]:
        buy_queue: deque[Tuple[TradeRecord, float]] = deque()
        success_returns: List[float] = []
        failure_returns: List[float] = []
        for trade in sorted(trades, key=lambda t: getattr(t, "date", datetime.min)):
            action = getattr(trade, "action", "").upper()
            qty = float(getattr(trade, "quantity", 0.0))
            if qty <= 0:
                continue
            if action == "BUY":
                buy_queue.append((trade, qty))
                continue
            if action != "SELL":
                continue
            remaining = qty
            sell_price = float(getattr(trade, "price", 0.0))
            while remaining > 1e-9 and buy_queue:
                buy_trade, buy_remaining = buy_queue[0]
                match_qty = min(buy_remaining, remaining)
                buy_price = float(getattr(buy_trade, "price", 0.0))
                if buy_price > 0:
                    pnl_pct = (sell_price - buy_price) / buy_price
                    if pnl_pct >= 0.24:
                        success_returns.append(pnl_pct)
                    else:
                        failure_returns.append(pnl_pct)
                buy_remaining -= match_qty
                remaining -= match_qty
                if buy_remaining <= 1e-9:
                    buy_queue.popleft()
                else:
                    buy_queue[0] = (buy_trade, buy_remaining)
        total = len(success_returns) + len(failure_returns)
        return {
            "success_count": len(success_returns),
            "failure_count": len(failure_returns),
            "success_ratio": len(success_returns) / total if total else 0.0,
            "failure_ratio": len(failure_returns) / total if total else 0.0,
            "success_avg": sum(success_returns) / len(success_returns) if success_returns else 0.0,
            "failure_avg": sum(failure_returns) / len(failure_returns) if failure_returns else 0.0,
        }

    # 전략 설명 섹션 (있을 때만 표시)
    if strategy_description:
        html += f"""

    <h2>전략 설명</h2>
    <div class="strategy">
      {strategy_description}
    </div>"""

    html += """
  </div>
</body>
</html>
"""

    html_path = os.path.join(config.output_dir, config.html_filename)

    # 메인 에쿼티 차트 이미지를 base64 로 인코딩하여 HTML 에 직접 포함
    try:
        with open(equity_chart_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            data_uri = f"data:image/png;base64,{img_base64}"
            html = html.replace(
                f'src="{chart_filename}"',
                f'src="{data_uri}"',
            )
    except OSError:
        # 이미지 인코딩에 실패하면 파일 경로 방식 그대로 둔다.
        pass

    tail_sections = ""

    # 종목별 매수/매도 차트가 있으면 함께 포함
    if per_ticker_charts:
        per_ticker_html_parts: list[str] = []
        per_ticker_html_parts.append('\n    <h2>종목별 매수/매도 차트</h2>')
        for ticker, path in sorted(per_ticker_charts.items()):
            try:
                with open(path, "rb") as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    data_uri = f"data:image/png;base64,{img_base64}"
            except OSError:
                continue

            per_ticker_html_parts.append(
                f"""
    <div class="ticker-chart">
      <div class="ticker-chart-title">{ticker}</div>
      <img class="chart" src="{data_uri}" alt="{ticker} Trade Chart">
    </div>"""
            )

        per_ticker_block = "".join(per_ticker_html_parts)
        tail_sections += per_ticker_block

    if trades_section:
        tail_sections += trades_section
    if trade_pairs_section:
        tail_sections += trade_pairs_section

    if tail_sections:
        html = html.replace("\n  </div>\n</body>", f"{tail_sections}\n  </div>\n</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    return os.path.abspath(html_path)

