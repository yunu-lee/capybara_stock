from __future__ import annotations

from datetime import date, datetime
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from core import BacktestDataProvider, BacktestExecutionEngine, BacktestReport, BacktestResult, Portfolio
from strategy.new_high_breakout import (
    DEFAULT_KOREA_UNIVERSE,
    NEW_HIGH_BACKTEST_CONFIG,
    NewHighBreakoutStrategy,
    build_korea_universe,
    send_new_high_backtest_report,
)


def _parse_universe_text(text: str) -> List[str]:
    tokens = re.split(r"[,\n\r\t ]+", (text or "").strip())
    tickers: List[str] = []
    for raw in tokens:
        t = raw.strip()
        if not t:
            continue
        if re.fullmatch(r"\d{6}", t):
            # 기본은 KOSPI(.KS)로 가정 (KOSDAQ는 사용자가 .KQ로 직접 입력)
            t = f"{t}.KS"
        tickers.append(t)
    # 순서 유지 중복 제거
    seen = set()
    deduped: List[str] = []
    for t in tickers:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def _run_new_high_backtest_with_progress(
    *,
    cfg: Dict[str, Any],
    universe: Sequence[str],
    benchmark: str,
    start_date: datetime,
    end_date: datetime,
    progress_cb: Optional[callable] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, list, BacktestDataProvider]:
    data_provider = BacktestDataProvider(universe=universe, benchmark=benchmark)

    initial_cash_value = float(cfg.get("initial_cash_krw", 100_000_000))
    portfolio = Portfolio(initial_cash={"KRW": initial_cash_value})
    for ticker in universe:
        portfolio.add_stock(ticker, "KRW")

    strategy = NewHighBreakoutStrategy(
        universe=universe,
        data_provider=data_provider,
        allocation_pct=float(cfg.get("allocation_pct", 0.05)),
        stop_loss_pct=float(cfg.get("stop_loss_pct", 0.08)),
        ma_window=int(cfg.get("ma_window", 20)),
        high_lookback=int(cfg.get("high_lookback", 252)),
        cost_pct=float(cfg.get("cost_pct", 0.1)),
        cash_currency="KRW",
        debug=bool(cfg.get("debug", False)),
        backtest_start=start_date,
        backtest_end=end_date,
        rank_weight_cap=float(cfg.get("rank_weight_cap", 0.0)),
        rank_weight_rs=float(cfg.get("rank_weight_rs", 0.0)),
        fresh_new_high_window=int(cfg.get("fresh_new_high_window", 30)),
        fresh_new_high_weight=float(cfg.get("fresh_new_high_weight", 1.0)),
        short_term_penalty_window=int(cfg.get("short_term_penalty_window", 10)),
        short_term_penalty_threshold=float(cfg.get("short_term_penalty_threshold", 0.12)),
        short_term_penalty_weight=float(cfg.get("short_term_penalty_weight", 0.0)),
        max_new_positions=int(cfg.get("max_new_positions", 20)),
    )

    execution_engine = BacktestExecutionEngine()
    dates = data_provider.get_trading_days(start_date, end_date)
    if not dates:
        raise RuntimeError("지정된 기간에 대한 거래일이 없습니다.")

    # 1) 전략 초기화 (데이터/지표 준비)
    if progress_cb:
        progress_cb("데이터/지표 준비 중...", 0.0)
    strategy.on_start(portfolio)

    # 2) 일자별 시뮬레이션 (진행 표시)
    snapshots: List[Tuple[datetime, float]] = []
    latest_prices: Dict[str, float] = {}
    total = len(dates)
    for idx, current_date in enumerate(dates):
        orders = strategy.on_bar(current_date, portfolio) or []
        execution_engine.execute_orders(
            orders=orders,
            date=current_date,
            data_provider=data_provider,
            portfolio=portfolio,
        )

        # equity 계산 (BacktestEngine._compute_equity 와 동일한 방식)
        prices: Dict[str, float] = {}
        for ticker, position in portfolio.positions.items():
            price = data_provider.get_price(ticker, current_date)
            if price is not None:
                position.last_price = price
            else:
                price = position.last_price
            if price is not None:
                prices[ticker] = float(price)
        snapshot = portfolio.snapshot(prices=prices)
        totals = snapshot.get("totals", {})
        equity = float(sum(float(v) for v in totals.values()))

        snapshots.append((current_date, equity))
        portfolio.annotate_trades_with_value(current_date, equity)
        latest_prices = prices.copy()

        if progress_cb:
            progress = (idx + 1) / total
            progress_cb(f"백테스트 진행 중... ({idx + 1}/{total}) {current_date:%Y-%m-%d}", progress)

    strategy.on_end(portfolio)

    equity_curve = pd.Series(
        data=[v for _, v in snapshots],
        index=[d for d, _ in snapshots],
        name="equity",
    )

    # 벤치마크 곡선 구성 (BacktestEngine.run 로직과 동일)
    benchmark_curve = None
    if benchmark:
        start = dates[0]
        end = dates[-1]
        bench_df = data_provider.get_history(benchmark, start, end)
        if not bench_df.empty and "Date" in bench_df.columns and "Close" in bench_df.columns:
            bench_df = bench_df.sort_values("Date")
            closes = bench_df["Close"].astype(float)
            if not closes.empty and closes.iloc[0] != 0:
                normalized = closes / float(closes.iloc[0])
                dt_series = pd.to_datetime(bench_df["Date"])
                normalized.index = [ts.to_pydatetime() for ts in dt_series]
                benchmark_curve = normalized

    trades = list(portfolio.get_trade_history())
    result = BacktestResult(
        equity_curve=equity_curve,
        benchmark_equity_curve=benchmark_curve,
        trades=trades,
        latest_prices=latest_prices or None,
    )
    report = BacktestReport(result)
    metrics = report.compute_metrics()
    df_equity = report.to_dataframe()

    return metrics, df_equity, trades, data_provider


def _render_new_high_breakout_page() -> None:
    st.subheader("New High Breakout (52주 신고가)")

    with st.form("new_high_breakout_form"):
        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            universe_mode = st.selectbox(
                "유니버스",
                options=["자동(KOSPI200+KOSDAQ150)", "샘플(20종)", "직접 입력"],
                index=1,
            )
        with col_u2:
            benchmark = st.text_input("벤치마크", value=str(NEW_HIGH_BACKTEST_CONFIG.get("benchmark", "069500.KS")))

        universe_text = ""
        if universe_mode == "직접 입력":
            universe_text = st.text_area(
                "종목 리스트(쉼표/줄바꿈 구분, 6자리 코드는 .KS로 자동 보정)",
                value="005930.KS, 000660.KS",
                height=120,
            )

        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        with col_d1:
            start_dt = st.date_input("시작일", value=date.fromisoformat(str(NEW_HIGH_BACKTEST_CONFIG.get("start_date", "2025-04-01"))))
        with col_d2:
            end_dt_default = date.today()
            end_dt = st.date_input("종료일", value=end_dt_default)
        with col_d3:
            initial_cash_krw = st.number_input("초기자금(KRW)", min_value=1_000_000, step=1_000_000, value=100_000_000)

        st.markdown("#### 종목 선정/매매 규칙")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            allocation_pct = st.number_input("1회 배분 비중(0~1)", min_value=0.001, max_value=1.0, value=0.05, step=0.01, format="%.3f")
        with c2:
            stop_loss_pct = st.number_input("손절 비율(0~1)", min_value=0.001, max_value=1.0, value=0.08, step=0.01, format="%.3f")
        with c3:
            ma_window = st.number_input("MA 윈도우", min_value=5, max_value=200, value=20, step=1)
        with c4:
            high_lookback = st.number_input("신고가 룩백(거래일)", min_value=20, max_value=600, value=252, step=1)

        st.markdown("#### 랭킹(종목 선정 기준)")
        r1, r2, r3 = st.columns(3)
        with r1:
            rank_weight_cap = st.number_input("시총 가중치", min_value=0.0, max_value=5.0, value=0.0, step=0.1, format="%.2f")
            rank_weight_rs = st.number_input("RS 가중치", min_value=0.0, max_value=5.0, value=0.0, step=0.1, format="%.2f")
        with r2:
            fresh_new_high_window = st.number_input("신선도 윈도우(일)", min_value=1, max_value=120, value=30, step=1)
            fresh_new_high_weight = st.number_input("신선도 가중치", min_value=0.0, max_value=5.0, value=1.0, step=0.1, format="%.2f")
        with r3:
            short_term_penalty_window = st.number_input("과열 윈도우(일)", min_value=1, max_value=60, value=10, step=1)
            short_term_penalty_threshold = st.number_input("과열 임계값(0~1)", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.3f")
            short_term_penalty_weight = st.number_input("과열 패널티 가중치", min_value=0.0, max_value=5.0, value=0.0, step=0.1, format="%.2f")

        opt1, opt2, opt3 = st.columns(3)
        with opt1:
            max_new_positions = st.number_input("하루 신규 편입 상한", min_value=1, max_value=100, value=20, step=1)
        with opt2:
            cost_pct = st.number_input("거래비용(%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05, format="%.2f")
        with opt3:
            debug = st.checkbox("디버그 로그", value=False)

        run_clicked = st.form_submit_button("백테스트 실행")

    if not run_clicked:
        # 이전 실행 결과가 있으면 보여주기
        html = st.session_state.get("new_high_last_html")
        if html:
            components.html(html, height=1100, scrolling=True)
        return

    # 유니버스 구성
    if universe_mode == "샘플(20종)":
        universe = list(DEFAULT_KOREA_UNIVERSE)
    elif universe_mode == "자동(KOSPI200+KOSDAQ150)":
        universe = build_korea_universe()
    else:
        universe = _parse_universe_text(universe_text)

    if not universe:
        st.error("유니버스가 비어 있습니다. 종목을 입력하거나 다른 유니버스를 선택해주세요.")
        return

    cfg: Dict[str, Any] = {
        "initial_cash_krw": float(initial_cash_krw),
        "allocation_pct": float(allocation_pct),
        "stop_loss_pct": float(stop_loss_pct),
        "ma_window": int(ma_window),
        "high_lookback": int(high_lookback),
        "cost_pct": float(cost_pct),
        "rank_weight_cap": float(rank_weight_cap),
        "rank_weight_rs": float(rank_weight_rs),
        "fresh_new_high_window": int(fresh_new_high_window),
        "fresh_new_high_weight": float(fresh_new_high_weight),
        "short_term_penalty_window": int(short_term_penalty_window),
        "short_term_penalty_threshold": float(short_term_penalty_threshold),
        "short_term_penalty_weight": float(short_term_penalty_weight),
        "max_new_positions": int(max_new_positions),
        "debug": bool(debug),
    }

    start_date = _to_datetime(start_dt)
    end_date = _to_datetime(end_dt)
    if end_date < start_date:
        st.error("종료일은 시작일보다 빠를 수 없습니다.")
        return

    progress = st.progress(0.0)
    status = st.empty()

    def _ui_progress(message: str, value: float) -> None:
        status.write(message)
        progress.progress(max(0.0, min(1.0, float(value))))

    try:
        metrics, df_equity, trades, data_provider = _run_new_high_backtest_with_progress(
            cfg=cfg,
            universe=universe,
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date,
            progress_cb=_ui_progress,
        )
    except Exception as exc:
        st.error(f"백테스트 실행 중 오류: {exc}")
        return

    # 리포트 생성 + 화면 렌더링
    report_progress = st.progress(0.0)
    report_status = st.empty()

    def _report_cb(msg: str, v: float) -> None:
        report_status.write(msg)
        report_progress.progress(max(0.0, min(1.0, float(v))))

    try:
        html_path = send_new_high_backtest_report(
            metrics=metrics,
            df_equity=df_equity,
            trades=trades,
            universe=universe,
            data_provider=data_provider,
            start_date=start_date,
            end_date=end_date,
            cfg=cfg,
            on_progress=_report_cb,
        )
    except Exception as exc:
        st.error(f"리포트 생성 중 오류: {exc}")
        return

    if not html_path or not os.path.exists(html_path):
        st.warning("HTML 리포트 파일을 찾지 못했습니다.")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    st.session_state["new_high_last_html"] = html

    colm1, colm2, colm3, colm4 = st.columns(4)
    with colm1:
        st.metric("총 수익률", f"{metrics.get('total_return', 0.0) * 100:.2f}%")
    with colm2:
        st.metric("MDD", f"{metrics.get('mdd', 0.0) * 100:.2f}%")
    with colm3:
        st.metric("거래 횟수", f"{metrics.get('num_trades', 0)}")
    with colm4:
        bench = metrics.get("benchmark_total_return")
        st.metric("벤치마크", "n/a" if bench is None else f"{bench * 100:.2f}%")

    st.download_button(
        "리포트 다운로드(HTML)",
        data=html,
        file_name=os.path.basename(html_path),
        mime="text/html",
    )
    components.html(html, height=1100, scrolling=True)


st.set_page_config(page_title="CAPYBARA STOCK", layout="wide")
st.title("CAPYBARA STOCK")

with st.sidebar:
    st.header("메인 메뉴")
    selected_menu = st.radio(
        "이동",
        options=["시장", "백테스트"],
        index=0,
    )

if selected_menu == "시장":
    st.subheader("시장")
    st.info("준비 중입니다.")
elif selected_menu == "백테스트":
    st.subheader("백테스트")
    sub_menu = st.selectbox("전략 선택", options=["New High Breakout"], index=0)
    if sub_menu == "New High Breakout":
        _render_new_high_breakout_page()
