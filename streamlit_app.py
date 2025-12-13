import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from core import Stock


def get_period_days(period: str) -> Optional[int]:
    """기간 선택 문자열을 일수로 변환"""
    period_map = {
        "최근 30일": 30,
        "최근 90일": 90,
        "최근 180일": 180,
        "최근 1년": 365,
        "전체": None,
    }
    return period_map.get(period, 365)


def create_stock_chart(
    ticker: str,
    period_days: Optional[int],
    indicators: List[str],
) -> Optional[go.Figure]:
    """
    Plotly를 사용하여 주식 차트 생성
    
    Args:
        ticker: 종목 코드
        period_days: 표시할 기간 (일수, None이면 전체)
        indicators: 선택된 보조지표 리스트
        
    Returns:
        Plotly Figure 객체 또는 None (에러 시)
    """
    try:
        # Stock 객체 생성 및 데이터 로드
        stock = Stock(ticker)
        
        # 기간에 따라 데이터 가져오기
        if period_days:
            df = stock.get_history(period_days=period_days)
        else:
            # 전체 데이터는 충분히 큰 값으로 가져오기
            df = stock.get_history(period_days=365 * 3)
        
        if df is None or df.empty:
            return None
        
        # Date 컬럼이 있으면 datetime으로 변환
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # 필수 컬럼 확인
        required_cols = ["Date", "Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required_cols):
            return None
        
        # 서브플롯 생성 (2개 행: 가격 차트, 거래량 차트)
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=("가격 차트", "거래량"),
        )
        
        # 캔들스틱 차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="가격",
            ),
            row=1,
            col=1,
        )
        
        # 보조지표 추가
        if stock._history is None or stock._history.empty:
            stock._history = df
            stock._indicators = None  # 리셋
        
        indicators_obj = stock.indicators
        
        # SMA 추가
        if "SMA" in indicators:
            sma20 = indicators_obj.sma(20)
            sma60 = indicators_obj.sma(60)
            
            # 길이 확인 및 NaN 처리
            if len(sma20) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=sma20.values,
                        name="SMA 20",
                        line=dict(color="blue", width=1.5),
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
            if len(sma60) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=sma60.values,
                        name="SMA 60",
                        line=dict(color="orange", width=1.5),
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
        
        # EMA 추가
        if "EMA" in indicators:
            ema12 = indicators_obj.ema(12)
            ema26 = indicators_obj.ema(26)
            
            if len(ema12) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=ema12.values,
                        name="EMA 12",
                        line=dict(color="purple", width=1.5, dash="dash"),
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
            if len(ema26) == len(df):
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=ema26.values,
                        name="EMA 26",
                        line=dict(color="brown", width=1.5, dash="dash"),
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
        
        # 볼린저 밴드 추가
        if "볼린저 밴드" in indicators:
            bb_mid, bb_upper, bb_lower = indicators_obj.bollinger_bands()
            
            if len(bb_mid) == len(df):
                # 상단 밴드 (먼저 추가)
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=bb_upper.values,
                        name="BB 상단",
                        line=dict(color="gray", width=1, dash="dot"),
                        fill=None,
                        showlegend=False,
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
                # 하단 밴드 (채우기)
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=bb_lower.values,
                        name="볼린저 밴드",
                        line=dict(color="gray", width=1, dash="dot"),
                        fill="tonexty",
                        fillcolor="rgba(128,128,128,0.15)",
                        showlegend=True,
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
                # 중간선
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=bb_mid.values,
                        name="BB 중간",
                        line=dict(color="gray", width=1),
                        showlegend=False,
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
        
        # RSI 추가 (별도 y축 사용)
        if "RSI" in indicators:
            rsi = indicators_obj.rsi(14)
            if len(rsi) == len(df):
                # RSI는 별도 y축으로 표시
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=rsi.values,
                        name="RSI",
                        line=dict(color="red", width=1.5),
                        yaxis="y2",
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
                # RSI 기준선 추가
                fig.add_hline(
                    y=70,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.5,
                    row=1,
                    col=1,
                    annotation_text="과매수 (70)",
                )
                fig.add_hline(
                    y=30,
                    line_dash="dash",
                    line_color="green",
                    opacity=0.5,
                    row=1,
                    col=1,
                    annotation_text="과매도 (30)",
                )
        
        # MACD 추가
        if "MACD" in indicators:
            macd, signal, hist = indicators_obj.macd()
            
            if len(macd) == len(df):
                # MACD와 Signal은 별도 y축 사용
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=macd.values,
                        name="MACD",
                        line=dict(color="blue", width=1.5),
                        yaxis="y3",
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=signal.values,
                        name="Signal",
                        line=dict(color="red", width=1.5),
                        yaxis="y3",
                        connectgaps=False,
                    ),
                    row=1,
                    col=1,
                )
                # 히스토그램은 작은 값이므로 별도 y축 사용
                colors = ["green" if h >= 0 else "red" for h in hist.values]
                fig.add_trace(
                    go.Bar(
                        x=df["Date"],
                        y=hist.values,
                        name="MACD Hist",
                        marker_color=colors,
                        opacity=0.4,
                        yaxis="y4",
                    ),
                    row=1,
                    col=1,
                )
        
        # 거래량 차트 추가
        if "Volume" in df.columns:
            colors_volume = [
                "red" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "blue"
                for i in range(len(df))
            ]
            fig.add_trace(
                go.Bar(
                    x=df["Date"],
                    y=df["Volume"],
                    name="거래량",
                    marker_color=colors_volume,
                    opacity=0.6,
                ),
                row=2,
                col=1,
            )
        
        # 레이아웃 업데이트
        stock_name = stock.name if stock.name else ticker
        fig.update_layout(
            title=f"{stock_name} ({ticker})",
            xaxis_rangeslider_visible=False,
            height=800,
            hovermode="x unified",
            template="plotly_white",
        )
        
        # Y축 레이블 설정
        fig.update_yaxes(title_text="가격", row=1, col=1)
        if "Volume" in df.columns:
            fig.update_yaxes(title_text="거래량", row=2, col=1)
        
        # RSI y축 설정 (오른쪽)
        if "RSI" in indicators:
            fig.update_layout(
                yaxis2=dict(
                    title="RSI",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    showgrid=False,
                )
            )
        
        # MACD y축 설정 (오른쪽, RSI 위)
        if "MACD" in indicators:
            fig.update_layout(
                yaxis3=dict(
                    title="MACD",
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=0.95,
                    showgrid=False,
                ),
                yaxis4=dict(
                    title="MACD Hist",
                    overlaying="y",
                    side="right",
                    anchor="free",
                    position=0.98,
                    showgrid=False,
                ),
            )
        
        return fig
        
    except Exception as e:
        st.error(f"차트 생성 중 오류 발생: {str(e)}")
        return None


# 메인 앱
st.set_page_config(page_title="CAPYBARA STOCK", layout="wide")

st.title("CAPYBARA STOCK")
st.write("주식 차트를 확인하세요")

# 사이드바
with st.sidebar:
    st.header("차트 설정")
    
    # 티커 입력
    ticker = st.text_input(
        "종목 코드",
        value="005930.KS",
        help="예: 005930.KS (삼성전자), 000660.KS (SK하이닉스)",
    )
    
    # 기간 선택
    period = st.selectbox(
        "기간",
        options=["최근 30일", "최근 90일", "최근 180일", "최근 1년", "전체"],
        index=2,
    )
    
    # 보조지표 선택
    st.subheader("보조지표")
    indicators = st.multiselect(
        "보조지표 선택",
        options=["SMA", "EMA", "RSI", "MACD", "볼린저 밴드"],
        default=["SMA"],
        help="여러 개 선택 가능합니다",
    )

# 메인 영역
if ticker:
    period_days = get_period_days(period)
    
    with st.spinner("차트를 생성하는 중..."):
        fig = create_stock_chart(ticker, period_days, indicators)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # 기본 정보 표시
            try:
                stock = Stock(ticker)
                if stock._history is not None and not stock._history.empty:
                    latest = stock._history.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("현재가", f"{latest['Close']:,.0f}")
                    with col2:
                        change = latest['Close'] - stock._history.iloc[-2]['Close'] if len(stock._history) > 1 else 0
                        st.metric("전일 대비", f"{change:+,.0f}")
                    with col3:
                        if len(stock._history) > 1:
                            change_pct = (change / stock._history.iloc[-2]['Close']) * 100
                            st.metric("등락률", f"{change_pct:+.2f}%")
                    with col4:
                        if "Volume" in latest:
                            st.metric("거래량", f"{latest['Volume']:,.0f}")
            except Exception as e:
                st.warning(f"기본 정보를 가져올 수 없습니다: {str(e)}")
        else:
            st.error("차트를 생성할 수 없습니다. 티커를 확인하거나 잠시 후 다시 시도해주세요.")
else:
    st.info("사이드바에서 종목 코드를 입력해주세요.")
