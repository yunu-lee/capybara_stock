import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradeSignal:
    """매매 신호 정보"""
    date: datetime  # 거래 날짜
    price: float    # 거래 가격
    type: Literal['buy', 'sell']  # 매매 타입
    
    def __repr__(self) -> str:
        return f"TradeSignal({self.type.upper()} at {self.price} on {self.date.strftime('%Y-%m-%d')})"


@dataclass
class ChartConfig:
    """차트 설정을 관리하는 config 클래스"""
    
    # 차트 타입
    chart_type: Literal['candle', 'line'] = 'candle'
    
    # 가격 차트 설정
    show_volume: bool = True
    log_price: bool = False
    
    # 이동평균선 (가격 차트에 오버레이)
    show_sma: bool = True
    sma_windows: List[int] = field(default_factory=lambda: [5, 20, 60, 120])
    
    show_ema: bool = False
    ema_windows: List[int] = field(default_factory=lambda: [12, 26])
    
    # 볼린저 밴드 (가격 차트에 오버레이)
    show_bollinger: bool = False
    bb_window: int = 20
    bb_std: float = 2.0
    
    # 하단 별도 차트 지표들
    show_rsi: bool = False
    rsi_window: int = 14
    
    show_macd: bool = False
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    show_stochastic: bool = False
    stoch_k_window: int = 14
    stoch_d_window: int = 3
    
    show_rs: bool = False
    rs_window: int = 52
    
    show_volume_ratio: bool = False
    volume_ratio_window: int = 20
    
    # 매매 신호 표시
    show_trade_signals: bool = True  # 매매 신호 표시 여부
    buy_marker_color: str = 'green'
    sell_marker_color: str = 'red'
    marker_size: int = 100
    
    # 차트 스타일
    dpi: int = 240
    figsize: tuple = (12, 8)
    
    # 색상 설정
    up_color: str = 'r'
    down_color: str = 'b'
    ma_colors: List[str] = field(default_factory=lambda: ['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
    
    def count_subplots(self) -> int:
        """필요한 subplot 개수 계산 (가격 + 거래량 + 보조지표)"""
        count = 1  # 가격 차트
        if self.show_volume:
            count += 1
        if self.show_rsi:
            count += 1
        if self.show_macd:
            count += 1
        if self.show_stochastic:
            count += 1
        if self.show_rs:
            count += 1
        if self.show_volume_ratio:
            count += 1
        return count
    
    def get_panel_ratios(self) -> tuple:
        """각 panel의 비율 계산"""
        ratios = [3]  # 가격 차트는 3
        if self.show_volume:
            ratios.append(1)
        if self.show_rsi:
            ratios.append(1)
        if self.show_macd:
            ratios.append(1)
        if self.show_stochastic:
            ratios.append(1)
        if self.show_rs:
            ratios.append(1)
        if self.show_volume_ratio:
            ratios.append(1)
        return tuple(ratios)


class ChartRenderer:
    """차트 렌더링 클래스"""
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        차트 렌더러 초기화
        
        Args:
            config: 차트 설정 (None이면 기본 설정 사용)
        """
        self.config = config if config is not None else ChartConfig()
        self._setup_korean_font()
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        korean_fonts = [
            'Malgun Gothic',
            'NanumGothic',
            'NanumBarunGothic',
            'Batang',
            'Dotum',
            'Gulim',
            'AppleGothic',
            'DejaVu Sans',  # Linux fallback
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        font_found = False
        for font_name in korean_fonts:
            if font_name in available_fonts:
                try:
                    mpl.rcParams['font.family'] = font_name
                    mpl.rcParams['axes.unicode_minus'] = False
                    mpl.rcParams['font.size'] = 10
                    font_found = True
                    break
                except Exception:
                    continue
        
        if not font_found:
            # Linux 환경을 위한 추가 시도
            try:
                # sans-serif 폰트 사용 (한글 지원)
                mpl.rcParams['font.family'] = 'sans-serif'
                mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
                mpl.rcParams['axes.unicode_minus'] = False
                mpl.rcParams['font.size'] = 10
            except Exception:
                pass
    
    def _get_korean_font(self) -> str:
        """현재 설정된 폰트 반환"""
        return str(mpl.rcParams['font.family'])
    
    def render(
        self,
        df: pd.DataFrame,
        indicators_data: Optional[Dict[str, pd.Series]] = None,
        trade_signals: Optional[List[TradeSignal]] = None,
        display_days: Optional[int] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        차트 렌더링
        
        Args:
            df: OHLCV 데이터프레임 (Date, Open, High, Low, Close, Volume 필요)
            indicators_data: 계산된 보조지표 데이터 딕셔너리
            trade_signals: 매매 신호 리스트 (TradeSignal 객체들)
            display_days: 표시할 일수 (None이면 전체)
            title: 차트 제목
            save_path: 저장 경로 (None이면 'output/chart.png')
            
        Returns:
            저장된 파일 경로
        """
        if indicators_data is None:
            indicators_data = {}
        
        # 데이터 준비
        dfp = df.copy()
        dfp = dfp.set_index('Date')
        dfp.index.name = 'Date'
        
        # display_days에 따라 필터링
        if display_days is not None and display_days > 0:
            cutoff = dfp.index.max() - pd.Timedelta(days=display_days)
            dfp_to_plot = dfp[dfp.index >= cutoff]
        else:
            dfp_to_plot = dfp
        
        # addplot 리스트 생성
        addplots = []
        panel_idx = 1 if self.config.show_volume else 0
        
        # Helper function to filter series by date range
        def filter_series(series: pd.Series) -> pd.Series:
            """날짜 범위에 맞게 시리즈 필터링하고 유효성 검사"""
            filtered = series[series.index.isin(dfp_to_plot.index)]
            # 빈 시리즈나 모두 NaN인 경우 None 반환
            if filtered.empty or filtered.notna().sum() == 0:
                return None
            return filtered
        
        # 1. 가격 차트에 오버레이할 지표들
        
        # SMA 추가
        if self.config.show_sma:
            for i, window in enumerate(self.config.sma_windows):
                sma_key = f'sma_{window}'
                if sma_key in indicators_data:
                    sma_series = filter_series(indicators_data[sma_key])
                    if sma_series is not None:
                        color = self.config.ma_colors[i % len(self.config.ma_colors)]
                        addplots.append(
                            mpf.make_addplot(
                                sma_series,
                                panel=0,
                                label=f'SMA{window}',
                                color=color,
                                width=1.0,
                            )
                        )
        
        # EMA 추가
        if self.config.show_ema:
            for i, window in enumerate(self.config.ema_windows):
                ema_key = f'ema_{window}'
                if ema_key in indicators_data:
                    ema_series = filter_series(indicators_data[ema_key])
                    if ema_series is not None:
                        color = self.config.ma_colors[
                            (len(self.config.sma_windows) + i) % len(self.config.ma_colors)
                        ]
                        addplots.append(
                            mpf.make_addplot(
                                ema_series,
                                panel=0,
                                label=f'EMA{window}',
                                color=color,
                                width=1.0,
                                linestyle='--',
                            )
                        )
        
        # 볼린저 밴드 추가
        if self.config.show_bollinger:
            if 'bb_upper' in indicators_data and 'bb_lower' in indicators_data:
                bb_upper = filter_series(indicators_data['bb_upper'])
                bb_lower = filter_series(indicators_data['bb_lower'])
                if bb_upper is not None and bb_lower is not None:
                    if 'bb_mid' in indicators_data:
                        bb_mid = filter_series(indicators_data['bb_mid'])
                        if bb_mid is not None:
                            addplots.append(
                                mpf.make_addplot(
                                    bb_mid,
                                    panel=0,
                                    label=f'BB Mid({self.config.bb_window})',
                                    color='gray',
                                    width=0.7,
                                    linestyle='--',
                                )
                            )
                    addplots.append(
                        mpf.make_addplot(
                            bb_upper,
                            panel=0,
                            label=f'BB Upper',
                            color='green',
                            width=0.7,
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            bb_lower,
                            panel=0,
                            label=f'BB Lower',
                            color='green',
                            width=0.7,
                        )
                    )
        
        # 2. 하단 별도 차트 지표들
        
        # RSI
        if self.config.show_rsi and 'rsi' in indicators_data:
            rsi = filter_series(indicators_data['rsi'])
            if rsi is not None:
                panel_idx += 1
                addplots.append(
                    mpf.make_addplot(
                        rsi,
                        panel=panel_idx,
                        ylabel=f'RSI({self.config.rsi_window})',
                        color='purple',
                        width=1.0,
                    )
                )
                # RSI 기준선 (30, 70)
                addplots.append(
                    mpf.make_addplot(
                        pd.Series([70] * len(rsi), index=rsi.index),
                        panel=panel_idx,
                        color='red',
                        width=0.5,
                        linestyle='--',
                        alpha=0.5,
                    )
                )
                addplots.append(
                    mpf.make_addplot(
                        pd.Series([30] * len(rsi), index=rsi.index),
                        panel=panel_idx,
                        color='green',
                        width=0.5,
                        linestyle='--',
                        alpha=0.5,
                    )
                )
        
        # MACD
        if self.config.show_macd:
            if 'macd' in indicators_data and 'macd_signal' in indicators_data:
                macd = filter_series(indicators_data['macd'])
                signal = filter_series(indicators_data['macd_signal'])
                if macd is not None and signal is not None:
                    panel_idx += 1
                    addplots.append(
                        mpf.make_addplot(
                            macd,
                            panel=panel_idx,
                            ylabel='MACD',
                            color='blue',
                            width=1.0,
                            label='MACD',
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            signal,
                            panel=panel_idx,
                            color='red',
                            width=1.0,
                            label='Signal',
                        )
                    )
                    
                    # Histogram
                    if 'macd_hist' in indicators_data:
                        hist = filter_series(indicators_data['macd_hist'])
                        if hist is not None:
                            colors = ['red' if v >= 0 else 'blue' for v in hist]
                            addplots.append(
                                mpf.make_addplot(
                                    hist,
                                    panel=panel_idx,
                                    type='bar',
                                    color=colors,
                                    alpha=0.3,
                                    width=0.7,
                                )
                            )
        
        # Stochastic
        if self.config.show_stochastic:
            if 'stoch_k' in indicators_data and 'stoch_d' in indicators_data:
                k = filter_series(indicators_data['stoch_k'])
                d = filter_series(indicators_data['stoch_d'])
                if k is not None and d is not None:
                    panel_idx += 1
                    addplots.append(
                        mpf.make_addplot(
                            k,
                            panel=panel_idx,
                            ylabel='Stochastic',
                            color='blue',
                            width=1.0,
                            label='%K',
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            d,
                            panel=panel_idx,
                            color='red',
                            width=1.0,
                            label='%D',
                        )
                    )
                    
                    # 기준선 (80, 20)
                    addplots.append(
                        mpf.make_addplot(
                            pd.Series([80] * len(k), index=k.index),
                            panel=panel_idx,
                            color='red',
                            width=0.5,
                            linestyle='--',
                            alpha=0.5,
                        )
                    )
                    addplots.append(
                        mpf.make_addplot(
                            pd.Series([20] * len(k), index=k.index),
                            panel=panel_idx,
                            color='green',
                            width=0.5,
                            linestyle='--',
                            alpha=0.5,
                        )
                    )
        
        # RS
        if self.config.show_rs and 'rs' in indicators_data:
            rs = filter_series(indicators_data['rs'])
            if rs is not None:
                panel_idx += 1
                addplots.append(
                    mpf.make_addplot(
                        rs,
                        panel=panel_idx,
                        ylabel=f'RS({self.config.rs_window})',
                        color='purple',
                        width=1.0,
                    )
                )
                # 0 기준선
                addplots.append(
                    mpf.make_addplot(
                        pd.Series([0] * len(rs), index=rs.index),
                        panel=panel_idx,
                        color='gray',
                        width=0.7,
                        linestyle='--',
                        alpha=0.5,
                    )
                )
        
        # 거래량 비율
        if self.config.show_volume_ratio and 'volume_ratio' in indicators_data:
            vol_ratio = filter_series(indicators_data['volume_ratio'])
            if vol_ratio is not None:
                panel_idx += 1
                addplots.append(
                    mpf.make_addplot(
                        vol_ratio,
                        panel=panel_idx,
                        ylabel='Vol Ratio',
                        color='orange',
                        width=1.0,
                    )
                )
                # 1.0 기준선
                addplots.append(
                    mpf.make_addplot(
                        pd.Series([1.0] * len(vol_ratio), index=vol_ratio.index),
                        panel=panel_idx,
                        color='gray',
                        width=0.7,
                        linestyle='--',
                        alpha=0.5,
                    )
                )
        
        # 매매 신호 마커 추가
        if self.config.show_trade_signals and trade_signals:
            # buy 신호와 sell 신호 분리
            buy_signals = [s for s in trade_signals if s.type == 'buy']
            sell_signals = [s for s in trade_signals if s.type == 'sell']
            
            # Buy 마커 (상향 삼각형)
            if buy_signals:
                buy_series = pd.Series(index=dfp_to_plot.index, dtype=float)
                for signal in buy_signals:
                    # 날짜를 datetime으로 변환
                    signal_date = pd.to_datetime(signal.date)
                    if signal_date in dfp_to_plot.index:
                        buy_series[signal_date] = signal.price
                
                # NaN이 아닌 값이 있으면 추가
                if buy_series.notna().sum() > 0:
                    addplots.append(
                        mpf.make_addplot(
                            buy_series,
                            type='scatter',
                            markersize=self.config.marker_size,
                            marker='^',
                            color=self.config.buy_marker_color,
                            panel=0,
                            secondary_y=False,
                            label=f'BUY ({len(buy_signals)})',  # 범례에 표시
                        )
                    )
            
            # Sell 마커 (하향 삼각형)
            if sell_signals:
                sell_series = pd.Series(index=dfp_to_plot.index, dtype=float)
                for signal in sell_signals:
                    signal_date = pd.to_datetime(signal.date)
                    if signal_date in dfp_to_plot.index:
                        sell_series[signal_date] = signal.price
                
                if sell_series.notna().sum() > 0:
                    addplots.append(
                        mpf.make_addplot(
                            sell_series,
                            type='scatter',
                            markersize=self.config.marker_size,
                            marker='v',
                            color=self.config.sell_marker_color,
                            panel=0,
                            secondary_y=False,
                            label=f'SELL ({len(sell_signals)})',  # 범례에 표시
                        )
                    )
        
        # 스타일 설정
        mc = mpf.make_marketcolors(
            up=self.config.up_color,
            down=self.config.down_color,
            edge='inherit',
            wick='inherit',
            volume='in',
        )
        
        # 마이너스 기호가 x로 표시되는 문제 방지
        # mplfinance가 설정을 덮어쓸 수 있으므로 명시적으로 재설정
        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            gridstyle='-',
            gridcolor='lightgray',
            y_on_right=False,
            rc={
                'font.family': mpl.rcParams['font.family'],
                'axes.unicode_minus': False,  # 마이너스 기호 문제 방지
            },
        )
        
        # 저장 경로
        out = save_path or 'output/chart.png'
        
        # 마이너스 기호 문제 방지를 위해 차트 생성 직전 재설정
        mpl.rcParams['axes.unicode_minus'] = False
        
        # plot 매개변수
        plot_kwargs = {
            'data': dfp_to_plot,
            'type': self.config.chart_type,
            'style': style,
            'volume': self.config.show_volume,
            'panel_ratios': self.config.get_panel_ratios(),
            'figsize': self.config.figsize,
            'returnfig': True,
        }
        
        # addplot이 있을 때만 추가
        if addplots:
            plot_kwargs['addplot'] = addplots
        
        if self.config.chart_type == 'candle':
            plot_kwargs['update_width_config'] = dict(
                candle_linewidth=0.8,
                volume_linewidth=0.0
            )
        
        # 차트 생성
        fig, axes = mpf.plot(**plot_kwargs)
        
        try:
            # 모든 axes에 대해 마이너스 기호 설정 적용
            for ax in axes:
                # y축 포맷터 수정하여 마이너스 기호 문제 해결
                try:
                    from matplotlib.ticker import ScalarFormatter
                    formatter = ScalarFormatter(useOffset=False)
                    formatter.set_scientific(False)
                    ax.yaxis.set_major_formatter(formatter)
                except Exception:
                    pass
            
            # 제목 설정
            if title:
                # 제목을 ASCII와 유니코드 모두 지원하도록 처리
                try:
                    axes[0].set_title(title, fontsize=12, fontweight='bold', pad=15)
                except Exception:
                    # 한글 폰트 문제 시 기본 처리
                    axes[0].set_title(title.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'), 
                                     fontsize=12, fontweight='bold', pad=15)
            
            # 로그 스케일
            if self.config.log_price:
                axes[0].set_yscale('log')
            
            # 범례
            axes[0].legend(loc='best')
            
            # 마지막 종가 표시
            try:
                last_close = float(dfp_to_plot['Close'].iloc[-1])
                axr = axes[0].twinx()
                axr.set_ylim(axes[0].get_ylim())
                axr.set_yticks([last_close])
                axr.set_yticklabels([f"{last_close:.2f}"], color='red')
                axr.tick_params(axis='y', colors='red', length=0)
                for spine in axr.spines.values():
                    spine.set_visible(False)
            except Exception:
                pass
            
            # 저장
            fig.savefig(out, bbox_inches='tight', dpi=self.config.dpi)
            
        finally:
            plt.close(fig)
        
        return out
