import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """
    주식 보조지표 계산 클래스
    
    주요 지표:
    - 이동평균: SMA, EMA, WMA
    - 볼린저 밴드
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Stochastic Oscillator
    - ATR (Average True Range)
    - Volume 관련 지표
    - OBV (On Balance Volume)
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        보조지표 계산 클래스 초기화
        
        Args:
            df: OHLCV 데이터프레임 (Date, Open, High, Low, Close, Volume 컬럼 필요)
        """
        self.df = df.copy()
        self._cache = {}
        self._validate_dataframe()
    
    def _validate_dataframe(self) -> None:
        """데이터프레임 유효성 검사"""
        required_cols = ['Close']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
    
    def _get_cache_key(self, name: str, *args, **kwargs) -> str:
        """캐시 키 생성"""
        params = '_'.join(str(v) for v in args)
        kw_params = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{name}_{params}_{kw_params}".rstrip('_')
    
    # ==================== 이동평균 지표 ====================
    
    def sma(self, window: int, column: str = 'Close') -> pd.Series:
        """
        단순 이동평균 (Simple Moving Average)
        
        Args:
            window: 이동평균 기간
            column: 계산할 컬럼명 (기본: 'Close')
            
        Returns:
            이동평균 시리즈
        """
        key = self._get_cache_key('sma', window, column=column)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            self._cache[key] = self.df[column].rolling(window=window).mean()
        return self._cache[key]
    
    def ema(self, window: int, column: str = 'Close') -> pd.Series:
        """
        지수 이동평균 (Exponential Moving Average)
        
        Args:
            window: 이동평균 기간
            column: 계산할 컬럼명 (기본: 'Close')
            
        Returns:
            지수 이동평균 시리즈
        """
        key = self._get_cache_key('ema', window, column=column)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            self._cache[key] = self.df[column].ewm(span=window, adjust=False).mean()
        return self._cache[key]
    
    def wma(self, window: int, column: str = 'Close') -> pd.Series:
        """
        가중 이동평균 (Weighted Moving Average)
        최근 데이터에 더 높은 가중치 부여
        
        Args:
            window: 이동평균 기간
            column: 계산할 컬럼명 (기본: 'Close')
            
        Returns:
            가중 이동평균 시리즈
        """
        key = self._get_cache_key('wma', window, column=column)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
            weights = np.arange(1, window + 1)
            
            def weighted_average(values):
                if len(values) < window:
                    return np.nan
                return np.sum(weights * values) / weights.sum()
            
            self._cache[key] = self.df[column].rolling(window=window).apply(weighted_average, raw=True)
        return self._cache[key]
    
    def rolling_max(
        self,
        window: int,
        column: str = 'Close',
        *,
        min_periods: Optional[int] = None,
    ) -> pd.Series:
        """
        구간 최대값 (inclusive rolling max)

        Args:
            window: 계산 기간
            column: 대상 컬럼 (기본: 'Close')
            min_periods: 최소 유효 데이터 수 (기본: window)
        """
        key = self._get_cache_key('rolling_max', window, column=column, min_periods=min_periods)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            mp = window if min_periods is None else min_periods
            self._cache[key] = self.df[column].rolling(window=window, min_periods=mp).max()
        return self._cache[key]
    
    def rolling_min(
        self,
        window: int,
        column: str = 'Close',
        *,
        min_periods: Optional[int] = None,
    ) -> pd.Series:
        """
        구간 최소값 (inclusive rolling min)

        Args:
            window: 계산 기간
            column: 대상 컬럼 (기본: 'Close')
            min_periods: 최소 유효 데이터 수 (기본: window)
        """
        key = self._get_cache_key('rolling_min', window, column=column, min_periods=min_periods)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            mp = window if min_periods is None else min_periods
            self._cache[key] = self.df[column].rolling(window=window, min_periods=mp).min()
        return self._cache[key]
    
    def new_high_signal(self, window: int = 252, column: str = 'Close') -> pd.Series:
        """
        신고가(rolling max 돌파) 여부를 반환하는 bool 시리즈

        Args:
            window: 52주 ≈ 252 거래일 등 돌파 기간
            column: 대상 컬럼 (기본: 'Close')
        """
        key = self._get_cache_key('new_high_signal', window, column=column)
        if key not in self._cache:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            rolling_max = self.rolling_max(window, column)
            prev_max = rolling_max.shift(1)
            prev_close = self.df[column].shift(1)
            signal = (
                (self.df[column] >= prev_max)
                & (prev_close < prev_max)
                & prev_max.notna()
            )
            self._cache[key] = signal.fillna(False)
        return self._cache[key]
    
    # ==================== 볼린저 밴드 ====================
    
    def bollinger_bands(self, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        볼린저 밴드 (Bollinger Bands)
        
        Args:
            window: 이동평균 기간 (기본: 20)
            num_std: 표준편차 배수 (기본: 2.0)
            
        Returns:
            (중심선, 상단밴드, 하단밴드) 튜플
        """
        key = self._get_cache_key('bb', window, num_std=num_std)
        if key not in self._cache:
            sma = self.sma(window)
            std = self.df['Close'].rolling(window=window).std()
            upper = sma + num_std * std
            lower = sma - num_std * std
            self._cache[key] = (sma, upper, lower)
        return self._cache[key]
    
    def bollinger_percent_b(self, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """
        볼린저 밴드 %B
        현재 가격이 밴드 내에서 어느 위치에 있는지 표시 (0~1)
        
        Args:
            window: 이동평균 기간 (기본: 20)
            num_std: 표준편차 배수 (기본: 2.0)
            
        Returns:
            %B 시리즈 (0: 하단밴드, 0.5: 중심선, 1: 상단밴드)
        """
        sma, upper, lower = self.bollinger_bands(window, num_std)
        return (self.df['Close'] - lower) / (upper - lower)
    
    def bollinger_bandwidth(self, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """
        볼린저 밴드 폭
        변동성 측정 지표
        
        Args:
            window: 이동평균 기간 (기본: 20)
            num_std: 표준편차 배수 (기본: 2.0)
            
        Returns:
            밴드폭 시리즈
        """
        sma, upper, lower = self.bollinger_bands(window, num_std)
        return (upper - lower) / sma
    
    # ==================== RSI ====================
    
    def rsi(self, window: int = 14) -> pd.Series:
        """
        상대 강도 지수 (Relative Strength Index)
        과매수/과매도 판단 지표 (0~100)
        
        Args:
            window: RSI 계산 기간 (기본: 14)
            
        Returns:
            RSI 시리즈 (70 이상: 과매수, 30 이하: 과매도)
        """
        key = self._get_cache_key('rsi', window)
        if key not in self._cache:
            delta = self.df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            self._cache[key] = rsi
        return self._cache[key]
    
    # ==================== MACD ====================
    
    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)
        추세 전환 및 모멘텀 지표
        
        Args:
            fast: 단기 EMA 기간 (기본: 12)
            slow: 장기 EMA 기간 (기본: 26)
            signal: 시그널선 EMA 기간 (기본: 9)
            
        Returns:
            (MACD선, 시그널선, 히스토그램) 튜플
        """
        key = self._get_cache_key('macd', fast, slow, signal)
        if key not in self._cache:
            ema_fast = self.ema(fast)
            ema_slow = self.ema(slow)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            self._cache[key] = (macd_line, signal_line, histogram)
        return self._cache[key]
    
    # ==================== Stochastic Oscillator ====================
    
    def stochastic(self, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        스토캐스틱 오실레이터 (Stochastic Oscillator)
        과매수/과매도 판단 지표 (0~100)
        
        Args:
            k_window: %K 계산 기간 (기본: 14)
            d_window: %D 계산 기간 (기본: 3)
            
        Returns:
            (%K, %D) 튜플 (80 이상: 과매수, 20 이하: 과매도)
        """
        key = self._get_cache_key('stoch', k_window, d_window)
        if key not in self._cache:
            if 'High' not in self.df.columns or 'Low' not in self.df.columns:
                raise ValueError("DataFrame must contain 'High' and 'Low' columns for Stochastic")
            
            low_min = self.df['Low'].rolling(window=k_window).min()
            high_max = self.df['High'].rolling(window=k_window).max()
            
            k_percent = 100 * (self.df['Close'] - low_min) / (high_max - low_min)
            d_percent = k_percent.rolling(window=d_window).mean()
            
            self._cache[key] = (k_percent, d_percent)
        return self._cache[key]
    
    # ==================== ATR ====================
    
    def atr(self, window: int = 14) -> pd.Series:
        """
        평균 진폭 (Average True Range)
        변동성 측정 지표
        
        Args:
            window: ATR 계산 기간 (기본: 14)
            
        Returns:
            ATR 시리즈
        """
        key = self._get_cache_key('atr', window)
        if key not in self._cache:
            if 'High' not in self.df.columns or 'Low' not in self.df.columns:
                raise ValueError("DataFrame must contain 'High' and 'Low' columns for ATR")
            
            high_low = self.df['High'] - self.df['Low']
            high_close = np.abs(self.df['High'] - self.df['Close'].shift())
            low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            self._cache[key] = atr
        return self._cache[key]
    
    # ==================== Volume 지표 ====================
    
    def volume_sma(self, window: int = 20) -> pd.Series:
        """
        거래량 이동평균
        
        Args:
            window: 이동평균 기간 (기본: 20)
            
        Returns:
            거래량 이동평균 시리즈
        """
        if 'Volume' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Volume' column")
        return self.sma(window, column='Volume')
    
    def volume_ratio(self, window: int = 20) -> pd.Series:
        """
        거래량 비율 (현재 거래량 / 평균 거래량)
        
        Args:
            window: 평균 계산 기간 (기본: 20)
            
        Returns:
            거래량 비율 시리즈 (1 이상: 평균보다 많은 거래량)
        """
        if 'Volume' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Volume' column")
        return self.df['Volume'] / self.volume_sma(window)
    
    def obv(self) -> pd.Series:
        """
        OBV (On Balance Volume)
        거래량 누적 지표, 가격 상승 시 거래량 더하고 하락 시 빼기
        
        Returns:
            OBV 시리즈
        """
        key = self._get_cache_key('obv')
        if key not in self._cache:
            if 'Volume' not in self.df.columns:
                raise ValueError("DataFrame must contain 'Volume' column")
            
            obv = pd.Series(index=self.df.index, dtype=float)
            obv.iloc[0] = self.df['Volume'].iloc[0]
            
            for i in range(1, len(self.df)):
                if self.df['Close'].iloc[i] > self.df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + self.df['Volume'].iloc[i]
                elif self.df['Close'].iloc[i] < self.df['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - self.df['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            self._cache[key] = obv
        return self._cache[key]
    
    # ==================== 기타 지표 ====================
    
    def rate_of_change(self, window: int = 10) -> pd.Series:
        """
        변화율 (Rate of Change, ROC)
        n일 전 대비 가격 변화율 (%)
        
        Args:
            window: 비교 기간 (기본: 10)
            
        Returns:
            ROC 시리즈 (%)
        """
        key = self._get_cache_key('roc', window)
        if key not in self._cache:
            self._cache[key] = ((self.df['Close'] - self.df['Close'].shift(window)) / 
                               self.df['Close'].shift(window) * 100)
        return self._cache[key]
    
    def momentum(self, window: int = 10) -> pd.Series:
        """
        모멘텀 (Momentum)
        n일 전 대비 가격 차이
        
        Args:
            window: 비교 기간 (기본: 10)
            
        Returns:
            모멘텀 시리즈
        """
        key = self._get_cache_key('momentum', window)
        if key not in self._cache:
            self._cache[key] = self.df['Close'] - self.df['Close'].shift(window)
        return self._cache[key]
    
    def typical_price(self) -> pd.Series:
        """
        대표 가격 (Typical Price)
        (High + Low + Close) / 3
        
        Returns:
            대표 가격 시리즈
        """
        key = self._get_cache_key('typical_price')
        if key not in self._cache:
            if 'High' not in self.df.columns or 'Low' not in self.df.columns:
                raise ValueError("DataFrame must contain 'High' and 'Low' columns")
            self._cache[key] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        return self._cache[key]
    
    # ==================== Relative Strength (RS) ====================
    
    def mansfield_rs(self, benchmark_df: pd.DataFrame, window: int = 52) -> pd.Series:
        """
        Mansfield Relative Strength (상대 강도)
        자산의 벤치마크 대비 상대적 성과를 측정
        
        Args:
            benchmark_df: 벤치마크의 OHLCV 데이터프레임 (Date, Close 컬럼 필요)
            window: 이동평균 기간 (기본: 52주 ≈ 1년)
            
        Returns:
            RS 시리즈 
            - 양수: 벤치마크보다 강함 (outperform)
            - 음수: 벤치마크보다 약함 (underperform)
            - 0: 벤치마크와 동일한 성과
            
        Example:
            # KOSPI 대비 삼성전자의 상대 강도
            samsung = Stock('005930.KS')
            samsung.get_history(365)
            
            kospi = Stock('102110.KS')  # KOSPI 200 ETF
            kospi.get_history(365)
            
            rs = samsung.indicators.mansfield_rs(kospi.history, window=52)
            
            # RS > 0: 삼성전자가 KOSPI를 아웃퍼폼
            # RS < 0: 삼성전자가 KOSPI를 언더퍼폼
        """
        key = self._get_cache_key('mansfield_rs', window, benchmark_id=id(benchmark_df))
        if key not in self._cache:
            if 'Date' not in self.df.columns or 'Date' not in benchmark_df.columns:
                raise ValueError("Both DataFrames must contain 'Date' column")
            if 'Close' not in benchmark_df.columns:
                raise ValueError("Benchmark DataFrame must contain 'Close' column")
            
            # 자산과 벤치마크의 종가 데이터 병합
            a = self.df[['Date', 'Close']].rename(columns={'Close': 'A'})
            b = benchmark_df[['Date', 'Close']].rename(columns={'Close': 'B'})
            m = pd.merge(a, b, on='Date', how='inner').set_index('Date')
            
            # 상대 강도 비율 (자산 / 벤치마크)
            rs_ratio = (m['A'] / m['B']).astype(float)
            
            # 이동평균 기준선
            base = rs_ratio.rolling(window=window, min_periods=1).mean()
            
            # Mansfield RS: (현재 비율 / 기준선) - 1
            mansfield = rs_ratio / base - 1.0
            
            # 원본 인덱스에 맞춰 재정렬
            mansfield = mansfield.reindex(self.df['Date']).reset_index(drop=True)
            
            self._cache[key] = mansfield
        return self._cache[key]

    def minervini_trend_template(
        self,
        latest_rs: float,
        rs_threshold: float = 0.10,
        pct_from_low: float = 0.25,
        pct_within_high: float = 0.25,
    ) -> dict:
        """
        Mark Minervini Trend Template 평가.

        Args:
            latest_rs: 최근 Mansfield RS 값
            rs_threshold: RS 최소 임계값
            pct_from_low: 52주 저점 대비 상승 비율 (기본 25%)
            pct_within_high: 52주 고점 대비 이격 허용치 (기본 25%)

        Returns:
            Template 충족 여부, 실패 조건, 주요 지표가 담긴 dict
        """
        if self.df is None or self.df.empty or len(self.df) < 200:
            return {"pass": False, "failed": ["insufficient_data"]}

        df = self.df.sort_values('Date') if 'Date' in self.df.columns else self.df.copy()
        df = df.reset_index(drop=True)
        close = df['Close']

        sma50 = close.rolling(50, min_periods=1).mean()
        sma150 = close.rolling(150, min_periods=1).mean()
        sma200 = close.rolling(200, min_periods=1).mean()

        last_close = float(close.iloc[-1])
        cond_price_above50 = last_close > float(sma50.iloc[-1])
        cond_price_above150 = last_close > float(sma150.iloc[-1])
        cond_price_above200 = last_close > float(sma200.iloc[-1])
        cond_ma_alignment = float(sma50.iloc[-1]) > float(sma150.iloc[-1]) > float(sma200.iloc[-1])

        lookback = min(20, len(sma200) - 1)
        cond_ma200_up = float(sma200.iloc[-1]) > float(sma200.iloc[-1 - lookback]) if lookback > 0 else False

        window_52w = min(252, len(close))
        last_window = close.iloc[-window_52w:]
        low_52w = float(last_window.min())
        high_52w = float(last_window.max())
        cond_above_low = last_close >= low_52w * (1.0 + pct_from_low)
        cond_within_high = last_close >= high_52w * (1.0 - pct_within_high)

        cond_rs_strong = float(latest_rs) >= float(rs_threshold)

        failed = []
        if not cond_price_above50:
            failed.append('price>SMA50')
        if not cond_price_above150:
            failed.append('price>SMA150')
        if not cond_price_above200:
            failed.append('price>SMA200')
        if not cond_ma_alignment:
            failed.append('SMA50>SMA150>SMA200')
        if not cond_ma200_up:
            failed.append('SMA200_up')
        if not cond_above_low:
            failed.append(f'>52wLow+{int(pct_from_low*100)}%')
        if not cond_within_high:
            failed.append(f'within_{int(pct_within_high*100)}%_of_52wHigh')
        if not cond_rs_strong:
            failed.append(f'RS>={rs_threshold:.2f}')

        return {
            "pass": len(failed) == 0,
            "failed": failed,
            "last_close": last_close,
            "sma50": float(sma50.iloc[-1]),
            "sma150": float(sma150.iloc[-1]),
            "sma200": float(sma200.iloc[-1]),
            "low_52w": low_52w,
            "high_52w": high_52w,
            "latest_rs": float(latest_rs),
            "params": {
                "rs_threshold": rs_threshold,
                "pct_from_low": pct_from_low,
                "pct_within_high": pct_within_high,
            },
        }

    
    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
    
    def get_all_indicators(self, 
                          sma_windows: list = [5, 10, 20, 60, 120],
                          ema_windows: list = [12, 26],
                          bb_window: int = 20,
                          rsi_window: int = 14,
                          macd_params: tuple = (12, 26, 9),
                          stoch_params: tuple = (14, 3),
                          atr_window: int = 14,
                          volume_sma_window: int = 20) -> pd.DataFrame:
        """
        주요 보조지표를 모두 계산하여 데이터프레임으로 반환
        
        Args:
            sma_windows: SMA 계산 기간 목록
            ema_windows: EMA 계산 기간 목록
            bb_window: 볼린저 밴드 기간
            rsi_window: RSI 기간
            macd_params: MACD 파라미터 (fast, slow, signal)
            stoch_params: Stochastic 파라미터 (k_window, d_window)
            atr_window: ATR 기간
            volume_sma_window: 거래량 이동평균 기간
            
        Returns:
            모든 지표가 포함된 데이터프레임
        """
        result = self.df.copy()
        
        # SMA
        for window in sma_windows:
            result[f'SMA_{window}'] = self.sma(window)
        
        # EMA
        for window in ema_windows:
            result[f'EMA_{window}'] = self.ema(window)
        
        # 볼린저 밴드
        bb_mid, bb_upper, bb_lower = self.bollinger_bands(bb_window)
        result[f'BB_Mid_{bb_window}'] = bb_mid
        result[f'BB_Upper_{bb_window}'] = bb_upper
        result[f'BB_Lower_{bb_window}'] = bb_lower
        result[f'BB_%B_{bb_window}'] = self.bollinger_percent_b(bb_window)
        result[f'BB_BW_{bb_window}'] = self.bollinger_bandwidth(bb_window)
        
        # RSI
        result[f'RSI_{rsi_window}'] = self.rsi(rsi_window)
        
        # MACD
        macd_line, signal_line, histogram = self.macd(*macd_params)
        result['MACD'] = macd_line
        result['MACD_Signal'] = signal_line
        result['MACD_Hist'] = histogram
        
        # Stochastic
        if 'High' in self.df.columns and 'Low' in self.df.columns:
            k, d = self.stochastic(*stoch_params)
            result['Stoch_K'] = k
            result['Stoch_D'] = d
            
            # ATR
            result[f'ATR_{atr_window}'] = self.atr(atr_window)
        
        # Volume 지표
        if 'Volume' in self.df.columns:
            result[f'Volume_SMA_{volume_sma_window}'] = self.volume_sma(volume_sma_window)
            result[f'Volume_Ratio_{volume_sma_window}'] = self.volume_ratio(volume_sma_window)
            result['OBV'] = self.obv()
        
        return result
