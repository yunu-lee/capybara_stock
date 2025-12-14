from functools import lru_cache
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
import re
from pykrx import stock as pykrx_stock
from core.data import DataFetcher
from core.data_source import DataSource
from core.indicators import TechnicalIndicators
from core.chart import ChartRenderer, ChartConfig, TradeSignal


def _default_benchmark(ticker: str) -> str:
    """티커에 따라 기본 벤치마크 반환"""
    return '102110.KS' if ticker.endswith('.KS') or ticker.endswith('.KQ') else 'SPY'


@lru_cache(maxsize=128)
def _resolve_name_from_yf(ticker: str) -> str:
    """Try to fetch a human-readable company name from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return ""
    
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        return ""
    
    # Prefer longName, fall back to shortName or symbol-like fields
    for key in ("longName", "shortName", "name"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    
    # Some tickers expose symbol in 'symbol' field
    value = info.get("symbol")
    if isinstance(value, str) and value.strip():
        return value.strip()
    
    return ""


class Stock:
    """Stock class with ticker, name, and price list"""
    
    _market_cap_cache: Dict[str, Tuple[float, datetime]] = {}
    _market_cap_cache_ttl = timedelta(hours=6)
    
    def __init__(self, ticker: str, name: str = "", price: List[float] = None, 
                 data_source: Optional[DataSource] = None, benchmark: Optional[str] = None):
        """
        Initialize Stock instance
        
        Args:
            ticker: Stock ticker symbol
            name: Stock name
            price: List of prices (default: empty list)
            data_source: Data source implementation (optional, uses default YFinanceDataSource if not provided)
            benchmark: Benchmark ticker for RS calculation (default: auto-detected)
        """
        self.ticker = ticker
        if name:
            self.name = name
        else:
            resolved = _resolve_name_from_yf(ticker)
            self.name = resolved if resolved else ticker
        self.price = price if price is not None else []
        self.benchmark = benchmark if benchmark is not None else _default_benchmark(ticker)
        self.data_fetcher = DataFetcher(data_source=data_source)
        self._history: Optional[pd.DataFrame] = None
        self._indicators: Optional[TechnicalIndicators] = None
    
    @staticmethod
    def normalize_code(code: str) -> str:
        if not code:
            return ""
        code = code.strip()
        match = re.search(r'\d{6}', code)
        return match.group(0) if match else ""
    
    @staticmethod
    def _parse_base_date(date: Optional[str]) -> datetime:
        if date is None:
            return datetime.now()
        if isinstance(date, datetime):
            return date
        if isinstance(date, str):
            value = date.strip()
            for fmt in ("%Y%m%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        print(f"지원하지 않는 날짜 형식: {date}. 현재 날짜를 사용합니다.")
        return datetime.now()
    
    @staticmethod
    def _candidate_dates(base_date: datetime, lookback: int = 5):
        for offset in range(lookback):
            yield base_date - timedelta(days=offset)
    
    @classmethod
    def fetch_market_caps(cls, codes: List[str], date: Optional[str] = None) -> Dict[str, float]:
        normalized_order: List[str] = []
        seen = set()
        for code in codes:
            norm = cls.normalize_code(code)
            if norm and norm not in seen:
                normalized_order.append(norm)
                seen.add(norm)
        
        if not normalized_order:
            return {}
        
        now = datetime.now()
        result: Dict[str, float] = {}
        missing: List[str] = []
        
        for code in normalized_order:
            cached = cls._market_cap_cache.get(code)
            if cached and now - cached[1] < cls._market_cap_cache_ttl:
                result[code] = cached[0]
            else:
                missing.append(code)
        
        base_date = cls._parse_base_date(date)
        
        for code in missing:
            data_source = "KRX"
            market_cap = cls._fetch_market_cap_from_krx(code, base_date)
            if not market_cap:
                print(f"{code}: KRX 시가총액 조회 실패, Yahoo Finance 대체 시도")
                data_source = "Yahoo Finance"
                market_cap = cls._fetch_market_cap_from_yfinance(code)
            if market_cap and market_cap > 0:
                print(f"{code}: 시가총액 {market_cap:,.0f}억원 ({data_source})")
            else:
                print(f"{code}: 시가총액 데이터를 찾을 수 없습니다.")
            result[code] = market_cap
            cls._market_cap_cache[code] = (market_cap, now)
        
        return {code: result.get(code, 0.0) for code in normalized_order}
    
    @classmethod
    def _fetch_market_cap_from_krx(cls, code: str, base_date: datetime) -> float:
        for target_date in cls._candidate_dates(base_date):
            end_date = target_date.strftime('%Y%m%d')
            start_date = (target_date - timedelta(days=5)).strftime('%Y%m%d')
            try:
                df = pykrx_stock.get_market_cap(start_date, end_date, code)
            except Exception as e:
                print(f"{code}: 시가총액 조회 실패 ({start_date}~{end_date}) - {e}")
                continue
            
            if df is None or df.empty:
                continue
            
            latest = df.iloc[-1]
            raw_value = latest.get('시가총액', 0)
            try:
                raw_value = float(raw_value)
            except (TypeError, ValueError):
                raw_value = 0
            if raw_value and raw_value > 0:
                return raw_value / 100_000_000
        return 0.0
    
    @staticmethod
    def _fetch_market_cap_from_yfinance(code: str) -> float:
        try:
            import yfinance as yf
        except ImportError:
            return 0.0
        
        suffixes = [".KS", ".KQ", ""]
        for suffix in suffixes:
            ticker = f"{code}{suffix}"
            try:
                ticker_obj = yf.Ticker(ticker)
                market_cap = 0.0
                try:
                    fast_info = getattr(ticker_obj, "fast_info", None)
                    if fast_info:
                        market_cap = fast_info.get('market_cap') or fast_info.get('marketCap') or 0.0
                except Exception:
                    market_cap = 0.0
                
                if not market_cap:
                    try:
                        info = ticker_obj.get_info()
                        market_cap = info.get('marketCap', 0)
                    except Exception:
                        market_cap = 0.0
                
                if market_cap and market_cap > 0:
                    return float(market_cap) / 100_000_000
            except Exception:
                continue
        return 0.0
    
    def get_market_cap(self, date: Optional[str] = None) -> float:
        code = self.normalize_code(self.ticker)
        if not code:
            return 0.0
        return self.fetch_market_caps([code], date=date).get(code, 0.0)
    
    def getPrice(self, n: int) -> List[float]:
        """
        Fetch price history for the past n days and store in price list
        
        Args:
            n: Number of days to fetch
            
        Returns:
            List of prices (Close prices)
        """
        df = self.data_fetcher.get_history(self.ticker, period_days=n)
        
        if df.empty or 'Close' not in df.columns:
            return []
        
        # Store full history for indicators
        self._history = df
        self._indicators = None  # Reset indicators cache when history changes
        
        # Extract Close prices and convert to list
        prices = df['Close'].tolist()
        self.price = prices
        
        return prices
    
    def get_history(self, period_days: int = 365) -> pd.DataFrame:
        """
        Fetch full OHLCV history and store in _history
        
        Args:
            period_days: Number of days to fetch (default: 365)
            
        Returns:
            DataFrame with OHLCV data
        """
        df = self.data_fetcher.get_history(self.ticker, period_days=period_days)
        self._history = df
        self._indicators = None  # Reset indicators cache when history changes
        return df
    
    @property
    def history(self) -> Optional[pd.DataFrame]:
        """
        가격 히스토리 데이터프레임
        
        Returns:
            OHLCV 데이터프레임 또는 None
        """
        return self._history
    
    @property
    def indicators(self) -> TechnicalIndicators:
        """
        보조지표 계산 객체 (Property 패턴)
        
        사용 예:
            stock = Stock('005930.KS')
            stock.get_history(100)
            sma20 = stock.indicators.sma(20)
            rsi = stock.indicators.rsi(14)
            bb_mid, bb_upper, bb_lower = stock.indicators.bollinger_bands()
        
        Returns:
            TechnicalIndicators 객체
            
        Raises:
            RuntimeError: history가 로드되지 않은 경우
        """
        if self._history is None or self._history.empty:
            raise RuntimeError('History not loaded. Call get_history() or getPrice() first.')
        
        # 캐싱: history가 변경되지 않으면 같은 객체 재사용
        if self._indicators is None:
            self._indicators = TechnicalIndicators(self._history)
        
        return self._indicators
    
    def compute_rs(self, window: int = 52) -> Optional[pd.Series]:
        """
        Mansfield Relative Strength 계산
        벤치마크 대비 상대 강도 측정
        
        Args:
            window: 이동평균 기간 (기본: 52주 ≈ 1년)
            
        Returns:
            RS 시리즈 (0 이상: 벤치마크보다 강함, 0 미만: 약함)
            
        Raises:
            RuntimeError: history가 로드되지 않은 경우
            
        Example:
            stock = Stock('005930.KS', name='삼성전자')
            stock.get_history(365)
            rs = stock.compute_rs(window=52)
            
            # RS 해석
            if rs.iloc[-1] > 0:
                print("벤치마크 대비 강세")
            else:
                print("벤치마크 대비 약세")
        """
        if self._history is None or self._history.empty:
            raise RuntimeError('History not loaded. Call get_history() or getPrice() first.')
        
        # 충분한 기간의 벤치마크 데이터 가져오기 (window의 3배)
        period_days = max(len(self._history) * 3, window * 7)  # 주간 단위이므로 *7
        bench_df = self.data_fetcher.get_history(self.benchmark, period_days=period_days)
        
        return self.data_fetcher.get_mansfield_rs(self._history, bench_df, window=window)
    
    def render_chart(
        self,
        config: Optional[ChartConfig] = None,
        trade_signals: Optional[List[TradeSignal]] = None,
        display_days: Optional[int] = None,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """
        차트 렌더링
        
        Args:
            config: 차트 설정 (None이면 기본 설정)
            trade_signals: 매매 신호 리스트 (TradeSignal 객체들)
            display_days: 표시할 일수 (None이면 전체)
            save_path: 저장 경로 (None이면 'output/chart.png')
            title: 차트 제목 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
            
        Raises:
            RuntimeError: history가 로드되지 않은 경우
            
        Example:
            # 기본 캔들 차트
            stock = Stock('005930.KS', name='삼성전자')
            stock.get_history(200)
            stock.render_chart()
            
            # 매매 신호 포함
            from core.chart import TradeSignal
            from datetime import datetime
            
            signals = [
                TradeSignal(date=datetime(2025, 11, 1), price=100000, type='buy'),
                TradeSignal(date=datetime(2025, 11, 15), price=105000, type='sell'),
            ]
            stock.render_chart(trade_signals=signals)
            
            # 커스텀 설정
            config = ChartConfig(
                chart_type='line',
                show_sma=True,
                sma_windows=[20, 60],
                show_rsi=True,
                show_macd=True,
            )
            stock.render_chart(config=config, trade_signals=signals, display_days=100)
        """
        if self._history is None or self._history.empty:
            raise RuntimeError('History not loaded. Call get_history() or getPrice() first.')
        
        # config가 없으면 기본 생성
        if config is None:
            config = ChartConfig()
        
        # 차트 렌더러 생성
        renderer = ChartRenderer(config)
        
        # 보조지표 데이터 수집
        indicators_data: Dict[str, pd.Series] = {}
        
        # 필터링된 인덱스 계산
        if display_days is not None and display_days > 0:
            cutoff = self._history['Date'].max() - pd.Timedelta(days=display_days)
            filtered_history = self._history[self._history['Date'] >= cutoff]
        else:
            filtered_history = self._history
        
        # SMA
        if config.show_sma:
            for window in config.sma_windows:
                sma_full = self.indicators.sma(window)
                # 인덱스를 Date로 변환하여 저장
                sma_series = pd.Series(sma_full.values, index=self._history['Date'].values)
                indicators_data[f'sma_{window}'] = sma_series
        
        # EMA
        if config.show_ema:
            for window in config.ema_windows:
                ema_full = self.indicators.ema(window)
                ema_series = pd.Series(ema_full.values, index=self._history['Date'].values)
                indicators_data[f'ema_{window}'] = ema_series
        
        # 볼린저 밴드
        if config.show_bollinger:
            bb_mid, bb_upper, bb_lower = self.indicators.bollinger_bands(
                window=config.bb_window,
                num_std=config.bb_std
            )
            indicators_data['bb_mid'] = pd.Series(bb_mid.values, index=self._history['Date'].values)
            indicators_data['bb_upper'] = pd.Series(bb_upper.values, index=self._history['Date'].values)
            indicators_data['bb_lower'] = pd.Series(bb_lower.values, index=self._history['Date'].values)
        
        # RSI
        if config.show_rsi:
            rsi_full = self.indicators.rsi(window=config.rsi_window)
            indicators_data['rsi'] = pd.Series(rsi_full.values, index=self._history['Date'].values)
        
        # MACD
        if config.show_macd:
            macd, signal, hist = self.indicators.macd(
                fast=config.macd_fast,
                slow=config.macd_slow,
                signal=config.macd_signal
            )
            indicators_data['macd'] = pd.Series(macd.values, index=self._history['Date'].values)
            indicators_data['macd_signal'] = pd.Series(signal.values, index=self._history['Date'].values)
            indicators_data['macd_hist'] = pd.Series(hist.values, index=self._history['Date'].values)
        
        # Stochastic
        if config.show_stochastic:
            k, d = self.indicators.stochastic(
                k_window=config.stoch_k_window,
                d_window=config.stoch_d_window
            )
            indicators_data['stoch_k'] = pd.Series(k.values, index=self._history['Date'].values)
            indicators_data['stoch_d'] = pd.Series(d.values, index=self._history['Date'].values)
        
        # RS
        if config.show_rs:
            rs_full = self.compute_rs(window=config.rs_window)
            # RS는 벤치마크와 merge되므로 길이가 다를 수 있음
            # 원본 데이터의 Date에 맞춰서 reindex
            rs_series = pd.Series(rs_full.values, index=rs_full.index)
            rs_series = rs_series.reindex(self._history['Date'].values)
            indicators_data['rs'] = rs_series
        
        # 거래량 비율
        if config.show_volume_ratio:
            vol_ratio_full = self.indicators.volume_ratio(window=config.volume_ratio_window)
            indicators_data['volume_ratio'] = pd.Series(vol_ratio_full.values, index=self._history['Date'].values)
        
        # 제목 자동 생성
        if title is None:
            title_parts = []
            if self.name and self.name != self.ticker:
                title_parts.append(f"{self.name} ({self.ticker})")
            else:
                title_parts.append(self.ticker)
            
            # 마지막 날짜 추가
            try:
                last_date = self._history['Date'].max()
                title_parts.append(last_date.strftime('%Y-%m-%d'))
            except Exception:
                pass
            
            title = " | ".join(title_parts)
        
        # 차트 렌더링
        return renderer.render(
            df=self._history,
            indicators_data=indicators_data,
            trade_signals=trade_signals,
            display_days=display_days,
            title=title,
            save_path=save_path,
        )
    
    def __repr__(self) -> str:
        return f"Stock(ticker='{self.ticker}', name='{self.name}', price_count={len(self.price)})"

