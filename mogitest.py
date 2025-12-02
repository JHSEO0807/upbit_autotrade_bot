#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
변동성 돌파 전략 자동매매 봇 (개선 버전)
- 로깅 시스템
- 에러 처리 강화
- API 레이트 리밋 대응
- 상태 저장/복구
- 메모리 관리 개선
"""

import time
import json
import logging
import sys
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pyupbit

# ======== 설정값 ========
ACCESS_KEY = "YOUR_ACCESS_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"

DRY_RUN = True  # ★ 기본 True (모의매매). 실전 주문시 False 로 변경 ★

INTERVAL = "minute15"  # "minute5", "minute15", "minute60", "day" 등
K = 0.5  # 변동성 계수
SLEEP_SEC = 10  # 메인 루프 대기 시간(초)
UNIVERSE_REFRESH_MIN = 60  # 유니버스 재계산 주기 (분) - 1분에서 60분으로 변경 (API 호출 감소)

INITIAL_VIRTUAL_KRW = 1_000_000  # DRY_RUN 가상 초기 자본
ORDER_KRW_PORTION = 0.3  # 매수 시 보유 KRW 의 30%를 한 종목에 투자

VOLUME_THRESHOLD = 20_000_000_000  # 거래대금 200억

# API 재시도 설정
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 초
BACKOFF_FACTOR = 2.0

# 상태 저장 파일
STATE_FILE = Path(__file__).parent / "trading_state.json"

# 로깅 설정
LOG_FILE = Path(__file__).parent / "trading.log"


# ======== 로깅 설정 ========
def setup_logging():
    """로깅 시스템 초기화"""
    logger = logging.getLogger("VolatilityBot")
    logger.setLevel(logging.INFO)

    # 파일 핸들러
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # DEBUG로 변경하여 더 많은 정보 출력
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ======== 유틸리티 함수 ========
def retry_on_failure(func, max_retries=MAX_RETRIES, delay=RETRY_DELAY,
                     backoff=BACKOFF_FACTOR, logger=None):
    """
    함수 실행 실패 시 재시도
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(
                    f"함수 {func.__name__} 실행 실패 (시도 {attempt + 1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                time.sleep(current_delay)
                current_delay *= backoff

    if logger:
        logger.error(f"함수 {func.__name__} 최종 실패: {last_exception}")
    return None


def validate_price(price: Optional[float]) -> bool:
    """가격 유효성 검증"""
    return price is not None and price > 0 and not np.isnan(price) and not np.isinf(price)


def validate_dataframe(df, min_length: int = 1) -> bool:
    """데이터프레임 유효성 검증"""
    if df is None:
        return False
    if len(df) < min_length:
        return False
    if df.isnull().all().any():  # 모든 값이 NaN인 컬럼이 있는지
        return False
    return True


# ======== 메인 트레이딩 클래스 ========
class VolatilityBreakoutBot:
    """변동성 돌파 전략 자동매매 봇"""

    def __init__(self):
        self.upbit: Optional[pyupbit.Upbit] = None
        self.universe: List[str] = []
        self.last_universe_update: Optional[datetime] = None

        # 가상 자산 (DRY_RUN)
        self.virtual_krw: float = INITIAL_VIRTUAL_KRW
        self.virtual_coin: Dict[str, float] = {}

        # 포지션 관리
        self.in_position: Dict[str, bool] = {}
        self.current_bar_time: Dict[str, datetime] = {}
        self.entry_price_map: Dict[str, Optional[float]] = {}
        self.invested_krw: Dict[str, float] = {}

        # 성능 최적화를 위한 캐시
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._price_cache_ttl = 5  # 초

        self.init_upbit()
        self.load_state()

    def init_upbit(self):
        """Upbit 초기화"""
        if DRY_RUN:
            logger.info("DRY_RUN 모드입니다. 실제 주문은 전혀 실행되지 않습니다.")
            self.upbit = None
        else:
            try:
                self.upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
                logger.info("실전 모드입니다. 실제 주문이 실행됩니다. 반드시 소액으로 테스트부터 하세요.")
            except Exception as e:
                logger.error(f"Upbit 초기화 실패: {e}")
                raise

    def load_state(self):
        """저장된 상태 불러오기"""
        if not STATE_FILE.exists():
            logger.info("저장된 상태 파일이 없습니다. 초기 상태로 시작합니다.")
            return

        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)

            if DRY_RUN:
                self.virtual_krw = state.get('virtual_krw', INITIAL_VIRTUAL_KRW)
                self.virtual_coin = state.get('virtual_coin', {})

            self.in_position = state.get('in_position', {})
            self.invested_krw = state.get('invested_krw', {})

            logger.info(f"상태 복구 완료: KRW={self.virtual_krw:,.0f}, 포지션={len(self.in_position)}")
        except Exception as e:
            logger.error(f"상태 파일 로드 실패: {e}. 초기 상태로 시작합니다.")

    def save_state(self):
        """현재 상태 저장"""
        try:
            state = {
                'virtual_krw': self.virtual_krw,
                'virtual_coin': self.virtual_coin,
                'in_position': self.in_position,
                'invested_krw': self.invested_krw,
                'timestamp': datetime.now().isoformat()
            }

            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")

    def cleanup_old_positions(self):
        """더 이상 유니버스에 없는 종목의 포지션 정리"""
        # 유니버스에 없는 종목들 찾기
        to_remove = [ticker for ticker in self.in_position.keys()
                     if ticker not in self.universe]

        for ticker in to_remove:
            logger.info(f"유니버스에서 제거된 종목 {ticker}의 데이터 정리")
            self.in_position.pop(ticker, None)
            self.current_bar_time.pop(ticker, None)
            self.entry_price_map.pop(ticker, None)

            # 포지션이 있으면 청산
            if DRY_RUN:
                if self.virtual_coin.get(ticker, 0) > 0:
                    logger.warning(f"{ticker} 포지션 강제 청산 (유니버스 제거)")
                    self.sell_market(ticker)

    def get_current_price(self, ticker: str, use_cache: bool = True) -> Optional[float]:
        """현재가 조회 (캐싱 지원)"""
        now = datetime.now()

        # 캐시 확인
        if use_cache and ticker in self._price_cache:
            cached_price, cached_time = self._price_cache[ticker]
            if (now - cached_time).total_seconds() < self._price_cache_ttl:
                return cached_price

        # API 호출
        def fetch_price():
            return pyupbit.get_current_price(ticker)

        price = retry_on_failure(fetch_price, logger=logger)

        if validate_price(price):
            self._price_cache[ticker] = (price, now)
            return price

        logger.warning(f"[{ticker}] 유효하지 않은 현재가: {price}")
        return None

    def build_universe(self):
        """유니버스 구축 (최적화 버전 - 업비트 ticker API 활용)"""
        logger.info("유니버스 계산 시작...")

        try:
            # 1단계: KRW 마켓 티커 목록 가져오기
            def fetch_tickers():
                return pyupbit.get_tickers(fiat="KRW")

            tickers = retry_on_failure(fetch_tickers, max_retries=5, delay=3.0, logger=logger)
            if not tickers:
                logger.error("티커 목록 조회 실패. 네트워크 연결을 확인하세요.")
                logger.warning("기존 유니버스를 유지합니다.")
                return

            logger.info(f"총 {len(tickers)}개 KRW 마켓 종목 발견")

            # 2단계: 업비트 ticker API로 한번에 모든 종목 정보 조회
            def fetch_ticker_data():
                # 업비트 API는 한번에 100개까지 조회 가능하므로 분할
                all_ticker_data = []
                batch_size = 100

                for i in range(0, len(tickers), batch_size):
                    batch = tickers[i:i + batch_size]
                    markets = ','.join(batch)
                    url = f"https://api.upbit.com/v1/ticker?markets={markets}"

                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    all_ticker_data.extend(response.json())

                    # API 호출 간격
                    if i + batch_size < len(tickers):
                        time.sleep(0.1)

                return all_ticker_data

            ticker_data = retry_on_failure(fetch_ticker_data, max_retries=5, delay=3.0, logger=logger)
            if not ticker_data:
                logger.error("Ticker 정보 조회 실패. 네트워크 연결을 확인하세요.")
                logger.warning("기존 유니버스를 유지합니다.")
                return

            # 3단계: 조건에 맞는 종목 필터링
            selected = []
            skip_reasons = {"등락률음수": 0, "거래대금부족": 0, "데이터없음": 0}

            for data in ticker_data:
                try:
                    market = data.get('market')
                    change_rate = data.get('signed_change_rate', 0) * 100  # 소수점 -> 퍼센트
                    acc_trade_price_24h = data.get('acc_trade_price_24h', 0)  # 24시간 누적 거래대금

                    # 조건 체크
                    if change_rate > 0 and acc_trade_price_24h >= VOLUME_THRESHOLD:
                        selected.append(market)
                        logger.info(f"✓ {market}: 등락률={change_rate:+.2f}%, 거래대금={acc_trade_price_24h/1e9:.1f}억원 → 유니버스 추가")
                    else:
                        if change_rate <= 0:
                            skip_reasons["등락률음수"] += 1
                        elif acc_trade_price_24h < VOLUME_THRESHOLD:
                            skip_reasons["거래대금부족"] += 1

                except Exception as e:
                    skip_reasons["데이터없음"] += 1
                    logger.debug(f"종목 데이터 파싱 실패: {e}")
                    continue

            self.universe = selected
            self.last_universe_update = datetime.now()

            logger.info("=" * 60)
            logger.info(f"유니버스 업데이트 완료: {len(self.universe)}개 종목")
            logger.info(f"총 검사: {len(ticker_data)}개 | 스킵 사유: {skip_reasons}")
            if self.universe:
                logger.info(f"✓ 감시 리스트: {', '.join(self.universe)}")
            else:
                logger.warning("⚠ 조건을 만족하는 종목이 없습니다!")
                logger.info(f"현재 조건: 전일대비 상승률 > 0%, 24시간 거래대금 >= {VOLUME_THRESHOLD/1e9:.0f}억원")
            logger.info("=" * 60)

            # 유니버스에서 제거된 종목 정리
            self.cleanup_old_positions()

        except Exception as e:
            logger.error(f"유니버스 구축 중 예외 발생: {e}", exc_info=True)

    def buy_market(self, ticker: str, amount_krw: float):
        """시장가 매수"""
        if amount_krw <= 0:
            logger.warning(f"[{ticker}] 매수 금액이 0 이하: {amount_krw}")
            return

        price = self.get_current_price(ticker, use_cache=False)
        if not validate_price(price):
            logger.warning(f"[{ticker}] 현재가 조회 실패, 매수 스킵")
            return

        fee_rate = 0.0005

        if DRY_RUN:
            use_krw = min(self.virtual_krw, amount_krw)
            if use_krw < 1000:
                logger.warning(f"[DRY_RUN][{ticker}] 사용 가능한 KRW가 부족: {use_krw:,.0f}")
                return

            buy_krw = use_krw * (1 - fee_rate)
            qty = buy_krw / price

            self.virtual_krw -= use_krw
            self.virtual_coin[ticker] = self.virtual_coin.get(ticker, 0.0) + qty
            self.invested_krw[ticker] = use_krw

            logger.info(
                f"[DRY_RUN][BUY][{ticker}] {qty:.8f}개 매수 @ {price:,.1f}원, "
                f"사용 KRW: {use_krw:,.0f}, 남은 KRW: {self.virtual_krw:,.0f}"
            )

            self.save_state()

        else:
            # 실전 매수
            if amount_krw < 5000:
                logger.warning(f"[REAL][{ticker}] 주문 금액이 5000원 미만, 매수 스킵")
                return

            try:
                def place_order():
                    return self.upbit.buy_market_order(ticker, amount_krw)

                order = retry_on_failure(place_order, logger=logger)
                if order:
                    logger.info(f"[REAL][BUY][{ticker}] 시장가 매수 주문 성공: {order}")
                    self.save_state()
                else:
                    logger.error(f"[REAL][BUY][{ticker}] 시장가 매수 주문 실패")

            except Exception as e:
                logger.error(f"[REAL][BUY][{ticker}] 매수 주문 예외: {e}")

    def sell_market(self, ticker: str):
        """시장가 매도"""
        price = self.get_current_price(ticker, use_cache=False)
        if not validate_price(price):
            logger.warning(f"[{ticker}] 현재가 조회 실패, 매도 스킵")
            return

        fee_rate = 0.0005

        if DRY_RUN:
            qty = self.virtual_coin.get(ticker, 0.0)
            if qty <= 0:
                logger.warning(f"[DRY_RUN][SELL][{ticker}] 보유 코인이 없습니다.")
                return

            sell_krw = qty * price * (1 - fee_rate)
            buy_krw = self.invested_krw.get(ticker, 0.0)

            pnl = sell_krw - buy_krw
            ret_pct = (pnl / buy_krw * 100.0) if buy_krw > 0 else 0.0

            self.virtual_krw += sell_krw
            self.virtual_coin[ticker] = 0.0
            self.invested_krw[ticker] = 0.0

            logger.info(
                f"[DRY_RUN][SELL][{ticker}] {qty:.8f}개 매도 @ {price:,.1f}원, "
                f"수령 KRW: {sell_krw:,.0f}, 수익률: {ret_pct:+.2f}%"
            )

            # 현재 총자산 계산
            total_equity = self.virtual_krw
            for tk, q in self.virtual_coin.items():
                if q > 0:
                    cp = self.get_current_price(tk)
                    if validate_price(cp):
                        total_equity += q * cp

            logger.info(f"[DRY_RUN] 현재 가상 총자산: {total_equity:,.0f}원 (KRW: {self.virtual_krw:,.0f})")

            self.save_state()

        else:
            # 실전 매도
            try:
                def get_balance():
                    return self.upbit.get_balances()

                balances = retry_on_failure(get_balance, logger=logger)
                if not balances:
                    logger.error(f"[REAL][SELL][{ticker}] 잔고 조회 실패")
                    return

                coin_symbol = ticker.split("-")[1]
                coin_balance = 0.0

                for b in balances:
                    if b.get('currency') == coin_symbol:
                        coin_balance = float(b.get('balance', 0))
                        break

                if coin_balance <= 0:
                    logger.warning(f"[REAL][SELL][{ticker}] 보유 코인이 없습니다.")
                    return

                def place_order():
                    return self.upbit.sell_market_order(ticker, coin_balance)

                order = retry_on_failure(place_order, logger=logger)
                if order:
                    logger.info(f"[REAL][SELL][{ticker}] 시장가 매도 주문 성공: {order}")
                    self.save_state()
                else:
                    logger.error(f"[REAL][SELL][{ticker}] 시장가 매도 주문 실패")

            except Exception as e:
                logger.error(f"[REAL][SELL][{ticker}] 매도 주문 예외: {e}")

    def process_symbol(self, ticker: str):
        """종목별 전략 처리"""
        try:
            def fetch_ohlcv():
                return pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=40)

            df = retry_on_failure(fetch_ohlcv, max_retries=2, logger=logger)

            if not validate_dataframe(df, min_length=20):
                logger.debug(f"[{ticker}] OHLCV 데이터 부족 또는 유효하지 않음")
                return

            # 이동평균 계산 (SMA5, SMA10, SMA20만 사용)
            df["sma5"] = df["close"].rolling(5, min_periods=5).mean()
            df["sma10"] = df["close"].rolling(10, min_periods=10).mean()
            df["sma20"] = df["close"].rolling(20, min_periods=20).mean()

            # NaN 체크
            if df[["sma5", "sma10", "sma20"]].isnull().any().any():
                logger.debug(f"[{ticker}] 이동평균 계산 결과에 NaN 존재")
                return

            # 완료봉과 현재봉
            prev = df.iloc[-2]  # 완료봉
            curr = df.iloc[-1]  # 진행중인 봉

            prev_time = prev.name
            curr_time = curr.name

            # ===== 1) 새 캔들 시작 시 이전 포지션 청산 =====
            stored_time = self.current_bar_time.get(ticker)

            if stored_time is None or curr_time != stored_time:
                # 새 캔들 시작!
                if self.in_position.get(ticker, False):
                    logger.info(f"[{ticker}] 새 캔들 시작 → 포지션 청산")
                    self.sell_market(ticker)
                    self.in_position[ticker] = False

                # 시간 갱신
                self.current_bar_time[ticker] = curr_time

            # ===== 2) 완료봉 기준으로 매수 조건 평가 (매 루프마다) =====
            if not self.in_position.get(ticker, False):
                # 완료봉의 이평선
                sma5_prev = prev["sma5"]
                sma10_prev = prev["sma10"]
                sma20_prev = prev["sma20"]

                # 정배열 체크
                is_ma_aligned = (
                    sma5_prev > sma10_prev and
                    sma10_prev > sma20_prev
                )

                if not is_ma_aligned:
                    # 정배열 아님
                    ma_status = []
                    if sma5_prev <= sma10_prev:
                        ma_status.append(f"SMA5({sma5_prev:.0f}) ≤ SMA10({sma10_prev:.0f})")
                    if sma10_prev <= sma20_prev:
                        ma_status.append(f"SMA10({sma10_prev:.0f}) ≤ SMA20({sma20_prev:.0f})")
                    logger.info(f"✗ [{ticker}] 정배열 조건 미충족: {' & '.join(ma_status)}")
                    return

                # 정배열 OK! 변동성 돌파 체크
                range_prev = prev["high"] - prev["low"]

                if range_prev <= 0:
                    logger.debug(f"[{ticker}] 변동성이 0 이하")
                    return

                # entry_price = 현재봉 시가 + 전일 변동폭 × K
                entry_price = curr["open"] + range_prev * K

                if not validate_price(entry_price):
                    logger.warning(f"[{ticker}] 유효하지 않은 entry_price: {entry_price}")
                    return

                # 현재봉의 고가와 현재가
                current_high = curr["high"]
                current_price = curr["close"]

                if not validate_price(current_high):
                    logger.warning(f"[{ticker}] 유효하지 않은 현재 고가: {current_high}")
                    return

                # 진행률 계산
                diff = current_high - entry_price
                progress = (current_high / entry_price - 1) * 100 if entry_price > 0 else 0

                # 실시간 모니터링
                logger.info(f"📊 [{ticker}] 현재가: {current_price:,.0f}원 | 현재고가: {current_high:,.0f}원 | 목표가: {entry_price:,.0f}원 | 진행률: {progress:+.2f}%")
                logger.info(f"   정배열 ✓ SMA5={sma5_prev:,.0f} > SMA10={sma10_prev:,.0f} > SMA20={sma20_prev:,.0f}")

                # 변동성 돌파 체크: 현재 고가가 이미 entry_price를 넘었는지
                if current_high >= entry_price:
                    # 돌파 발생! 즉시 매수
                    if DRY_RUN:
                        amount_krw = self.virtual_krw * ORDER_KRW_PORTION
                    else:
                        def get_balance():
                            balances = self.upbit.get_balances()
                            for b in balances:
                                if b.get('currency') == 'KRW':
                                    return float(b.get('balance', 0))
                            return 0.0

                        krw_balance = retry_on_failure(get_balance, logger=logger) or 0.0
                        amount_krw = krw_balance * ORDER_KRW_PORTION

                    logger.info("🚀" * 10)
                    logger.info(f"🚀 [{ticker}] 변동성 돌파 발생! 즉시 매수!")
                    logger.info(f"   현재 고가: {current_high:,.0f}원 | Entry: {entry_price:,.0f}원 | 돌파: +{diff:,.0f}원 ({progress:+.2f}%)")

                    self.buy_market(ticker, amount_krw)
                    self.in_position[ticker] = True

        except Exception as e:
            logger.error(f"[{ticker}] process_symbol 예외: {e}", exc_info=True)

    def run(self):
        """메인 루프"""
        logger.info("=" * 60)
        logger.info("변동성 돌파 전략 자동매매 시작")
        logger.info(f"INTERVAL={INTERVAL}, DRY_RUN={DRY_RUN}, K={K}")
        logger.info(f"초기 가상 KRW: {INITIAL_VIRTUAL_KRW:,.0f}원")
        logger.info("=" * 60)
        logger.info("네트워크 연결 확인 중...")

        # 최초 유니버스 생성 (실패 시 재시도)
        retry_count = 0
        while not self.universe and retry_count < 10:
            self.build_universe()
            if not self.universe:
                retry_count += 1
                logger.warning(f"유니버스 구축 실패. 30초 후 재시도... ({retry_count}/10)")
                time.sleep(30)

        if not self.universe:
            logger.critical("유니버스 구축에 계속 실패합니다. 네트워크 연결을 확인하세요.")
            logger.info("네트워크 연결이 복구되면 자동으로 유니버스가 갱신됩니다.")

        while True:
            try:
                now = datetime.now()

                # 유니버스 재계산
                if (self.last_universe_update is None or
                    (now - self.last_universe_update) > timedelta(minutes=UNIVERSE_REFRESH_MIN)):
                    self.build_universe()

                if not self.universe:
                    logger.info("유니버스에 종목이 없습니다. 1분 후 유니버스 재구축 시도...")
                    time.sleep(60)
                    self.build_universe()
                    continue

                # 각 종목 처리
                logger.info(f"\n{'='*60}")
                logger.info(f"⏰ {now.strftime('%Y-%m-%d %H:%M:%S')} - 유니버스 {len(self.universe)}개 종목 모니터링 중")
                logger.info(f"{'='*60}")

                for ticker in self.universe:
                    self.process_symbol(ticker)
                    # API 호출 간격 조절 (초당 10회 제한)
                    time.sleep(0.1)

                # 상태 요약 출력
                logger.info(f"\n{'='*60}")
                logger.info("📈 현재 상태 요약:")
                logger.info(f"   보유 포지션: {sum(1 for v in self.in_position.values() if v)}개")
                if DRY_RUN:
                    logger.info(f"   가상 KRW: {self.virtual_krw:,.0f}원")
                    if self.virtual_coin:
                        logger.info(f"   보유 종목: {list(k for k, v in self.virtual_coin.items() if v > 0)}")
                logger.info(f"   Entry Price 설정된 종목: {sum(1 for v in self.entry_price_map.values() if v is not None)}개")
                logger.info(f"{'='*60}\n")

                # 주기적으로 상태 저장
                self.save_state()

                logger.info(f"⏸️  {SLEEP_SEC}초 대기 중...\n")
                time.sleep(SLEEP_SEC)

            except KeyboardInterrupt:
                logger.info("사용자가 프로그램을 종료했습니다.")
                self.save_state()
                break

            except Exception as e:
                logger.error(f"메인 루프 예외: {e}", exc_info=True)
                time.sleep(SLEEP_SEC)


# ======== 프로그램 시작 ========
def main():
    """메인 함수"""
    try:
        bot = VolatilityBreakoutBot()
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n프로그램을 종료합니다.")
    except Exception as e:
        logger.critical(f"프로그램 치명적 오류: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
