"""
변동성 돌파 전략 자동매매 봇 (정배열 필터 제거 버전)
- 매수: (이전캔들 고가 - 이전캔들 저가)*K + 현재캔들 시가 < 현재가격 → 매수
- 매도: 현재 캔들이 마감되고 새로운 캔들이 시작될 때 전량 매도
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

DRY_RUN = True  # 모의매매 기본값(True)
INTERVAL = "minute15"  # "minute5", "minute15", "minute60", "day" 등

K = 0.5
SLEEP_SEC = 10
UNIVERSE_REFRESH_MIN = 15

INITIAL_VIRTUAL_KRW = 1_000_000
ORDER_KRW_PORTION = 0.3  # 보유KRW의 30%

VOLUME_THRESHOLD = 20_000_000_000  # 200억


# ======== API 재시도 ========
MAX_RETRIES = 3
RETRY_DELAY = 1.0
BACKOFF_FACTOR = 2.0

STATE_FILE = Path(__file__).parent / "trading_state.json"
LOG_FILE = Path(__file__).parent / "trading.log"


# ======== 로깅 설정 ========
def setup_logging():
    logger = logging.getLogger("VolatilityBot")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()


# ======== 기본 유틸 ========
def retry_on_failure(func, max_retries=MAX_RETRIES, delay=RETRY_DELAY,
                     backoff=BACKOFF_FACTOR, logger=None):
    last_exception = None
    now_delay = delay
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(f"{func.__name__} 실패({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(now_delay)
                now_delay *= backoff
    if logger:
        logger.error(f"{func.__name__} 최종 실패: {last_exception}")
    return None


def validate_price(price):
    return price is not None and price > 0 and not np.isnan(price)


def validate_dataframe(df, min_length):
    return df is not None and len(df) >= min_length


# ======== 메인 클래스 ========
class VolatilityBreakoutBot:

    def __init__(self):
        self.upbit = None
        self.universe: List[str] = []
        self.last_universe_update: Optional[datetime] = None

        self.virtual_krw = INITIAL_VIRTUAL_KRW
        self.virtual_coin: Dict[str, float] = {}

        self.in_position: Dict[str, bool] = {}
        self.current_bar_time: Dict[str, datetime] = {}
        self.entry_price_map: Dict[str, Optional[float]] = {}
        self.invested_krw: Dict[str, float] = {}

        self.total_trades = 0
        self.win_trades = 0
        self.total_pnl_krw = 0

        self._price_cache: Dict[str, Tuple[float, datetime]] = {}
        self._price_cache_ttl = 5

        self.init_upbit()
        self.load_state()

    # -----------------------
    #    기본 설정
    # -----------------------
    def init_upbit(self):
        if DRY_RUN:
            logger.info("DRY_RUN 모드 (모의매매)")
            self.upbit = None
        else:
            self.upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
            logger.info("실전 매매 활성화됨")

    def load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)

            if DRY_RUN:
                self.virtual_krw = state.get("virtual_krw", INITIAL_VIRTUAL_KRW)
                self.virtual_coin = state.get("virtual_coin", {})

            self.in_position = state.get("in_position", {})
            self.invested_krw = state.get("invested_krw", {})
            self.entry_price_map = state.get("entry_price_map", {})

            self.total_trades = state.get("total_trades", 0)
            self.win_trades = state.get("win_trades", 0)
            self.total_pnl_krw = state.get("total_pnl_krw", 0.0)

            logger.info(f"상태 복구 완료 — 보유KRW {self.virtual_krw:,.0f}")
        except:
            logger.error("상태 복구 실패. 새로 시작합니다.")

    def save_state(self):
        try:
            state = {
                "virtual_krw": self.virtual_krw,
                "virtual_coin": self.virtual_coin,
                "in_position": self.in_position,
                "invested_krw": self.invested_krw,
                "entry_price_map": self.entry_price_map,
                "total_trades": self.total_trades,
                "win_trades": self.win_trades,
                "total_pnl_krw": self.total_pnl_krw,
                "timestamp": datetime.now().isoformat()
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except:
            logger.error("상태 저장 실패")

    # -----------------------
    #     유니버스 관리
    # -----------------------
    def get_current_price(self, ticker, use_cache=True):
        now = datetime.now()

        if use_cache and ticker in self._price_cache:
            price, ts = self._price_cache[ticker]
            if (now - ts).total_seconds() < self._price_cache_ttl:
                return price

        def func():
            return pyupbit.get_current_price(ticker)

        price = retry_on_failure(func, logger=logger)
        if validate_price(price):
            self._price_cache[ticker] = (price, now)
        return price

    def build_universe(self):
        logger.info("유니버스 계산 중...")

        def fetch_tickers():
            return pyupbit.get_tickers(fiat="KRW")

        tickers = retry_on_failure(fetch_tickers, logger=logger)
        if not tickers:
            logger.error("티커 목록 조회 실패")
            return

        # Upbit Ticker API (100개씩 호출)
        def fetch_ticker_data():
            res = []
            batch_size = 100
            for i in range(0, len(tickers), batch_size):
                url = "https://api.upbit.com/v1/ticker?markets=" + ",".join(
                    tickers[i:i + batch_size]
                )
                r = requests.get(url)
                r.raise_for_status()
                res.extend(r.json())
                time.sleep(0.1)
            return res

        data = retry_on_failure(fetch_ticker_data, logger=logger)
        if not data:
            logger.error("ticker 데이터 조회 실패")
            return

        sel = []
        for d in data:
            try:
                market = d["market"]
                change_rate = d["signed_change_rate"] * 100
                acc = d["acc_trade_price_24h"]

                if change_rate > 0 and acc >= VOLUME_THRESHOLD:
                    sel.append(market)
            except:
                continue

        self.universe = sel
        self.last_universe_update = datetime.now()

        # 콘솔 출력 (깔끔하게)
        logger.info("====================================================")
        logger.info(f"🎯 유니버스 업데이트 — {len(sel)}개 종목")
        for m in sel:
            logger.info(f"  - {m}")
        if not sel:
            logger.info("⚠ 조건 만족 종목 없음")
        logger.info("====================================================")

        self.cleanup_old_positions()

    def cleanup_old_positions(self):
        for t in list(self.in_position.keys()):
            if t not in self.universe:
                if DRY_RUN and self.virtual_coin.get(t, 0) > 0:
                    logger.warning(f"{t} — 유니버스 제외 → 강제 청산")
                    self.sell_market(t)

                self.in_position.pop(t, None)
                self.current_bar_time.pop(t, None)
                self.entry_price_map.pop(t, None)

    # -----------------------
    #      매수 / 매도
    # -----------------------
    def buy_market(self, ticker, amount_krw):
        price = self.get_current_price(ticker, use_cache=False)
        if not validate_price(price):
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fee = 0.0005

        if DRY_RUN:
            use_krw = min(self.virtual_krw, amount_krw)
            if use_krw < 1000:
                return

            qty = (use_krw * (1 - fee)) / price
            self.virtual_krw -= use_krw
            self.virtual_coin[ticker] = self.virtual_coin.get(ticker, 0) + qty
            self.invested_krw[ticker] = use_krw

            logger.info(
                f"[BUY] {now} | {ticker}\n"
                f"  가격 {price:,.0f}원 | 수량 {qty:.6f}개 | KRW {use_krw:,.0f}"
            )

            self.save_state()
            return

    def sell_market(self, ticker):
        price = self.get_current_price(ticker, use_cache=False)
        if not validate_price(price):
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        qty = self.virtual_coin.get(ticker, 0)
        if qty <= 0:
            return

        fee = 0.0005
        sell_krw = qty * price * (1 - fee)
        buy_krw = self.invested_krw.get(ticker, 0)

        pnl = sell_krw - buy_krw
        ret_pct = pnl / buy_krw * 100 if buy_krw > 0 else 0

        self.virtual_krw += sell_krw
        self.virtual_coin[ticker] = 0
        self.invested_krw[ticker] = 0
        self.entry_price_map[ticker] = None

        self.total_trades += 1
        if pnl > 0:
            self.win_trades += 1
        self.total_pnl_krw += pnl

        win_rate = self.win_trades / self.total_trades * 100 if self.total_trades > 0 else 0
        total_equity = self.virtual_krw
        total_return_pct = (total_equity / INITIAL_VIRTUAL_KRW - 1) * 100

        logger.info(
            f"[SELL] {now} | {ticker}\n"
            f"  가격 {price:,.0f}원 | 수익률 {ret_pct:+.2f}% | PnL {pnl:,.0f}\n"
            f"  총 자산 {total_equity:,.0f}원 | 누적수익률 {total_return_pct:+.2f}% | 승률 {win_rate:.2f}%"
        )

        self.save_state()

    # -----------------------
    #       전략 로직
    # -----------------------
    def process_symbol(self, ticker):
        try:
            df = retry_on_failure(
                lambda: pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=40),
                logger=logger
            )
            if not validate_dataframe(df, 20):
                return

            prev = df.iloc[-2]
            curr = df.iloc[-1]
            curr_time = curr.name

            # 1) 새 캔들 시작 → 매도
            stored_time = self.current_bar_time.get(ticker)
            if stored_time is None or curr_time != stored_time:
                if self.in_position.get(ticker, False):
                    self.sell_market(ticker)
                    self.in_position[ticker] = False
                self.current_bar_time[ticker] = curr_time

            # 2) 매수 조건 (변동성 돌파 ONLY)
            if not self.in_position.get(ticker, False):

                # 변동폭
                range_prev = prev["high"] - prev["low"]
                if range_prev <= 0:
                    return

                entry_price = curr["open"] + range_prev * K
                current_price = curr["close"]

                if not validate_price(entry_price) or not validate_price(current_price):
                    return

                logger.debug(f"[{ticker}] 현재가={current_price:.0f}, 기준가={entry_price:.0f}")

                # ---- 변동성 돌파 ----
                if current_price >= entry_price:

                    if DRY_RUN:
                        amount_krw = self.virtual_krw * ORDER_KRW_PORTION
                    else:
                        balances = retry_on_failure(self.upbit.get_balances, logger=logger)
                        krw = 0
                        for b in balances:
                            if b["currency"] == "KRW":
                                krw = float(b["balance"])
                        amount_krw = krw * ORDER_KRW_PORTION

                    self.buy_market(ticker, amount_krw)
                    self.in_position[ticker] = True
                    self.entry_price_map[ticker] = float(entry_price)

        except Exception as e:
            logger.error(f"[{ticker}] process 오류: {e}", exc_info=True)

    # -----------------------
    #        메인 루프
    # -----------------------
    def run(self):
        logger.info("변동성 돌파 자동매매 시작")

        # 초기 유니버스 구축
        self.build_universe()

        while True:
            try:
                now = datetime.now()

                # 유니버스 주기 갱신
                if (self.last_universe_update is None or
                        (now - self.last_universe_update).seconds >= UNIVERSE_REFRESH_MIN * 60):
                    self.build_universe()

                if not self.universe:
                    time.sleep(30)
                    continue

                # 종목별 매매 처리
                for t in self.universe:
                    self.process_symbol(t)
                    time.sleep(0.1)

                self.save_state()
                time.sleep(SLEEP_SEC)

            except KeyboardInterrupt:
                logger.info("사용자 종료 요청 → 저장 후 종료")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"메인루프 오류: {e}", exc_info=True)
                time.sleep(SLEEP_SEC)


# ======== 시작점 ========
def main():
    try:
        bot = VolatilityBreakoutBot()
        bot.run()
    except Exception as e:
        logger.critical(f"치명적 오류: {e}", exc_info=True)


if __name__ == "__main__":
    main()

