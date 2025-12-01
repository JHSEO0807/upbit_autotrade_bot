import time
from datetime import datetime, timedelta

import numpy as np
import pyupbit

# ======== 설정값 ========
ACCESS_KEY = "YOUR_ACCESS_KEY"   # 실전 모드에서 본인 키로 교체
SECRET_KEY = "YOUR_SECRET_KEY"

DRY_RUN = True   # ★ 기본 True (모의매매). 실전 주문시 False 로 변경 ★

INTERVAL = "minute15"            # "minute5", "minute15", "minute60", "day" 등
K = 0.5                         # 변동성 계수 (파인스크립트 k)
SLEEP_SEC = 10                  # 메인 루프 대기 시간(초)
UNIVERSE_REFRESH_MIN = 1       # 유니버스(감시 종목 리스트) 재계산 주기 (분 단위)

INITIAL_VIRTUAL_KRW = 1_000_000  # DRY_RUN 가상 초기 자본
ORDER_KRW_PORTION = 0.3          # 매수 시 보유 KRW 의 몇 %를 한 종목에 투자할지 (20%)

VOLUME_THRESHOLD = 20_000_000_000  # 거래대금 200억 (일봉 'value' 기준)


# ======== 전역 상태 변수 ========
upbit = None

universe = []      # 감시 대상 티커 리스트
last_universe_update = None

virtual_krw = INITIAL_VIRTUAL_KRW        # DRY_RUN 용 가상 원화
virtual_coin = {}                        # {ticker: 수량}
in_position = {}                         # {ticker: bool}

current_bar_time = {}                    # {ticker: 현재 모니터링 중인 캔들의 시간}
entry_price_map = {}                     # {ticker: 이번 캔들에서 사용할 돌파 기준가}
invested_krw = {}                        # {ticker: 이 거래에 실제 투입한 KRW}


# ======== Upbit 초기화 ========
def init_upbit():
    global upbit
    if DRY_RUN:
        print("[INFO] DRY_RUN 모드입니다. 실제 주문은 전혀 실행되지 않습니다.")
        upbit = None
    else:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        print("[INFO] 실전 모드입니다. 실제 주문이 실행됩니다. 반드시 소액으로 테스트부터 하세요.")


# ======== 유니버스 구축 (전일대비 상승률 양수 + 거래대금 200억 이상) ========
def build_universe():
    """
    업비트 KRW 마켓 전체를 순회하면서:
      - 최근 2일 일봉 데이터 기준
      - 전일 대비 등락률 > 0
      - 거래대금(value) >= 200억
    인 종목들만 universe 에 포함
    """
    global universe, last_universe_update

    tickers = pyupbit.get_tickers(fiat="KRW")
    selected = []

    print("[INFO] 유니버스 계산 시작... (KRW 마켓 전체 스캔)")

    for ticker in tickers:
        try:
            df_day = pyupbit.get_ohlcv(ticker, interval="day", count=2)
            if df_day is None or len(df_day) < 2:
                continue

            prev_close = df_day["close"].iloc[-2]
            last_close = df_day["close"].iloc[-1]
            value = df_day["value"].iloc[-1]  # 일봉 거래대금

            if prev_close == 0:
                continue

            change_rate = (last_close / prev_close - 1) * 100.0

            if change_rate > 0 and value >= VOLUME_THRESHOLD:
                selected.append(ticker)
        except Exception:
            # 개별 종목 에러는 무시
            continue

    universe = selected
    last_universe_update = datetime.now()

    print(f"[유니버스 업데이트] {len(universe)}개 종목 감시중")
    print(f"감시 리스트: {universe}")


# ======== 현재가 조회 ========
def get_current_price(ticker):
    try:
        return pyupbit.get_current_price(ticker)
    except Exception:
        return None


# ======== DRY_RUN / REAL 공통: 시장가 매수 ========
def buy_market(ticker, amount_krw):
    global virtual_krw, virtual_coin, invested_krw, entry_price_map

    if amount_krw <= 0:
        return

    price = get_current_price(ticker)
    if price is None:
        print(f"[WARN][{ticker}] 현재가 조회 실패, 매수 스킵")
        return

    fee_rate = 0.0005  # 업비트 수수료 대략 0.05% 가정

    if DRY_RUN:
        # 가상 매수
        use_krw = min(virtual_krw, amount_krw)
        if use_krw < 1000:
            print(f"[DRY_RUN][{ticker}] 사용 가능한 KRW가 너무 적어서 매수 스킵")
            return

        buy_krw = use_krw * (1 - fee_rate)
        qty = buy_krw / price

        virtual_krw -= use_krw
        virtual_coin[ticker] = virtual_coin.get(ticker, 0.0) + qty
        invested_krw[ticker] = use_krw  # 수수료 포함 투입금
        print(f"[DRY_RUN][BUY][{ticker}] {qty:.6f}개 매수 @ {price:.1f}원, "
              f"사용 KRW: {use_krw:,.0f}, 남은 KRW: {virtual_krw:,.0f}")
    else:
        # 실전 매수
        if amount_krw < 5000:
            print(f"[REAL][{ticker}] 주문 금액이 5000원 미만, 매수 스킵")
            return
        try:
            order = upbit.buy_market_order(ticker, amount_krw)
            print(f"[REAL][BUY][{ticker}] 시장가 매수 주문 전송: {order}")
        except Exception as e:
            print(f"[ERROR][{ticker}] 매수 주문 실패: {e}")


# ======== DRY_RUN / REAL 공통: 시장가 매도 ========
def sell_market(ticker):
    global virtual_krw, virtual_coin, invested_krw

    price = get_current_price(ticker)
    if price is None:
        print(f"[WARN][{ticker}] 현재가 조회 실패, 매도 스킵")
        return

    fee_rate = 0.0005

    if DRY_RUN:
        qty = virtual_coin.get(ticker, 0.0)
        if qty <= 0:
            print(f"[DRY_RUN][SELL][{ticker}] 보유 코인이 없습니다.")
            return

        sell_krw = qty * price * (1 - fee_rate)
        buy_krw = invested_krw.get(ticker, 0.0)

        pnl = sell_krw - buy_krw
        ret_pct = (pnl / buy_krw * 100.0) if buy_krw > 0 else 0.0

        virtual_krw += sell_krw
        virtual_coin[ticker] = 0.0
        invested_krw[ticker] = 0.0

        print(f"[DRY_RUN][SELL][{ticker}] {qty:.6f}개 매도 @ {price:.1f}원, "
              f"수령 KRW: {sell_krw:,.0f}, 이번 트레이드 수익률: {ret_pct:.2f}%")

        # 현재 가상 총자산
        total_equity = virtual_krw
        for tk, q in virtual_coin.items():
            if q > 0:
                cp = get_current_price(tk) or 0
                total_equity += q * cp

        print(f"[DRY_RUN][{ticker}] 현재 가상 총자산: {total_equity:,.0f}원 "
              f"(KRW: {virtual_krw:,.0f})")

    else:
        # 실전 매도
        try:
            balances = upbit.get_balances()
            coin_symbol = ticker.split("-")[1]
            coin_balance = 0.0
            for b in balances:
                if b['currency'] == coin_symbol:
                    coin_balance = float(b['balance'])
                    break

            if coin_balance <= 0:
                print(f"[REAL][SELL][{ticker}] 보유 코인이 없습니다.")
                return

            order = upbit.sell_market_order(ticker, coin_balance)
            print(f"[REAL][SELL][{ticker}] 시장가 매도 주문 전송: {order}")
        except Exception as e:
            print(f"[ERROR][{ticker}] 매도 주문 실패: {e}")


# ======== 각 종목별 전략 처리 ========
def process_symbol(ticker):
    """
    종목 하나에 대해:
      - 직전 캔들로 entry_price 계산
      - 현재 진행 중인 캔들의 고가가 entry_price 돌파하면 매수
      - 새 캔들이 시작되면, 이전 캔들에서 진입한 포지션은 매도
    """
    global current_bar_time, entry_price_map, in_position

    try:
        df = pyupbit.get_ohlcv(ticker, interval=INTERVAL, count=60)
    except Exception:
        print(f"[WARN][{ticker}] OHLCV 조회 실패, 스킵")
        return

    if df is None or len(df) < 40:
        # 이동평균 계산에 필요한 최소 갯수 부족
        return

    # 이동평균 계산
    df["sma5"] = df["close"].rolling(5).mean()
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma40"] = df["close"].rolling(40).mean()

    # 마지막 캔들: 현재 진행중인 캔들 (실시간)
    # 마지막에서 두 번째 캔들: 완전히 끝난 직전 캔들
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_time = prev.name
    curr_time = curr.name

    # ===== 1) 새 캔들이 시작되었는지 체크 =====
    stored_time = current_bar_time.get(ticker)

    if stored_time is None or curr_time != stored_time:
        # (1) 이전 캔들에서 포지션이 있었다면 → 지금 시점(새 캔들 시작)에 매도
        if in_position.get(ticker, False):
            print(f"[INFO][{ticker}] 새 캔들 시작 감지 → 이전 캔들 포지션 청산")
            sell_market(ticker)
            in_position[ticker] = False

        # (2) 이번에 새로 모니터링할 캔들의 시간 갱신
        current_bar_time[ticker] = curr_time

        # (3) 이번 캔들에서 사용할 entry_price 계산 (직전 캔들 기준)
        sma5_prev = prev["sma5"]
        sma10_prev = prev["sma10"]
        sma20_prev = prev["sma20"]
        sma40_prev = prev["sma40"]

        if np.isnan(sma40_prev):
            entry_price_map[ticker] = None
            return

        is_ma_aligned = (
            sma5_prev > sma10_prev
            and sma10_prev > sma20_prev
            and sma20_prev > sma40_prev
        )

        if is_ma_aligned:
            range_prev = prev["high"] - prev["low"]
            entry_price = prev["close"] + range_prev * (K * K)
            entry_price_map[ticker] = entry_price
            print(f"[INFO][{ticker}] 새 캔들 시작, 정배열 ON, entry_price = {entry_price:.1f}")
        else:
            entry_price_map[ticker] = None
            print(f"[INFO][{ticker}] 새 캔들 시작, 정배열 아님 → 이번 캔들 매수 안 함")
            return

    # ===== 2) 현재 캔들에서 돌파 발생 여부 체크 =====
    # 여기까지 왔다는 건: current_bar_time[ticker] == curr_time 인 상태
    if not in_position.get(ticker, False):
        entry_price = entry_price_map.get(ticker)
        if entry_price is None:
            return

        current_high = curr["high"]   # 현재 진행중인 캔들의 고가
        if current_high >= entry_price:
            # 돌파 발생 → 시장가 매수
            if DRY_RUN:
                amount_krw = virtual_krw * ORDER_KRW_PORTION
            else:
                balances = upbit.get_balances()
                krw_balance = 0.0
                for b in balances:
                    if b['currency'] == 'KRW':
                        krw_balance = float(b['balance'])
                        break
                amount_krw = krw_balance * ORDER_KRW_PORTION

            print(f"[SIGNAL][{ticker}] 변동성 돌파 발생! high={current_high:.1f}, entry={entry_price:.1f}")
            buy_market(ticker, amount_krw)
            in_position[ticker] = True
    else:
        # 이미 포지션 보유 중이면, 이 캔들 끝날 때까지 그냥 보유
        pass


# ======== 메인 루프 ========
def main():
    global last_universe_update

    init_upbit()

    print(f"[START] 변동성 돌파 + 정배열 + 유니버스 필터 자동매매 시작")
    print(f"        INTERVAL={INTERVAL}, DRY_RUN={DRY_RUN}")
    print(f"        초기 가상 KRW: {INITIAL_VIRTUAL_KRW:,.0f}원")

    # 최초 유니버스 생성
    build_universe()

    while True:
        try:
            now = datetime.now()

            # 일정 시간마다 유니버스 재계산
            if (last_universe_update is None) or \
               ((now - last_universe_update) > timedelta(minutes=UNIVERSE_REFRESH_MIN)):
                build_universe()

            if not universe:
                print("[INFO] 유니버스에 종목이 없습니다. 잠시 대기.")
                time.sleep(SLEEP_SEC)
                continue

            # 유니버스 종목들 순회
            for ticker in universe:
                process_symbol(ticker)

            time.sleep(SLEEP_SEC)

        except Exception as e:
            print(f"[ERROR] 메인 루프 에러: {e}")
            time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    main()
