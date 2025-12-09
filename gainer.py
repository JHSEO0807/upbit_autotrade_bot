import time
import uuid
import json
import hashlib
import requests
import pandas as pd
from datetime import datetime
import jwt
from ta.momentum import RSIIndicator

# =========================
# 업비트 설정/인증
# =========================
UPBIT_API = "https://api.upbit.com/v1"

# 🔑 본인 API 키 입력 (조회+거래 권한 필요, 출금 불필요)
ACCESS_KEY = ""
SECRET_KEY = ""

# =========================
# 전략/루프 설정
# =========================
SLEEP_SEC      = 180          # 3분 간격 모니터링
TOPN           = 20           # 전일대비 상승률 TOP N
INVEST_RATIO   = 0.20         # 보유 KRW의 20%로 매수
MIN_ORDER_KRW  = 6000         # 최소 주문 금액 가드
TIMEOUT        = 12           # HTTP 타임아웃(초)
MAX_HOLDINGS   = 3            # 동시 보유 최대 개수

# RSI 계산용
CANDLE_UNIT    = 3            # 3분봉
CANDLE_COUNT   = 200          # 지표 계산용 캔들 수
RSI_PERIOD     = 14

# =========================
# 상태 관리
# =========================
prev_top_list: list[str] = []          # 직전 루프 TOP20 (순서 유지)
prev_top_set: set[str] = set()
prev_ranks: dict[str, int] = {}        # 직전 루프의 랭킹 맵 {market: rank}

# 신규 진입 후보: 바로 다음 루프에서만 랭크 개선 시 매수
# 예: candidates["KRW-BTC"] = {"rank": 7, "round": 12}
candidates: dict[str, dict] = {}

# 보유/주문 상태
positions: dict[str, dict] = {}        # {market: {"buy_ts": float, "buy_price": float}}
buy_blocklist: set[str] = set()        # 보유/미청산 동안 재매수 금지

loop_round = 0                          # 루프 카운터

# =========================
# 공통: JWT 헤더 생성
# =========================
def _jwt_headers(query: dict | None):
    payload = {"access_key": ACCESS_KEY, "nonce": str(uuid.uuid4())}
    if query:
        q = "&".join([f"{k}={v}" for k, v in query.items()])
        m = hashlib.sha512(); m.update(q.encode("utf-8"))
        payload["query_hash"] = m.hexdigest()
        payload["query_hash_alg"] = "SHA512"
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}

# =========================
# 퍼블릭 API
# =========================
def get_krw_markets():
    r = requests.get(f"{UPBIT_API}/market/all", params={"isDetails": "false"}, timeout=TIMEOUT)
    r.raise_for_status()
    return [d["market"] for d in r.json() if d["market"].startswith("KRW-")]

def get_top_change_markets(limit=TOPN) -> pd.DataFrame:
    markets = get_krw_markets()
    r = requests.get(f"{UPBIT_API}/ticker", params={"markets": ",".join(markets)}, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    # 상승률 내림차순 정렬
    df = df.sort_values("signed_change_rate", ascending=False).reset_index(drop=True).head(limit)
    # 1~N 랭킹 부여
    df.insert(0, "rank", df.reset_index().index + 1)
    return df[["rank", "market", "trade_price", "signed_change_rate"]]

def get_candles(market, unit=CANDLE_UNIT, count=CANDLE_COUNT):
    r = requests.get(f"{UPBIT_API}/candles/minutes/{unit}",
                     params={"market": market, "count": count},
                     timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
    out = pd.DataFrame({
        "time":   pd.to_datetime(df["candle_date_time_kst"]),
        "open":   df["opening_price"].astype(float),
        "high":   df["high_price"].astype(float),
        "low":    df["low_price"].astype(float),
        "close":  df["trade_price"].astype(float),
        "volume": df["candle_acc_trade_volume"].astype(float),
    })
    return out

def get_last_two_rsi(market) -> tuple[float | None, float | None]:
    """해당 마켓의 3분봉 RSI 최근 2개 값 반환 (rsi_prev, rsi_now). 부족하면 (None, None)"""
    ohlcv = get_candles(market)
    if ohlcv is None or len(ohlcv) < RSI_PERIOD + 2:
        return (None, None)
    rsi_series = RSIIndicator(close=ohlcv["close"], window=RSI_PERIOD).rsi()
    rsi_now = float(rsi_series.iloc[-1])
    rsi_prev = float(rsi_series.iloc[-2])
    return (rsi_prev, rsi_now)

# =========================
# 프라이빗 API (계좌/주문)
# =========================
def get_accounts():
    r = requests.get(f"{UPBIT_API}/accounts", headers=_jwt_headers(None), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_krw_balance():
    for a in get_accounts():
        if a["currency"] == "KRW":
            return float(a["balance"])
    return 0.0

def get_coin_balance(market: str):
    symbol = market.split("-")[1]
    for a in get_accounts():
        if a["currency"] == symbol:
            return float(a["balance"])
    return 0.0

def place_market_buy_krw(market: str, krw_amount: float):
    if krw_amount < MIN_ORDER_KRW:
        print(f"⏸ 최소주문 미만: {krw_amount:.0f} KRW")
        return None
    body = {"market": market, "side": "bid", "ord_type": "price", "price": str(int(krw_amount))}
    headers = _jwt_headers(body) | {"Content-Type": "application/json"}
    r = requests.post(f"{UPBIT_API}/orders", headers=headers, data=json.dumps(body), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def place_market_sell_all(market: str, volume: float):
    if volume <= 0:
        print("⏸ 매도 스킵: 수량 0")
        return None
    body = {"market": market, "side": "ask", "ord_type": "market", "volume": f"{volume:.8f}"}
    headers = _jwt_headers(body) | {"Content-Type": "application/json"}
    r = requests.post(f"{UPBIT_API}/orders", headers=headers, data=json.dumps(body), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# =========================
# 주문 훅
# =========================
def buy_with_ratio(market: str):
    if len(positions) >= MAX_HOLDINGS:
        print(f"⛔ 매수스킵: 보유 {len(positions)}/{MAX_HOLDINGS} (한도)")
        return None
    if market in positions or market in buy_blocklist:
        return None

    krw = get_krw_balance()
    budget = krw * INVEST_RATIO
    if budget < MIN_ORDER_KRW:
        print(f"⏸ 매수스킵: KRW*{INVEST_RATIO:.0%}={budget:.0f}원 < 최소주문")
        return None

    try:
        res = place_market_buy_krw(market, budget)
        if res:
            print(f"[BUY] {market} | KRW {budget:.0f} 시장가 매수 접수 | uuid={res.get('uuid')}")
        return res
    except requests.HTTPError as he:
        print(f"❌ 매수 실패({market}) HTTP {he.response.status_code}: {he.response.text}")
        return None
    except Exception as e:
        print(f"❌ 매수 실패({market}): {e}")
        return None

def sell_all(market: str):
    vol = get_coin_balance(market)
    if vol <= 0:
        print(f"⏸ 매도스킵: {market} 보유수량 없음")
        return None
    try:
        res = place_market_sell_all(market, vol)
        if res:
            print(f"[SELL] {market} | 전량({vol:.8f}) 시장가 매도 접수 | uuid={res.get('uuid')}")
        return res
    except requests.HTTPError as he:
        print(f"❌ 매도 실패({market}) HTTP {he.response.status_code}: {he.response.text}")
        return None
    except Exception as e:
        print(f"❌ 매도 실패({market}): {e}")
        return None

# =========================
# 메인 루프
# =========================
def run_loop():
    global loop_round, prev_top_list, prev_top_set, prev_ranks, candidates

    print(f"🚀 전략 시작: 3분 간격 | TOP{TOPN} 추적 → 신규 진입 후 다음 라운드 순위상승 시 매수 → RSI 하강 반전(현재 < 이전) 시 매도")
    while True:
        loop_round += 1
        loop_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            # --- 1) 현재 TOP20 취득 ---
            top_df = get_top_change_markets(TOPN)
            curr_top_list = top_df["market"].tolist()
            curr_top_set  = set(curr_top_list)
            curr_ranks    = dict(zip(top_df["market"], top_df["rank"]))

            print(f"\n[{loop_ts}] 📈 전일대비 상승률 TOP {TOPN}")
            for _, row in top_df.iterrows():
                print(f"{row['rank']:2d}. {row['market']:10s} | 가격 {row['trade_price']:.0f} | 변화율 {row['signed_change_rate']:.4f}")

            # --- 2) 신규 진입 탐지 ---
            if prev_top_set:
                new_entries = curr_top_set - prev_top_set
            else:
                new_entries = set()

            if new_entries:
                print("🔎 신규 진입:", ", ".join(sorted(new_entries)))
            else:
                print("🔎 신규 진입: 없음")

            # 현재 라운드의 신규 진입을 후보로 등록 (다음 라운드에서만 평가)
            for m in new_entries:
                candidates[m] = {"rank": curr_ranks[m], "round": loop_round}

            # --- 3) 후보 평가: 바로 '다음 라운드'에서만 순위 개선 시 매수 ---
            to_delete = []
            for m, info in list(candidates.items()):
                born_round = info["round"]
                base_rank  = info["rank"]

                # 다음 라운드가 되었을 때만 체크
                if loop_round == born_round + 1:
                    if m in curr_ranks:
                        new_rank = curr_ranks[m]
                        if new_rank < base_rank:
                            print(f"✅ 랭크 개선 매수 신호: {m} | {base_rank} → {new_rank}")
                            if buy_with_ratio(m):
                                positions[m] = {
                                    "buy_ts": time.time(),
                                    "buy_price": None  # 필요시 체결 조회로 보완 가능
                                }
                                buy_blocklist.add(m)
                        else:
                            print(f"⏸ 매수 패스(개선 없음): {m} | {base_rank} → {new_rank}")
                    else:
                        print(f"⏸ 매수 패스(탈락): {m} | 현재 TOP{TOPN} 밖")
                    to_delete.append(m)

                # 만료(1라운드 지나면 후보 삭제)
                elif loop_round > born_round + 1:
                    to_delete.append(m)

            for m in to_delete:
                candidates.pop(m, None)

            # --- 4) 매도: 보유 종목 RSI 하강 반전(현재 < 이전) 시 매도 ---
            for m in list(positions.keys()):
                try:
                    rsi_prev, rsi_now = get_last_two_rsi(m)
                    if rsi_prev is None or rsi_now is None:
                        print(f"⏸ 매도체크 스킵(RSI 데이터 부족): {m}")
                        continue

                    if rsi_now < rsi_prev:
                        vol = get_coin_balance(m)
                        print(f"🔻 RSI 하강 반전 감지: {m} | {rsi_prev:.2f} → {rsi_now:.2f} (매도)")
                        if sell_all(m):
                            positions.pop(m, None)
                            buy_blocklist.discard(m)
                    else:
                        print(f"📊 보유유지: {m} | RSI {rsi_prev:.2f} → {rsi_now:.2f} (상승/유지)")
                except Exception as se:
                    print(f"⚠️ 매도체크 실패({m}): {se}")

            # --- 5) 이번 라운드 결과를 다음 라운드 비교용으로 저장 ---
            prev_top_list = curr_top_list
            prev_top_set  = curr_top_set
            prev_ranks    = curr_ranks

        except requests.HTTPError as he:
            print(f"❌ HTTP 오류: {he.response.status_code} {he.response.text}")
        except Exception as e:
            print(f"❌ 루프 오류: {e}")

        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    run_loop()
