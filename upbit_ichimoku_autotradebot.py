import time
import math
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import pyupbit

# ============== 認証キー（実売買時のみ設定） =================
# 実売買を行う場合は下記を入力し、DRY_RUN=False にしてください。
ACCESS_KEY = ""   # 例: "UPBIT-ACCESS-KEY"
SECRET_KEY = ""   # 例: "UPBIT-SECRET-KEY"
# ============================================================


# ================= ユーザー設定（戦略・運用） =================
INTERVAL         = "minute5"     # 5分足
BARS             = 300           # 取得ローソク足本数（インジ計算の安定化に十分な長さ）

UNIVERSE_FIAT    = "KRW"         # KRW マーケットのみ
MIN_24H_TURNOVER = 50_000 * 1_000_000   # 24h売買代金の下限（5,000億 KRW）
ONLY_POS_CHANGE  = True          # 前日比 > 0 の銘柄のみ買い候補
EXCLUDE          = {"KRW-BTC","KRW-USDT","KRW-ETH","KRW-XRP"}  # 買いから除外

# 一目均衡表（転換線・基準線）
TENKAN_LEN = 9
KIJUN_LEN  = 26

# Squeeze Momentum（LazyBear 互換）
BB_LENGTH  = 20
KC_LENGTH  = 20
KC_MULT    = 1.5   # ★オリジナルと同様：BB の σ に KC の係数を使用
USE_TR     = True  # True Range を使用（KC 計算）

# 移動平均フィルタ
SMA48_LEN      = 48
SMA48_SLOPE_N  = 5   # 48SMA の傾き判定に使う直近本数（線形回帰の傾き > 0）

# RSI（売り判定）
RSI_LEN    = 14
RSI_OVER   = 80.0

# DMI/ADX・MACD（TK≈KJ 条件で使用）
ADX_LEN      = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9

# TK≈KJ（“ほぼ一致”）の許容誤差（相対誤差）
TK_TOUCH_RTOL = 1e-4

# リスク管理・発注
STOP_LOSS_PCT = 0.01       # 損切り -1%
INVEST_RATIO  = 0.20       # 保有 KRW のうち何割を投入するか
MIN_KRW_ORDER = 5000       # Upbitの最小成行金額
FEE_RATE      = 0.0005     # 手数料（片道）
SLIPPAGE      = 0.0005     # すべり（片道）
DRY_RUN       = True       # ★デフォルトは模擬取引（True）

# ループ設定
LOOP_SLEEP_SEC    = 15
WATCH_REFRESH_MIN = 5

# 各インジの安定化に必要な最小バー数（不足時はスキップ）
NEED_BARS_MIN = max(
    60,
    SMA48_LEN + SMA48_SLOPE_N + 2,
    KIJUN_LEN + 2,
    KC_LENGTH + 2,
    BB_LENGTH + 2,
    ADX_LEN + 2,
    MACD_SLOW + MACD_SIGNAL + 2
)


# ===================== 共通ユーティリティ =====================
def now_kst() -> datetime:
    """KST 現在時刻（タイムゾーン付き）"""
    return datetime.now(timezone(timedelta(hours=9)))


def safe_print(*a, **k):
    """Windows 控えめな環境でも落ちないようプリント"""
    try:
        print(*a, **k)
    except Exception:
        print(str(a), **k)


# ===================== テクニカル関数群 ======================
def sma(s: pd.Series, n: int) -> pd.Series:
    """単純移動平均（欠損期間は NA）"""
    return s.rolling(n, min_periods=n).mean()


def stdev_pine(s: pd.Series, n: int) -> pd.Series:
    """Pine の ta.stdev 互換（母分散ベース）"""
    m1, m2 = sma(s, n), sma(s*s, n)
    return np.sqrt((m2 - m1*m1).clip(lower=0))


def true_range_pine(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Pine の ta.tr 互換"""
    pc = c.shift(1)
    return pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)


def linreg_last_pine(y: pd.Series, n: int) -> pd.Series:
    """
    Pine の ta.linreg(y, n, 0) 互換実装（ウィンドウの最終点の回帰値）
    """
    def f(a: np.ndarray) -> float:
        x = np.arange(len(a), dtype=float)
        yv = a.astype(float)
        xm, ym = x.mean(), yv.mean()
        vx = ((x - xm) ** 2).sum()
        if vx == 0:
            return yv[-1]
        slope = ((x - xm) * (yv - ym)).sum() / vx
        inter = ym - slope * xm
        return inter + slope * (len(a) - 1)
    return y.rolling(n, min_periods=n).apply(f, raw=True)


def compute_smi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Squeeze Momentum（LazyBear 変形の仕様を厳密再現）
      - BB の σ に KC_MULT を用いる（原作準拠）
      - isLime = val>0 かつ val>val[1]
    """
    h, l, c = df["high"], df["low"], df["close"]
    # BB（σ は KC_MULT ベース）
    dev   = KC_MULT * stdev_pine(c, BB_LENGTH)
    basis = sma(c, BB_LENGTH)
    upperBB, lowerBB = basis + dev, basis - dev

    # KC
    ma      = sma(c, KC_LENGTH)
    rng     = true_range_pine(h, l, c) if USE_TR else (h - l)
    rangema = sma(rng, KC_LENGTH)
    upperKC, lowerKC = ma + rangema * KC_MULT, ma - rangema * KC_MULT

    # Squeeze 判定（参考：可視化用途、今回は isLime のみ使用）
    sqzOn  = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)

    # モメンタム値
    hh = h.rolling(KC_LENGTH).max()
    ll = l.rolling(KC_LENGTH).min()
    midHL = (hh + ll) / 2.0
    center = (midHL + sma(c, KC_LENGTH)) / 2.0
    val = linreg_last_pine(c - center, KC_LENGTH)

    isLime = (val > 0) & (val > val.shift(1).fillna(0.0))
    return pd.DataFrame({"val": val, "isLime": isLime, "sqzOn": sqzOn, "sqzOff": sqzOff})


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """一目均衡表（転換線・基準線・GC 判定のみ）"""
    h, l = df["high"], df["low"]
    tenkan = (h.rolling(TENKAN_LEN).max() + l.rolling(TENKAN_LEN).min()) / 2
    kijun  = (h.rolling(KIJUN_LEN).max() + l.rolling(KIJUN_LEN).min()) / 2
    tk_golden = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun, "tk_golden": tk_golden})


def compute_adx_di(df: pd.DataFrame, n: int = ADX_LEN) -> pd.DataFrame:
    """Wilder の DMI/ADX（RMA 平滑）"""
    h, l, c = df["high"], df["low"], df["close"]
    up_move   = h.diff()
    down_move = -l.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range_pine(h, l, c)

    tr_rma      = pd.Series(tr).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_dm_rma = pd.Series(plus_dm, index=h.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    minus_dm_rma= pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    plus_di  = 100 * (plus_dm_rma  / tr_rma.replace(0, np.nan))
    minus_di = 100 * (minus_dm_rma / tr_rma.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx})


def compute_macd(close: pd.Series,
                 fast: int = MACD_FAST,
                 slow: int = MACD_SLOW,
                 signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD（EMA ベース、標準 12-26-9）"""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - macd_sig
    return pd.DataFrame({"MACD": macd, "SIGNAL": macd_sig, "HIST": hist})


def compute_rsi_rma(c: pd.Series, n: int) -> pd.Series:
    """Pine の ta.rsi と同等の RMA 実装"""
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    rma_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rma_dn = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = rma_up / rma_dn.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


# ===================== データ取得（KST） =====================
def get_ohlcv_kst(ticker: str, interval: str, count: int) -> pd.DataFrame:
    """
    pyupbit.get_ohlcv は KST（naive index）で返却。
    欠損・空データは空 DataFrame を返す。
    """
    df = pyupbit.get_ohlcv(ticker=ticker, interval=interval, count=count)
    return pd.DataFrame() if df is None or df.empty else df


# ===================== Upbit 公開 API ヘルパ =================
def fetch_ticker_infos(markets):
    """複数マーケットの 24h 売買代金や前日比を一括取得"""
    if not markets:
        return []
    out, CHUNK = [], 100
    for i in range(0, len(markets), CHUNK):
        sub = markets[i:i+CHUNK]
        r = requests.get("https://api.upbit.com/v1/ticker",
                         params={"markets": ",".join(sub)}, timeout=5)
        r.raise_for_status()
        out.extend(r.json())
        time.sleep(0.05)
    return out


def get_caution_markets():
    """投資注意（CAUTION）銘柄を除外"""
    r = requests.get("https://api.upbit.com/v1/market/all",
                     params={"isDetails": "true"}, timeout=5)
    r.raise_for_status()
    data = r.json()
    return {d["market"] for d in data if d.get("market_warning") == "CAUTION"}


def get_universe():
    """
    買い候補ユニバース抽出：
      - KRW マーケット
      - 前日比 > 0（任意）
      - 24h 売買代金 下限以上
      - 投資注意＆除外銘柄は除外
    """
    mkts = [m for m in pyupbit.get_tickers(fiat=UNIVERSE_FIAT)
            if m.startswith("KRW-") and m not in EXCLUDE]
    if not mkts:
        return []
    caution = get_caution_markets()
    infos = fetch_ticker_infos(mkts)

    out = []
    for i in infos:
        try:
            m = i["market"]
            if m in caution or m in EXCLUDE:
                continue
            if ONLY_POS_CHANGE and float(i["signed_change_rate"]) <= 0:
                continue
            if float(i["acc_trade_price_24h"]) < MIN_24H_TURNOVER:
                continue
            out.append(m)
        except Exception:
            continue
    return sorted(out)


# ===================== 認証・発注ラッパ =====================
def upbit_client():
    """
    DRY_RUN=False かつ APIキーが設定されていれば Upbit クライアントを返す。
    そうでなければ DRY_RUN=True に強制して None を返す（安全第一）。
    """
    global DRY_RUN
    if ACCESS_KEY.strip() and SECRET_KEY.strip() and not DRY_RUN:
        return pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    if not DRY_RUN and (not ACCESS_KEY or not SECRET_KEY):
        safe_print("[WARN] APIキー未設定のため DRY_RUN=True に切替えます。")
    DRY_RUN = True
    return None


def get_balance_coin(up: pyupbit.Upbit, ticker: str) -> float:
    """銘柄の残高（数量）。DRY_RUN の場合は 0 を返す想定で呼び出し側が対処。"""
    sym = ticker.split("-")[1]
    bal = up.get_balance(sym) if up else 0
    return 0.0 if bal is None else float(bal)


def get_balance_krw(up: pyupbit.Upbit) -> float:
    """KRW 残高"""
    bal = up.get_balance("KRW") if up else 0
    return 0.0 if bal is None else float(bal)


def buy_market(up: pyupbit.Upbit, ticker: str, krw_amt: float):
    """成行買い（DRY_RUN なら結果だけ返す）"""
    if DRY_RUN:
        return {"mock": True, "side": "buy", "ticker": ticker, "krw": krw_amt}
    return up.buy_market_order(ticker, krw_amt)


def sell_market(up: pyupbit.Upbit, ticker: str, volume: float):
    """成行売り（DRY_RUN なら結果だけ返す）"""
    if DRY_RUN:
        return {"mock": True, "side": "sell", "ticker": ticker, "qty": volume}
    return up.sell_market_order(ticker, volume)


# ===================== ポジション管理 ======================
class Position:
    """保有ポジション（最小限の情報のみ保持）"""
    def __init__(self, ticker, entry_raw, entry_exec, qty):
        self.ticker = ticker
        self.entry_raw = entry_raw    # エントリー時の始値（損切りライン計算用）
        self.entry_exec = entry_exec  # 手数料・すべりを加味した実効取得単価
        self.qty = qty
        self.rsi_over = False         # 一度でも RSI>80 を踏んだかのフラグ


# ===================== シグナル評価 ======================
def evaluate_signals(df: pd.DataFrame):
    """
    確定足（直近から1本前）で全インジを評価し、買い/売り判定用の値を返す。
    戻り値：
      { buy, reason, open, close, RSI, SMA48, SMA48_UP(slope>0) }
    """
    smi   = compute_smi(df)
    ichi  = compute_ichimoku(df)
    rsi   = compute_rsi_rma(df["close"], RSI_LEN)
    sma48 = sma(df["close"], SMA48_LEN)
    adx   = compute_adx_di(df, ADX_LEN)
    macd  = compute_macd(df["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    sig = pd.concat([
        df[["open","close"]],
        smi[["val","isLime"]],
        ichi[["tenkan","kijun","tk_golden"]],
        rsi.rename("RSI"),
        sma48.rename("SMA48"),
        adx[[ "+DI","-DI","ADX"]],
        macd[["MACD","SIGNAL"]],
    ], axis=1)

    # インジ安定用の最小本数を満たさない場合は無視
    if len(sig) < max(10, SMA48_LEN + SMA48_SLOPE_N + 2):
        return None

    # 判定は「1つ前の確定足」で行う
    row = sig.iloc[-2]

    # 48SMA の傾き（直近 N 本の線形回帰）> 0 なら上昇とみなす
    sma48_series = sig["SMA48"].dropna()
    if len(sma48_series) >= SMA48_SLOPE_N:
        y = sma48_series.values[-SMA48_SLOPE_N:]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        sma48_up = slope > 0
    else:
        sma48_up = False

    # --- 共通フィルタ ---
    smi_lime    = bool(row.isLime)
    above_sma48 = (row.close > row.SMA48) if pd.notna(row.SMA48) else False
    common_ok   = smi_lime and above_sma48 and sma48_up

    # --- トリガー A：転換線が基準線を GC ---
    cond_A = bool(row.tk_golden) and common_ok

    # --- トリガー B：転換線 ≒ 基準線（ほぼ一致）+ DI/MACD 条件 ---
    tk_touch = (
        np.isclose(row.tenkan, row.kijun, rtol=TK_TOUCH_RTOL, atol=0.0)
        if pd.notna(row.tenkan) and pd.notna(row.kijun) else False
    )
    di_ok   = (
        (row["+DI"] > row["ADX"]) and (row["+DI"] > row["-DI"])
        if pd.notna(row["+DI"]) and pd.notna(row["-DI"]) and pd.notna(row["ADX"])
        else False
    )
    macd_ok = (
        (row["MACD"] > row["SIGNAL"])
        if pd.notna(row["MACD"]) and pd.notna(row["SIGNAL"])
        else False
    )
    cond_B = tk_touch and di_ok and macd_ok and common_ok

    buy = cond_A or cond_B
    reason = ("A-GOLDEN" if cond_A and not cond_B else
              "B-TKTOUCH" if cond_B and not cond_A else
              "A|B") if buy else ""

    return {
        "buy": buy,
        "reason": reason,
        "open": float(row.open),
        "close": float(row.close),
        "RSI": float(row.RSI),
        "SMA48": float(row.SMA48) if pd.notna(row.SMA48) else None,
        "SMA48_UP": sma48_up
    }


# ===================== メインループ ======================
def run():
    up = upbit_client()  # DRY_RUN=False & キー設定 なら Upbit クライアント、そうでなければ None
    positions = {}       # {ticker: Position}
    last_watch = None
    watch = []

    safe_print(f"[START] {now_kst():%Y-%m-%d %H:%M:%S}  DRY_RUN={DRY_RUN}  INTERVAL={INTERVAL}")

    while True:
        try:
            # ---- 買い候補ユニバースの定期更新（買い専用）----
            if (last_watch is None) or (now_kst() - last_watch >= timedelta(minutes=WATCH_REFRESH_MIN)):
                watch = get_universe()
                last_watch = now_kst()
                safe_print(f"[WATCH] 買い候補={len(watch)}（前日比>0 / 24h≥{MIN_24H_TURNOVER:,} KRW / 除外={len(EXCLUDE)}）")

            # ---- 新規エントリー判定（買い）----
            for t in list(watch):
                df = get_ohlcv_kst(t, INTERVAL, BARS)
                if df.empty or len(df) < NEED_BARS_MIN:
                    continue

                sig = evaluate_signals(df)
                if sig and sig["buy"] and t not in positions:
                    # 投入資金の計算
                    krw = get_balance_krw(up) if up else 1_000_000  # DRY_RUN のときは仮想残高
                    invest = math.floor(krw * INVEST_RATIO)
                    if invest < MIN_KRW_ORDER:
                        continue

                    # 成行想定の実効取得単価（手数料・すべり込み）
                    entry_raw  = sig["open"]  # エントリーは「次足の始値」想定
                    entry_exec = entry_raw * (1 + SLIPPAGE) * (1 + FEE_RATE)
                    qty = invest / entry_exec

                    if up:
                        _ = buy_market(up, t, invest)

                    positions[t] = Position(t, entry_raw, entry_exec, qty)
                    sma48_str = f"{sig['SMA48']:.6f}" if sig["SMA48"] is not None else "nan"
                    safe_print(f"[BUY ] {t}  px={entry_exec:.3f}  invest={invest:,} KRW  reason={sig['reason']}  "
                               f"RSI={sig['RSI']:.1f}  SMA48={sma48_str}  SMA48_UP={sig['SMA48_UP']}")

            # ---- 保有銘柄の監視（ユニバース外でも必ず監視）----
            for t, pos in list(positions.items()):
                df = get_ohlcv_kst(t, INTERVAL, BARS)
                if df.empty:
                    continue

                sig = evaluate_signals(df)
                if sig is None:
                    continue

                # 手動売却検知：口座の数量が 0 なら内部ポジションを削除
                real_qty = get_balance_coin(up, t) if up else pos.qty
                if real_qty <= 0:
                    safe_print(f"[INFO] {t} 手動売却を検知 → ポジションを削除")
                    del positions[t]
                    continue

                # 損切り（現在値ベース、即時）
                cur = pyupbit.get_current_price(t) or sig["close"]
                if cur <= pos.entry_raw * (1 - STOP_LOSS_PCT):
                    sell_px = cur * (1 - SLIPPAGE) * (1 - FEE_RATE)
                    pnl = (sell_px / pos.entry_exec - 1.0) * 100.0
                    if up:
                        _ = sell_market(up, t, real_qty)
                    safe_print(f"[SELL] {t}  STOP(-{STOP_LOSS_PCT*100:.0f}%)  pnl={pnl:.2f}%")
                    del positions[t]
                    continue

                # RSI ルール（確定足ベースの利確・撤退）
                rsi = sig["RSI"]
                if rsi > RSI_OVER:
                    pos.rsi_over = True
                if pos.rsi_over and rsi <= RSI_OVER:
                    sell_px = sig["open"] * (1 - SLIPPAGE) * (1 - FEE_RATE)
                    pnl = (sell_px / pos.entry_exec - 1.0) * 100.0
                    if up:
                        _ = sell_market(up, t, real_qty)
                    safe_print(f"[SELL] {t}  RSI<=80  pnl={pnl:.2f}%")
                    del positions[t]
                    continue

            time.sleep(LOOP_SLEEP_SEC)

        except KeyboardInterrupt:
            safe_print("停止（Ctrl+C）")
            break
        except Exception as e:
            safe_print("[ERR]", e)
            traceback.print_exc()
            time.sleep(3)


# ============================================================
if __name__ == "__main__":
    run()
