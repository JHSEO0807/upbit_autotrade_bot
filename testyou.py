#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyupbit
import pandas as pd
import numpy as np
import time
import logging
import requests
import sys
import os
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upbit_autotrade.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Upbit API keys (recommended to use environment variables or separate config file)
ACCESS_KEY = "your_access_key"
API_SECRET = "your_secret_key"

# Global settings
DRY_RUN = True  # Set to True for paper trading (simulation), False for real trading
INITIAL_BALANCE = 1000000  # Initial virtual balance for dry run (1 million KRW)
EXCLUDED_COINS = ['KRW-XRP', 'KRW-BTC', 'KRW-ETH', 'KRW-USDT']  # Coins to exclude
MIN_VOLUME = 20000000000  # Minimum trading volume: 20 billion KRW
TOP_GAINERS_COUNT = 20  # Top N gainers
MONITOR_INTERVAL = 300  # Monitoring interval in seconds (5 minutes)
INVESTMENT_PER_COIN = 0.95  # Investment ratio per coin (95% of balance)

# Trading interval settings
# Options: "minute1", "minute3", "minute5", "minute10", "minute15", "minute30", "minute60", "minute240", "day", "week", "month"
CANDLE_INTERVAL = "minute5"  # Candle interval for technical analysis

# Excel log file
TRADE_LOG_FILE = "trade_history.xlsx"  # Excel file to save trade history


class UpbitAutoTrader:
    def __init__(self, access_key, secret_key, dry_run=True):
        """Initialize Upbit auto-trading bot"""
        self.dry_run = dry_run
        self.upbit = pyupbit.Upbit(access_key, secret_key) if not dry_run else None
        self.target_coins = []  # Current target coins list

        # Virtual portfolio for dry run mode
        if self.dry_run:
            self.virtual_krw_balance = INITIAL_BALANCE
            self.virtual_portfolio = {}  # {ticker: {amount, avg_buy_price}}
            self.trade_history = []  # Track all trades
            self.win_count = 0
            self.lose_count = 0
            logger.info(f"*** 모의매매 모드 활성화 - 실제 거래가 체결되지 않습니다 ***")
            logger.info(f"초기 가상 자본금: {self.virtual_krw_balance:,.0f}원")

            # Load existing trade history from Excel
            self.load_trade_history_from_excel()

    def load_trade_history_from_excel(self):
        """Load existing trade history from Excel file"""
        if not os.path.exists(TRADE_LOG_FILE):
            logger.info(f"새로운 거래 기록 파일 생성 예정: {TRADE_LOG_FILE}")
            return

        try:
            df = pd.read_excel(TRADE_LOG_FILE)
            if len(df) > 0:
                # Count wins and losses from existing data
                sell_trades = df[df['거래유형'] == '매도']
                if len(sell_trades) > 0:
                    self.win_count = len(sell_trades[sell_trades['수익률(%)'] > 0])
                    self.lose_count = len(sell_trades[sell_trades['수익률(%)'] <= 0])

                logger.info(f"기존 거래 기록 로드 완료: {len(df)}건 (승: {self.win_count}, 패: {self.lose_count})")
        except Exception as e:
            logger.warning(f"거래 기록 파일 로드 실패: {e}")

    def save_trade_to_excel(self, trade):
        """Save trade to Excel file (append mode)"""
        try:
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning)

            # Prepare trade data for Excel
            trade_data = {
                '시간': trade['time'].strftime('%Y-%m-%d %H:%M:%S'),
                '거래유형': trade['type_kr'],
                '종목': trade['ticker'],
                '수량': trade['amount'],
                '체결가격': trade['price'],
                '거래금액': trade['value']
            }

            # Add profit data for sell trades
            if trade['type'] == 'SELL':
                trade_data['매수가격'] = trade.get('buy_price', 0)
                trade_data['손익(원)'] = trade.get('profit', 0)
                trade_data['수익률(%)'] = trade.get('profit_rate', 0)
            else:
                trade_data['매수가격'] = 0
                trade_data['손익(원)'] = 0
                trade_data['수익률(%)'] = 0

            # Convert to DataFrame
            new_row = pd.DataFrame([trade_data])

            # Append to existing file or create new
            if os.path.exists(TRADE_LOG_FILE):
                existing_df = pd.read_excel(TRADE_LOG_FILE)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row

            # Save to Excel
            updated_df.to_excel(TRADE_LOG_FILE, index=False, engine='openpyxl')

        except Exception as e:
            pass  # Silent fail for Excel logging

    def get_top_gainers(self):
        """Get top gainers by daily change rate with volume filter (optimized with batch API call)"""
        try:
            # Get all KRW market tickers
            tickers = pyupbit.get_tickers(fiat="KRW")

            # Filter out excluded coins
            tickers = [t for t in tickers if t not in EXCLUDED_COINS]

            # Batch request to Upbit API for all tickers at once
            url = "https://api.upbit.com/v1/ticker"
            params = {"markets": ",".join(tickers)}

            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"API request failed with status code: {response.status_code}")
                return []

            ticker_data = response.json()

            market_data = []

            for data in ticker_data:
                try:
                    ticker = data['market']
                    change_rate = data['signed_change_rate'] * 100  # Convert to percentage
                    volume_krw = data['acc_trade_price_24h']  # 24-hour accumulated trade price
                    current_price = data['trade_price']

                    # Filter by minimum volume (20 billion KRW)
                    if volume_krw >= MIN_VOLUME:
                        market_data.append({
                            'ticker': ticker,
                            'change_rate': change_rate,
                            'volume_krw': volume_krw,
                            'current_price': current_price
                        })

                except Exception as e:
                    logger.warning(f"Failed to parse data for {data.get('market', 'unknown')}: {e}")
                    continue

            # Sort by change rate and select top N
            market_data.sort(key=lambda x: x['change_rate'], reverse=True)
            top_coins = market_data[:TOP_GAINERS_COUNT]

            return [coin['ticker'] for coin in top_coins]

        except Exception as e:
            logger.error(f"상위 종목 조회 중 오류 발생: {e}")
            return []

    def calculate_sma(self, df, period):
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=period).mean()

    def calculate_adx(self, df, period=14):
        """Calculate ADX, DI+, DI-"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM, -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.copy()
        minus_dm = low_diff.copy()

        plus_dm[((high_diff < low_diff) | (high_diff < 0))] = 0
        minus_dm[((low_diff < high_diff) | (low_diff < 0))] = 0

        # Calculate TR (True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = tr.rolling(window=period).mean()

        # Calculate +DI, -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di

    def check_buy_condition(self, ticker):
        """
        Check buy conditions (DMI/ADX strategy):
        1. DI+ > ADX > DI-
        2. DI+ rising (DI+ > DI+[1])
        3. DI- falling (DI- < DI-[1])
        4. DI+ angle >= 70 degrees
        """
        try:
            # Get sufficient data for ADX calculation
            df = pyupbit.get_ohlcv(ticker, interval=CANDLE_INTERVAL, count=200)

            if df is None or len(df) < 50:
                return False

            # Calculate ADX, DI+, DI-
            df['adx'], df['di_plus'], df['di_minus'] = self.calculate_adx(df, period=14)

            # Get latest and previous data
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # Check if values are valid
            if pd.isna(latest['adx']) or pd.isna(latest['di_plus']) or pd.isna(latest['di_minus']):
                return False
            if pd.isna(prev['di_plus']) or pd.isna(prev['di_minus']):
                return False

            # Condition 1: DI+ > ADX > DI-
            dmi_order_ok = (latest['di_plus'] > latest['adx']) and (latest['adx'] > latest['di_minus'])

            # Condition 2: DI+ rising
            di_plus_up = latest['di_plus'] > prev['di_plus']

            # Condition 3: DI- falling
            di_minus_down = latest['di_minus'] < prev['di_minus']

            # Condition 4: DI+ angle >= 70 degrees
            slope = latest['di_plus'] - prev['di_plus']
            angle_deg = np.arctan(slope) * 180 / np.pi
            angle_condition = angle_deg >= 70

            # All conditions must be true
            result = dmi_order_ok and di_plus_up and di_minus_down and angle_condition
            return result

        except Exception as e:
            logger.error(f"매수 조건 확인 중 오류 ({ticker}): {e}")
            return False

    def check_sell_condition(self, ticker):
        """
        Check sell conditions (DMI/ADX strategy):
        - DI+ <= DI-
        """
        try:
            df = pyupbit.get_ohlcv(ticker, interval=CANDLE_INTERVAL, count=200)

            if df is None or len(df) < 50:
                return False

            # Calculate ADX, DI+, DI-
            df['adx'], df['di_plus'], df['di_minus'] = self.calculate_adx(df, period=14)

            # Get latest data
            latest = df.iloc[-1]

            if pd.isna(latest['di_plus']) or pd.isna(latest['di_minus']):
                return False

            # Sell condition: DI+ <= DI-
            result = latest['di_plus'] <= latest['di_minus']
            return result

        except Exception as e:
            logger.error(f"매도 조건 확인 중 오류 ({ticker}): {e}")
            return False

    def get_balance(self, ticker=None):
        """Get balance (virtual or real depending on mode)"""
        try:
            if self.dry_run:
                # Virtual balance for dry run
                if ticker is None:
                    return self.virtual_krw_balance
                else:
                    portfolio = self.virtual_portfolio.get(ticker, {})
                    return portfolio.get('amount', 0)
            else:
                # Real balance
                if ticker is None:
                    return self.upbit.get_balance("KRW")
                else:
                    coin = ticker.split('-')[1]
                    return self.upbit.get_balance(coin)
        except Exception as e:
            logger.error(f"잔고 조회 중 오류 ({ticker}): {e}")
            return 0

    def buy_coin(self, ticker):
        """Buy coin (virtual or real depending on mode)"""
        try:
            krw_balance = self.get_balance()

            if krw_balance < 5000:  # Minimum order amount
                logger.warning(f"잔고 부족: {krw_balance:,.0f}원")
                return False

            # Check current holdings
            holdings = [t for t in self.target_coins if self.get_balance(t) > 0]

            # Calculate buy amount for diversification
            buy_amount = krw_balance * INVESTMENT_PER_COIN

            if buy_amount < 5000:
                logger.warning(f"매수 금액이 너무 작음: {buy_amount:,.0f}원")
                return False

            # Get current price
            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                logger.warning(f"현재가 조회 실패: {ticker}")
                return False

            if self.dry_run:
                # Virtual buy for dry run
                coin_amount = buy_amount / current_price
                self.virtual_krw_balance -= buy_amount

                # Update portfolio with average buy price
                if ticker in self.virtual_portfolio:
                    existing = self.virtual_portfolio[ticker]
                    total_amount = existing['amount'] + coin_amount
                    avg_price = ((existing['amount'] * existing['avg_buy_price']) +
                                (coin_amount * current_price)) / total_amount
                    self.virtual_portfolio[ticker] = {
                        'amount': total_amount,
                        'avg_buy_price': avg_price
                    }
                else:
                    self.virtual_portfolio[ticker] = {
                        'amount': coin_amount,
                        'avg_buy_price': current_price
                    }

                # Record trade
                trade = {
                    'time': datetime.now(),
                    'type': 'BUY',
                    'type_kr': '매수',
                    'ticker': ticker,
                    'amount': coin_amount,
                    'price': current_price,
                    'value': buy_amount
                }
                self.trade_history.append(trade)
                self.save_trade_to_excel(trade)

                # Calculate current win rate
                total_trades = self.win_count + self.lose_count
                win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

                time_str = datetime.now().strftime('%H:%M:%S')
                logger.info(f"[매수] {ticker} {time_str} {current_price:,.0f}원")
                return True
            else:
                # Real buy
                result = self.upbit.buy_market_order(ticker, buy_amount)

                if result:
                    logger.info(f"[실전매매] 매수 체결: {ticker}, 금액: {buy_amount:,.0f}원")
                    return True
                else:
                    logger.warning(f"매수 실패: {ticker}")
                    return False

        except Exception as e:
            logger.error(f"매수 중 오류 발생 ({ticker}): {e}")
            return False

    def sell_coin(self, ticker):
        """Sell coin (virtual or real depending on mode)"""
        try:
            balance = self.get_balance(ticker)

            if balance <= 0:
                return False

            current_price = pyupbit.get_current_price(ticker)
            if current_price is None:
                return False

            sell_value = balance * current_price

            if self.dry_run:
                # Get average buy price
                portfolio = self.virtual_portfolio.get(ticker, {})
                avg_buy_price = portfolio.get('avg_buy_price', current_price)
                buy_value = balance * avg_buy_price

                # Calculate profit/loss
                profit = sell_value - buy_value
                profit_rate = (profit / buy_value) * 100 if buy_value > 0 else 0

                # Update win/lose count
                if profit > 0:
                    self.win_count += 1
                    result_text = "익절"
                else:
                    self.lose_count += 1
                    result_text = "손절"

                # Virtual sell for dry run
                self.virtual_krw_balance += sell_value

                # Record trade before removing from portfolio
                trade = {
                    'time': datetime.now(),
                    'type': 'SELL',
                    'type_kr': '매도',
                    'ticker': ticker,
                    'amount': balance,
                    'price': current_price,
                    'value': sell_value,
                    'buy_price': avg_buy_price,
                    'profit': profit,
                    'profit_rate': profit_rate
                }
                self.trade_history.append(trade)
                self.save_trade_to_excel(trade)

                # Remove from portfolio
                if ticker in self.virtual_portfolio:
                    del self.virtual_portfolio[ticker]

                # Calculate current win rate
                total_trades = self.win_count + self.lose_count
                win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

                time_str = datetime.now().strftime('%H:%M:%S')
                logger.info(f"[매도] {ticker} {time_str} {current_price:,.0f}원 {profit_rate:+.2f}% 승률:{win_rate:.1f}%")
                return True
            else:
                # Real sell
                result = self.upbit.sell_market_order(ticker, balance)

                if result:
                    logger.info(f"[실전매매] 매도 체결: {ticker}, 금액: {sell_value:,.0f}원")
                    return True
                else:
                    logger.warning(f"매도 실패: {ticker}")
                    return False

        except Exception as e:
            logger.error(f"매도 중 오류 발생 ({ticker}): {e}")
            return False

    def update_target_coins(self):
        """Update target coins list"""
        new_targets = self.get_top_gainers()

        # Sell coins that are no longer in target list
        for ticker in self.target_coins:
            if ticker not in new_targets:
                balance = self.get_balance(ticker)
                if balance > 0:
                    self.sell_coin(ticker)

        self.target_coins = new_targets

    def monitor_and_trade(self):
        """Monitor and execute trades"""
        # Check holdings and sell/buy conditions
        for ticker in self.target_coins:
            balance = self.get_balance(ticker)

            if balance > 0:
                # Holding coin - check sell condition
                if self.check_sell_condition(ticker):
                    self.sell_coin(ticker)
            else:
                # Not holding - check buy condition
                if self.check_buy_condition(ticker):
                    self.buy_coin(ticker)

    def print_portfolio(self):
        """Print current portfolio"""
        krw_balance = self.get_balance()
        total_value = krw_balance
        total_profit = 0
        holding_count = 0

        if self.dry_run and self.virtual_portfolio:
            for ticker in self.virtual_portfolio:
                portfolio = self.virtual_portfolio[ticker]
                balance = portfolio['amount']
                avg_buy_price = portfolio['avg_buy_price']

                current_price = pyupbit.get_current_price(ticker)
                if current_price:
                    value = balance * current_price
                    buy_value = balance * avg_buy_price
                    profit = value - buy_value
                    profit_rate = (profit / buy_value * 100) if buy_value > 0 else 0

                    total_value += value
                    total_profit += profit
                    holding_count += 1

                    logger.info(f"  [보유] {ticker}: {avg_buy_price:,.0f}->{current_price:,.0f}원 | "
                              f"평가: {value:,.0f}원 | 손익: {profit:+,.0f}원({profit_rate:+.2f}%)")

        # Show P&L for dry run mode
        if self.dry_run:
            pnl = total_value - INITIAL_BALANCE
            pnl_percent = (pnl / INITIAL_BALANCE) * 100
            total_trades = self.win_count + self.lose_count
            win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

            logger.info(f"[요약] 총평가: {total_value:,.0f}원 | 총손익: {pnl:+,.0f}원({pnl_percent:+.2f}%) | "
                      f"보유: {holding_count}개 | 승률: {win_rate:.1f}%({self.win_count}승{self.lose_count}패)")

        logger.info("")  # Empty line for readability

    def print_trade_summary(self):
        """Print trade history summary (for dry run mode)"""
        if not self.dry_run or len(self.trade_history) == 0:
            return

        logger.info("\n" + "=" * 70)
        logger.info("=== 거래 내역 요약 ===")
        logger.info("=" * 70)

        buy_count = sum(1 for t in self.trade_history if t['type'] == 'BUY')
        sell_count = sum(1 for t in self.trade_history if t['type'] == 'SELL')

        logger.info(f"총 거래 횟수: {len(self.trade_history)}회 (매수: {buy_count}회, 매도: {sell_count}회)")
        logger.info(f"\n최근 거래 내역:")

        # Show last 20 trades
        for trade in self.trade_history[-20:]:
            trade_type = "매수" if trade['type'] == 'BUY' else "매도"
            time_str = trade['time'].strftime('%Y-%m-%d %H:%M:%S')

            if trade['type'] == 'SELL' and 'profit_rate' in trade:
                logger.info(f"  [{time_str}] {trade_type:2s} {trade['ticker']:12s} "
                          f"{trade['amount']:>10.4f}개 @ {trade['price']:>10,.0f}원 = "
                          f"{trade['value']:>12,.0f}원 "
                          f"(손익: {trade['profit']:+,.0f}원 / {trade['profit_rate']:+.2f}%)")
            else:
                logger.info(f"  [{time_str}] {trade_type:2s} {trade['ticker']:12s} "
                          f"{trade['amount']:>10.4f}개 @ {trade['price']:>10,.0f}원 = "
                          f"{trade['value']:>12,.0f}원")

        # Calculate final P&L
        krw_balance = self.get_balance()
        total_value = krw_balance

        for ticker in self.virtual_portfolio:
            portfolio = self.virtual_portfolio[ticker]
            balance = portfolio['amount']
            if balance > 0:
                current_price = pyupbit.get_current_price(ticker)
                if current_price:
                    total_value += balance * current_price

        final_pnl = total_value - INITIAL_BALANCE
        final_pnl_percent = (final_pnl / INITIAL_BALANCE) * 100
        win_rate = (self.win_count / (self.win_count + self.lose_count) * 100) if (self.win_count + self.lose_count) > 0 else 0

        logger.info(f"\n" + "=" * 70)
        logger.info(f"=== 최종 결과 ===")
        logger.info(f"초기 자본금: {INITIAL_BALANCE:,.0f}원")
        logger.info(f"최종 평가금액: {total_value:,.0f}원")
        logger.info(f"총 손익: {final_pnl:+,.0f}원 ({final_pnl_percent:+.2f}%)")
        logger.info(f"최종 승률: {win_rate:.1f}% ({self.win_count}승 {self.lose_count}패)")
        logger.info("=" * 70)

    def run(self):
        """Main execution loop"""
        # Convert interval to Korean display
        interval_display = {
            "minute1": "1분봉", "minute3": "3분봉", "minute5": "5분봉",
            "minute10": "10분봉", "minute15": "15분봉", "minute30": "30분봉",
            "minute60": "60분봉", "minute240": "240분봉",
            "day": "일봉", "week": "주봉", "month": "월봉"
        }

        logger.info("\n" + "=" * 70)
        logger.info("=== 업비트 자동매매 봇 시작 ===")
        logger.info("=" * 70)
        logger.info(f"모드: {'모의매매 (DRY RUN)' if self.dry_run else '실전매매 (LIVE)'}")
        logger.info(f"차트 분석 주기: {interval_display.get(CANDLE_INTERVAL, CANDLE_INTERVAL)}")
        logger.info(f"모니터링 주기: {MONITOR_INTERVAL}초")
        logger.info(f"최소 거래대금: {MIN_VOLUME/100000000:.0f}억원")
        logger.info(f"상위 종목 수: {TOP_GAINERS_COUNT}개")
        logger.info("=" * 70 + "\n")

        iteration = 0

        while True:
            try:
                iteration += 1

                # Update target coins list
                self.update_target_coins()

                # Monitor and trade
                self.monitor_and_trade()

                # Wait
                time.sleep(MONITOR_INTERVAL)

            except KeyboardInterrupt:
                logger.info("\n\n사용자에 의해 봇이 중지되었습니다.")
                break
            except Exception as e:
                logger.error(f"메인 루프 오류 발생: {e}")
                logger.info(f"{MONITOR_INTERVAL}초 후 재시도합니다...")
                time.sleep(MONITOR_INTERVAL)

        # Print final summary when bot stops
        self.print_trade_summary()


def main():
    """Main function"""
    # Check API keys (not required for dry run mode)
    if not DRY_RUN:
        if ACCESS_KEY == "your_access_key" or API_SECRET == "your_secret_key":
            logger.error("업비트 API 키를 설정해주세요!")
            logger.error("API 키 발급: https://upbit.com/mypage/open_api_management")
            return
    else:
        logger.info("모의매매 모드로 실행 중 - API 키가 필요하지 않습니다")

    # Run auto-trading bot
    trader = UpbitAutoTrader(ACCESS_KEY, API_SECRET, dry_run=DRY_RUN)
    trader.run()


if __name__ == "__main__":
    main()
