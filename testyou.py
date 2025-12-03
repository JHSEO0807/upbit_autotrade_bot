#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyupbit
import pandas as pd
import numpy as np
import time
import logging
import requests
from datetime import datetime

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upbit_autotrade.log'),
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

            logger.info(f"매매대상 {len(top_coins)}개 선정 완료 (상승률 TOP, 거래대금 {MIN_VOLUME/100000000:.0f}억 이상)")
            if top_coins:
                top_3 = top_coins[:3]
                coin_list = ", ".join([f"{c['ticker']}({c['change_rate']:+.1f}%)" for c in top_3])
                logger.info(f"  상위 3개: {coin_list}")

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
        Check buy conditions:
        - SMA5 > SMA10 > SMA20
        - SMA40 > SMA40[1]
        - DI+ > ADX > DI-
        """
        try:
            # Get sufficient data for ADX calculation
            df = pyupbit.get_ohlcv(ticker, interval=CANDLE_INTERVAL, count=200)

            if df is None or len(df) < 50:
                return False

            # Calculate SMAs
            df['sma5'] = self.calculate_sma(df, 5)
            df['sma10'] = self.calculate_sma(df, 10)
            df['sma20'] = self.calculate_sma(df, 20)
            df['sma40'] = self.calculate_sma(df, 40)

            # Calculate ADX, DI+, DI-
            df['adx'], df['plus_di'], df['minus_di'] = self.calculate_adx(df)

            # Get latest data
            latest = df.iloc[-1]
            prev_sma40 = df.iloc[-2]['sma40']

            # Check conditions
            condition1 = latest['sma5'] > latest['sma10'] > latest['sma20']
            condition2 = latest['sma40'] > prev_sma40
            condition3 = latest['plus_di'] > latest['adx'] > latest['minus_di']

            if pd.isna(condition1) or pd.isna(condition2) or pd.isna(condition3):
                return False

            result = condition1 and condition2 and condition3

            if result:
                logger.info(f"✅ 매수 시그널: {ticker}")

            return result

        except Exception as e:
            logger.error(f"매수 조건 확인 중 오류 ({ticker}): {e}")
            return False

    def check_sell_condition(self, ticker):
        """
        Check sell conditions:
        - ADX < ADX[1] and ADX[1] < ADX[2] and ADX[2] < ADX[3]
        (ADX declining for 3 consecutive candles)
        """
        try:
            df = pyupbit.get_ohlcv(ticker, interval=CANDLE_INTERVAL, count=200)

            if df is None or len(df) < 50:
                return False

            # Calculate ADX
            df['adx'], _, _ = self.calculate_adx(df)

            # Get last 4 ADX values
            adx_0 = df.iloc[-1]['adx']  # Current
            adx_1 = df.iloc[-2]['adx']  # 1 candle ago
            adx_2 = df.iloc[-3]['adx']  # 2 candles ago
            adx_3 = df.iloc[-4]['adx']  # 3 candles ago

            if pd.isna(adx_0) or pd.isna(adx_1) or pd.isna(adx_2) or pd.isna(adx_3):
                return False

            result = (adx_0 < adx_1) and (adx_1 < adx_2) and (adx_2 < adx_3)

            if result:
                logger.info(f"🔴 매도 시그널: {ticker}")

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
                    'ticker': ticker,
                    'amount': coin_amount,
                    'price': current_price,
                    'value': buy_amount
                }
                self.trade_history.append(trade)

                # Calculate current win rate
                total_trades = self.win_count + self.lose_count
                win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

                logger.info(f"💰 매수체결: {ticker} | {current_price:,.0f}원 × {coin_amount:.4f}개 = {buy_amount:,.0f}원 | 승률: {win_rate:.1f}%")
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
                    'ticker': ticker,
                    'amount': balance,
                    'price': current_price,
                    'value': sell_value,
                    'buy_price': avg_buy_price,
                    'profit': profit,
                    'profit_rate': profit_rate
                }
                self.trade_history.append(trade)

                # Remove from portfolio
                if ticker in self.virtual_portfolio:
                    del self.virtual_portfolio[ticker]

                # Calculate current win rate
                total_trades = self.win_count + self.lose_count
                win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

                result_emoji = "🟢" if profit > 0 else "🔴"
                logger.info(f"{result_emoji} 매도체결({result_text}): {ticker} | {avg_buy_price:,.0f}→{current_price:,.0f}원 | "
                          f"손익: {profit:+,.0f}원({profit_rate:+.2f}%) | 승률: {win_rate:.1f}%({self.win_count}승{self.lose_count}패)")
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
        removed_count = 0
        for ticker in self.target_coins:
            if ticker not in new_targets:
                balance = self.get_balance(ticker)
                if balance > 0:
                    logger.info(f"⚠️ {ticker} 대상제외 - 전량매도")
                    self.sell_coin(ticker)
                    removed_count += 1

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

        # Print current portfolio
        self.print_portfolio()

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

                    logger.info(f"  📊 {ticker}: {avg_buy_price:,.0f}→{current_price:,.0f}원 | "
                              f"평가: {value:,.0f}원 | 손익: {profit:+,.0f}원({profit_rate:+.2f}%)")

        # Show P&L for dry run mode
        if self.dry_run:
            pnl = total_value - INITIAL_BALANCE
            pnl_percent = (pnl / INITIAL_BALANCE) * 100
            total_trades = self.win_count + self.lose_count
            win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0

            logger.info(f"💼 총평가: {total_value:,.0f}원 | 총손익: {pnl:+,.0f}원({pnl_percent:+.2f}%) | "
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
                logger.info(f"\n{'='*70}")
                logger.info(f"🔄 #{iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")

                # Update target coins list
                self.update_target_coins()

                # Monitor and trade
                self.monitor_and_trade()

                # Wait
                logger.info(f"⏳ {MONITOR_INTERVAL}초 대기...\n")
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
