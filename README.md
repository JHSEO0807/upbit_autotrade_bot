## upbit_ichimoku_autotradebot
## Created by JAEHEN SEO
============================================================================================
## 日本語バージョン
## 概要
このプロジェクトは、一目均衡表・スキューズモメンタム・ADX・MACD・RSIを組み合わせた自動売買戦略をPythonで実装したものです。
アップビット（KRW市場）における短期トレンド転換を検出し、明確な上昇初動のみを狙う構成になっています。

## 売買ロジック概要
買い条件：
スキューズモメンタムがライム色（上昇圧力）
終値が48SMAより上、かつ48SMAの傾きが上向き（上昇トレンド）
以下のいずれか：
一目均衡表の転換線が基準線をゴールデンクロス
転換線と基準線がほぼ一致し（横ばい）、+DI > ADX > -DI かつ MACD > シグナル
売り条件：
RSI(14) が 80 を超えた後に再び 80 以下へ下落
またはエントリー価格から -1% の下落でストップロス

## 技術仕様
データソース：Upbit Public API（pyupbit）
分析周期：5分足
インジケータ：Ichimoku, Squeeze Momentum, ADX, MACD, RSI, SMA
売買方式：成行（スリッページ・手数料を考慮）
DRY_RUNモードでの模擬取引を標準搭載（実運用はAPIキー設定で可能）

## 技術スタック
| カテゴリ | 使用技術 | 説明 |
| 言語| Python 3.10+ | 売買ロジック・データ処理の中心 |
| データソース| Upbit Open API (`pyupbit`) | KRWマーケットのリアルタイムデータを取得 |
| データ処理 | pandas, numpy*| 時系列データ処理・統計計算 |
| インジケータ | Ichimoku・Squeeze Momentum・ADX・MACD・RSI・SMA| 数式ベース |
| 通信/API | requests | ティッカー・マーケット情報のREST通信 |
| ループ構造 | time.sleep(15秒) | 5分足の完了タイミングを監視する軽量ループ |
| 実行モード | DRY_RUNモード | 実取引なしでリアルタイム検証可能（デフォルト）|
| 実売買 | pyupbit.Upbitクライアント | APIキー設定により実運用対応 |
| リスク管理 | -1%ストップロス / RSI80反落売却 / 重複エントリー防止 / 手動売却検出 |

## コンセプト
「上昇初動×モメンタム強化」を主軸とした戦略であり、  
ノイズの多い相場環境でも確度の高いトレンド転換を捉えることを目的としています。  
リアルタイム模擬取引（DRY_RUN）により、バックテストなしで即時検証が可能です。

=====================================================================================================================
## 🇺🇸 English Version
This project is a Python-based automated trading system that combines
Ichimoku Cloud, Squeeze Momentum, ADX, MACD, and RSI indicators
to detect early uptrend momentum on Upbit (KRW market)

## Strategy Overview
Buy Conditions:
Squeeze Momentum histogram turns lime (positive momentum)
Price is above 48SMA, and 48SMA slope > 0 (upward trend)
Either:
Tenkan-sen crosses above Kijun-sen (Ichimoku Golden Cross), or
Tenkan ≈ Kijun (flat alignment) and +DI > ADX > -DI and MACD > Signal
Sell Conditions:
RSI(14) drops back below 80 after exceeding it
OR Stop-loss at -1% from entry price

## Technical Details
Data Source: Upbit API via pyupbit
Timeframe: 5-minute candles
Indicators: Ichimoku, Squeeze Momentum, ADX, MACD, RSI, SMA
Execution: Market orders (slippage & fees included)
Built-in DRY_RUN mode for safe paper trading enable live mode with API keys for real execution

## Tech Stack
| Category | Tool / Library | Description |
| Language | Python 3.10+ | Core trading and data logic |
| Data Source | Upbit Open API (`pyupbit`) | Real-time OHLCV and ticker data for KRW pairs |
| Data Handling | pandas, numpy | Time-series processing, rolling calculations |
| Indicators | Ichimoku, Squeeze Momentum, ADX, MACD, RSI, SMA** | Pure Python implementations |
| Networking / API | requests | REST API calls for market info |
| Execution Loop | time.sleep(15s) | Lightweight 5-minute candle monitor |
| Simulation Mode | DRY_RUN (default) | Real-time paper trading without live orders |
| Live Trading | pyupbit.Upbit | Enabled with API keys |
| Risk Management | Stop-loss (-1%), RSI 80 fallback, duplicate entry prevention, manual-sell detection |

## Concept
This strategy aims to capture the early stage of a bullish breakout* 
through the synergy of Ichimoku signals and Squeeze Momentum compression.  
With the built-in DRY_RUN mode, users can test live signals  
without financial risk, making it ideal for algorithmic strategy validation.

======================================================================================================================
