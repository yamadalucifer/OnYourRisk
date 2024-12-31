import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

# データの読み込みと整形
df = pd.read_csv('5min_data.csv', sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# 特徴量の計算
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
df['Intraday_Range'] = (df['High'] - df['Low']) / df['Low']
df['MA_Bias'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
df['VWAP'] = (df['Close'] * df['TickVol']).cumsum() / df['TickVol'].cumsum()
stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()
df['ROC'] = df['Close'].pct_change(periods=12) * 100
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek

# ターゲット（1時間後の終値）を作成
df['Target'] = df['Close'].shift(-12)

# 欠損値を削除
df.dropna(inplace=True)

# 特徴量リスト
features = ['Close', 'SMA_10', 'BB_upper', 'BB_lower', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
            'Intraday_Range', 'MA_Bias', 'VWAP', 'Stoch_K', 'Stoch_D', 'ROC', 'Hour', 'DayOfWeek', 'Target']

# 最小値と最大値を取得
min_values = df[features].min().values
max_values = df[features].max().values

# 結果をDataFrameにまとめる
scaling_info = pd.DataFrame({
    'Feature': features,
    'Min': min_values,
    'Max': max_values
})

# CSVとして保存（MQL5で利用する場合）
scaling_info.to_csv('scaling_info.csv', index=False)
print(scaling_info)
