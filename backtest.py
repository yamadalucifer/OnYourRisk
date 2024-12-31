import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
import matplotlib.pyplot as plt
import gc  # メモリ管理用

# 1. 学習済みモデルのロード
model = load_model('lstm_model.h5')
print("保存済みの学習済みモデルをロードしました")

# 2. データの読み込みと整形
df = pd.read_csv('5min_data.csv', sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# 3. データのサブセット（最新10万行を利用してメモリ負荷軽減）
df = df.tail(100000)

# 4. 特徴量の計算（学習時と同じ手順）
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
df.dropna(inplace=True)

# 特徴量リスト
features = ['Close', 'SMA_10', 'BB_upper', 'BB_lower', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
            'Intraday_Range', 'MA_Bias', 'VWAP', 'Stoch_K', 'Stoch_D', 'ROC', 'Hour', 'DayOfWeek']

# 5. データのスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features + ['Target']])

# 6. 時系列データの作成（学習時と同じ手順）
lookback = 300  # 過去300本のデータを使用

def create_sequences(data, lookback, target_index):
    X, y = [], []
    for i in range(lookback, len(data) - target_index):
        X.append(data[i-lookback:i, :-1])  # 過去`lookback`期間の特徴量
        y.append(data[i + target_index, -1])  # `target_index`先の値
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback, target_index=12)

# 7. バッチ処理を使ったモデル予測
batch_size = 10000  # バッチサイズを設定
y_pred = []
for i in range(0, len(X), batch_size):
    batch = X[i:i + batch_size]
    y_pred.extend(model.predict(batch))
y_pred = np.array(y_pred)  # リストを配列に変換

# 8. 予測値と実際値をスケールを元に戻す
y_pred_rescaled = scaler.inverse_transform(
    np.hstack((y_pred, np.zeros((y_pred.shape[0], scaled_data.shape[1] - 1)))))[:, 0]

y_rescaled = scaler.inverse_transform(
    np.hstack((y.reshape(-1, 1), np.zeros((y.shape[0], scaled_data.shape[1] - 1)))))[:, 0]

# 9. バックテストの実施（ストップロス・テイクプロフィットを適用）
backtest_df = pd.DataFrame({
    'Actual': y_rescaled,
    'Predicted': y_pred_rescaled,
    'Close': df['Close'].iloc[lookback:len(y_pred) + lookback].values,
    'RSI': df['RSI'].iloc[lookback:len(y_pred) + lookback].values,
    'ATR': df['ATR'].iloc[lookback:len(y_pred) + lookback].values
})

# 売買シグナル生成（予測値と現在値の差に基づく）
threshold = 0.01  # 1pipの閾値
backtest_df['Signal'] = np.where(
    (backtest_df['Predicted'] - backtest_df['Close'] > threshold) & (backtest_df['RSI'] < 70), 1,  # 買いシグナル
    np.where((backtest_df['Predicted'] - backtest_df['Close'] < -threshold) & (backtest_df['RSI'] > 30), -1, 0)  # 売りシグナルまたは何もしない
)

# 戦略リターンの計算（シグナルに基づく取引）
backtest_df['Return'] = backtest_df['Close'].pct_change()  # 実際値のリターン
backtest_df['Strategy_Return'] = backtest_df['Signal'].shift(1) * backtest_df['Return']  # 戦略リターン

# ストップロスとテイクプロフィットの適用
stop_loss = -0.001  # 0.1%の損失で決済
take_profit = 0.002  # 0.2%の利益で決済
backtest_df['Strategy_Return'] = np.where(
    backtest_df['Strategy_Return'] < stop_loss, stop_loss,
    np.where(backtest_df['Strategy_Return'] > take_profit, take_profit, backtest_df['Strategy_Return'])
)

# 累積リターンの計算
backtest_df['Cumulative_Actual'] = (1 + backtest_df['Return']).cumprod()
backtest_df['Cumulative_Strategy'] = (1 + backtest_df['Strategy_Return']).cumprod()

# シャープレシオの計算
strategy_returns = backtest_df['Strategy_Return'].dropna()
sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

# 最大ドローダウンの計算
def max_drawdown(cumulative_returns):
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    return drawdown.min()

max_dd = max_drawdown(backtest_df['Cumulative_Strategy'])

# 勝率の計算
win_rate = (backtest_df['Strategy_Return'] > 0).mean()

# プロフィットファクターの計算
total_profit = backtest_df['Strategy_Return'][backtest_df['Strategy_Return'] > 0].sum()
total_loss = abs(backtest_df['Strategy_Return'][backtest_df['Strategy_Return'] < 0].sum())
profit_factor = total_profit / total_loss

# 結果の出力
print(f"シャープレシオ: {sharpe_ratio:.2f}")
print(f"最大ドローダウン: {max_dd:.2%}")
print(f"勝率: {win_rate:.2%}")
print(f"プロフィットファクター: {profit_factor:.2f}")

# 結果のプロット
plt.figure(figsize=(12, 6))
plt.plot(backtest_df['Cumulative_Actual'], label='Buy and Hold')
plt.plot(backtest_df['Cumulative_Strategy'], label='Strategy with Stop Loss/Take Profit')
plt.legend()
plt.grid()
plt.title('Cumulative Returns: Strategy with Stop Loss/Take Profit')
plt.savefig('backtest_improved.png')
plt.close()
print("バックテスト結果を保存しました: backtest_improved.png")

# 10. メモリ解放
del X, y, scaled_data, df, backtest_df
gc.collect()
