import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from scipy.stats import linregress
import time

# 半精度計算を有効化
set_global_policy('mixed_float16')

# GPUメモリ制限
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 特徴量計算関数
def calculate_features(df):
    start_time = time.time()

    # 既存の特徴量
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_500'] = df['Close'].rolling(window=500).mean()
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
    print(f"SMA and Bollinger Bands: {time.time() - start_time:.2f} seconds")

    # MACD
    start_time = time.time()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    print(f"MACD: {time.time() - start_time:.2f} seconds")

    # RSI
    start_time = time.time()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    print(f"RSI: {time.time() - start_time:.2f} seconds")

    # VWAP
    start_time = time.time()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_TP_Vol'] = (df['Typical_Price'] * df['Vol']).cumsum()
    df['Cumulative_Vol'] = df['Vol'].cumsum()
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Vol']
    print(f"VWAP: {time.time() - start_time:.2f} seconds")

    # Slope
    start_time = time.time()
    # Slopeの簡易的な計算
    def calculate_slope(series, window):
        return (series - series.shift(window)) / window

    df['Slope_100'] = calculate_slope(df['Close'], window=100)
    df['Slope_500'] = calculate_slope(df['Close'], window=500)
    print(f"Slope: {time.time() - start_time:.2f} seconds")

    # Volatility
    start_time = time.time()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    print(f"Volatility: {time.time() - start_time:.2f} seconds")

    # Pivot Points
    start_time = time.time()
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Support1'] = 2 * df['Pivot'] - df['High']
    df['Resistance1'] = 2 * df['Pivot'] - df['Low']
    print(f"Pivot Points: {time.time() - start_time:.2f} seconds")

    # Cumulative Return
    start_time = time.time()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cumulative_Return'] = df['Log_Return'].cumsum()
    print(f"Cumulative Return: {time.time() - start_time:.2f} seconds")

    # 微分要素の追加
    start_time = time.time()
    df['Price_Diff'] = df['Close'].diff()
    df['Price_Second_Diff'] = df['Price_Diff'].diff()
    df['SMA_100_Diff'] = df['SMA_100'].diff()
    df['SMA_500_Diff'] = df['SMA_500'].diff()
    df['MACD_Diff'] = df['MACD'].diff()
    df['RSI_Diff'] = df['RSI'].diff()
    df['Slope_100_Diff'] = df['Slope_100'].diff()
    df['Slope_500_Diff'] = df['Slope_500'].diff()
    print(f"Differentiation Features: {time.time() - start_time:.2f} seconds")

    return df

# データ読み込みと前処理
print("Reading and preprocessing data...")
file_path = 'USDJPY.sml_M5_201107210835_202501021155.csv'
try:
    df = pd.read_csv(file_path, sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
except Exception as e:
    print(f"Error reading file: {e}")

# 特徴量計算
print("Calculating features...")
df = calculate_features(df)

# ターゲットの設定
future_steps = 288
threshold = 0.001
df['Target'] = 1
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] > threshold, 'Target'] = 2
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] < -threshold, 'Target'] = 0
df['Target'] = df['Target'].astype(int)
df.dropna(inplace=True)

# 特徴量リスト
#features = ['SMA_100', 'SMA_500', 'BB_upper', 'BB_lower',
#            'MACD', 'MACD_signal', 'RSI', 'VWAP', 'Slope_100', 'Slope_500', 'Volatility_10', 'Pivot', 'Support1', 'Resistance1', 'Cumulative_Return']
features = [
    'SMA_100', 'SMA_500', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 
    'RSI', 'VWAP', 'Slope_100', 'Slope_500', 'Volatility_10', 'Pivot', 
    'Support1', 'Resistance1', 'Cumulative_Return', 
    'Price_Diff', 'Price_Second_Diff', 'SMA_100_Diff', 'SMA_500_Diff', 
    'MACD_Diff', 'RSI_Diff', 'Slope_100_Diff', 'Slope_500_Diff'
]

X = df[features].values.astype('float32')
y = df['Target'].values

# スケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# データ分割
print("Splitting data...")
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
import numpy as np

unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution in training data: {dict(zip(unique, counts))}")

# モデル構築
lookback = 288
chunk_size = 10000
batch_size = 64
from tensorflow.keras.layers import GRU

print("Building GRU model...")
#model = Sequential([
#    GRU(1000, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
#    Dropout(0.2),
#    GRU(500, return_sequences=True),
#    Dropout(0.2),
#    GRU(250, return_sequences=True),
#    Dropout(0.2),
#    GRU(100, return_sequences=False),
#    Dropout(0.2),
#    Dense(3, activation='softmax')
#])
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
    Dropout(0.2),
    GRU(25, return_sequences=False),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# トレーニング
from tensorflow.keras.callbacks import EarlyStopping

# EarlyStoppingの設定
early_stopping = EarlyStopping(
    monitor='loss',  # トレーニング損失を監視
    patience=2,      # 収束まで待つエポック数
    verbose=1
)

# トレーニング
print("Starting training...")
# chunk_generatorを再定義
def chunk_generator(X, y, chunk_size, lookback):
    total_chunks = (len(X) - lookback) // chunk_size
    if (len(X) - lookback) % chunk_size != 0:
        total_chunks += 1

    print(f"Total number of chunks: {total_chunks}")

    chunk_count = 0
    for start in range(0, len(X) - lookback, chunk_size):
        chunk_count += 1
        print(f"Processing chunk {chunk_count} of {total_chunks}...")
        end = min(start + chunk_size, len(X) - lookback)
        yield np.array([X[i:i+lookback] for i in range(start, end)]), \
              np.array([to_categorical(y[i+lookback], num_classes=3) for i in range(start, end)])

train_chunk_gen = chunk_generator(X_train, y_train, chunk_size, lookback)

for i, (X_chunk, y_chunk) in enumerate(train_chunk_gen):
    print(f"Training on chunk {i+1} with shape {X_chunk.shape}...")

    history = model.fit(
        X_chunk, y_chunk,
        batch_size=batch_size,
        epochs=50,  # 最大エポック数を指定
        verbose=1,
        shuffle=False  # シャッフルを無効にする
    )
    
    print(f"Finished training on chunk {i+1}")

    # EarlyStoppingで終了条件が満たされた場合、中断
    if early_stopping.stopped_epoch > 0:
        print("Training stopped early due to convergence.")
        break

# 評価
print("Starting evaluation...")
from sklearn.metrics import classification_report

all_y_true = []
all_y_pred = []

test_chunk_gen = chunk_generator(X_test, y_test, chunk_size, lookback)
for i, (X_chunk, y_chunk) in enumerate(test_chunk_gen):
    print(f"Evaluating chunk {i+1} with shape {X_chunk.shape}...")
    y_pred = model.predict(X_chunk, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_chunk, axis=1)

    # クラスごとの予測結果を保存
    all_y_true.extend(y_true_classes)
    all_y_pred.extend(y_pred_classes)

# チャンクごとの評価結果をまとめて表示
print("Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Down', 'Flat', 'Up']))

