import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator

print("Starting script...")

# **1. データ準備用ジェネレーター**
def data_generator(X, y, batch_size):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[start:end], y[start:end]

# **2. データの読み込みと前処理**
print("Reading and processing data...")
file_path = 'USDJPY.sml_M5_201107210835_202501021155.csv'

df = pd.read_csv(file_path, sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)

# 特徴量の計算
print("Calculating features...")
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = (df['Close'].diff() > 0).rolling(window=14).mean()
df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()
df['BB_Width'] = df['BB_upper'] - df['BB_lower']
adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
df['ADX'] = adx.adx()
df['SMA_Diff'] = df['SMA_10'] - df['SMA_50']
df['High_Low_Spread'] = df['High'] - df['Low']
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
df['Momentum_50'] = df['Close'] - df['Close'].shift(50)

# ターゲットの設定
print("Setting target...")
future_steps = 288  # 288本未来を予測
df['Target'] = (df['Close'].shift(-future_steps) - df['Close']) > 0  # 上昇=1, 下降=0
df['Target'] = df['Target'].astype(int)

# 欠損値の削除
df.dropna(inplace=True)
print(f"Data prepared. Data shape: {df.shape}")

# 特徴量リスト
features = ['SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR', 'BB_Width',
            'ADX', 'SMA_Diff', 'High_Low_Spread', 'Momentum_10', 'Momentum_50']
X = df[features].values
y = df['Target'].values

# スケーリング
print("Scaling data...")
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# 時系列用のデータ準備
print("Preparing LSTM input...")
lookback = 288  # 過去288本を入力として使用
X_lstm, y_lstm = [], []
for i in range(len(X) - lookback):
    X_lstm.append(X[i:i+lookback])
    y_lstm.append(y[i+lookback])

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# データ分割
print("Splitting data...")
split_index = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_index], X_lstm[split_index:]
y_train, y_test = y_lstm[:split_index], y_lstm[split_index:]

print(f"Data split completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# **3. LSTMモデル構築**
print("Building LSTM model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # 2値分類
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("LSTM model built.")

# **4. 学習用ジェネレーターの使用**
batch_size = 64
train_generator = data_generator(X_train, y_train, batch_size)
val_generator = data_generator(X_test, y_test, batch_size)

print("Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    epochs=20,
    validation_data=val_generator,
    validation_steps=len(X_test) // batch_size
)
print("Model training completed.")

# **5. モデル評価**
print("Evaluating model...")
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary, target_names=['Down', 'Up']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))
