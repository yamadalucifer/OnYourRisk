import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# CSVファイルの読み込み
print("Reading and preprocessing data...")
file_path = 'USDJPY.sml_M5_201107210835_202501021155.csv'
try:
    df = pd.read_csv(file_path, sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 特徴量計算
print("Calculating features...")
df['SMA_100'] = df['Close'].rolling(window=100).mean()
df['SMA_500'] = df['Close'].rolling(window=500).mean()
df['Price_Diff'] = df['Close'].diff()
df.dropna(inplace=True)

# ターゲットの設定
future_steps = 288
threshold = 0.001
df['Target'] = 1
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] > threshold, 'Target'] = 2
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] < -threshold, 'Target'] = 0
df['Target'] = df['Target'].astype(int)
df.dropna(inplace=True)

# 特徴量とラベル
features = ['SMA_100', 'SMA_500', 'Price_Diff']
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

# クラス重みの計算
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# MLPモデル構築
def build_mlp(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

# MLPモデルのインスタンス化
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
model = build_mlp(input_dim, num_classes)

# コンパイル
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 学習
print("Training MLP...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    class_weight=class_weight_dict,
    verbose=1
)

# 評価
print("Evaluating MLP...")
y_mlp_pred = model.predict(X_test).argmax(axis=1)

# 結果の表示
mlp_accuracy = accuracy_score(y_test, y_mlp_pred)
print(f"MLP Accuracy: {mlp_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_mlp_pred, target_names=['Down', 'Flat', 'Up']))
