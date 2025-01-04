import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dense, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random

# 全体のランダムシードを固定
SEED = 42
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# TensorFlowのログレベル設定
tf.debugging.set_log_device_placement(True)

# MirroredStrategyを使用して両方のGPUを活用
strategy = tf.distribute.MirroredStrategy()

# GPUメモリの制限（仮想デバイス設定の変更）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB に制限
            )
    except RuntimeError as e:
        print("Virtual devices already initialized, skipping reconfiguration.")

# CSVファイルの読み込み
print("Reading and preprocessing data...")
file_path = 'USDJPY.sml_M5_201107210835_202501021155.csv'
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

try:
    df = pd.read_csv(file_path, sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 特徴量計算
print("Calculating features...")
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['MA_Gap_Short_Mid'] = df['SMA_10'] - df['SMA_50']
df['MA_Gap_Mid_Long'] = df['SMA_50'] - df['SMA_200']
df['Price_Diff'] = df['Close'].diff()

# 無限大やNaNを削除
print("Checking for infinite or NaN values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ターゲットの設定
future_steps = 12
threshold = 0.0002
df['Target'] = 1
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] > threshold, 'Target'] = 2
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] < -threshold, 'Target'] = 0
df['Target'] = df['Target'].astype(int)
df.dropna(inplace=True)

# 特徴量とラベル
X = df[['SMA_10', 'SMA_50', 'SMA_200', 'MA_Gap_Short_Mid', 'MA_Gap_Mid_Long', 'Price_Diff']].values.astype('float32')
y = df['Target'].values.astype('int32')

# スケーリングの追加
print("Scaling features...")
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# データ整形
lookback = 360
X_reshaped = np.array([X[i:i+lookback] for i in range(len(X) - lookback)]).astype('float32')
y_reshaped = y[lookback:].astype('int32')

# データ分割
print("Splitting data...")
split_index = int(len(X_reshaped) * 0.8)
X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
y_train, y_test = y_reshaped[:split_index], y_reshaped[split_index:]

# デバッグ用のデータ型と形状を出力
print("X_train shape:", X_train.shape, "X_train dtype:", X_train.dtype)
print("y_train shape:", y_train.shape, "y_train dtype:", y_train.dtype)

# MirroredStrategy スコープ内でのモデル学習
with strategy.scope():
    num_gpus = strategy.num_replicas_in_sync
    adjusted_batch_size = 4 * num_gpus  # バッチサイズを調整
    print(f"Using {num_gpus} GPUs. Adjusted batch size: {adjusted_batch_size}")

    # Transformerモデル構築
    def custom_dropout(x, rate=0.3, training=True):
        if training:
            return Lambda(lambda x: tf.nn.dropout(x, rate))(x)
        return x

    def build_transformer(input_shape, num_classes):
        inputs = Input(shape=input_shape)
        attention_output = MultiHeadAttention(num_heads=16, key_dim=64)(inputs, inputs)
        attention_output = LayerNormalization()(attention_output + inputs)
        dense_output = Dense(512, activation='relu')(attention_output)
        dense_output = custom_dropout(dense_output)  # カスタムDropoutを使用
        flat_output = Flatten()(dense_output)
        outputs = Dense(num_classes, activation='softmax')(flat_output)
        return Model(inputs=inputs, outputs=outputs)

    model = build_transformer((lookback, X.shape[1]), len(np.unique(y)))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # tf.data.Datasetの作成（修正版）
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)

    train_dataset = tf.data.Dataset.from_generator(
        lambda: ((X_train[i], y_train[i]) for i in range(len(X_train))),
        output_signature=(
            tf.TensorSpec(shape=X_train.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    train_dataset = train_dataset.batch(adjusted_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # 同様にテストデータセットを作成
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: ((X_test[i], y_test[i]) for i in range(len(X_test))),
        output_signature=(
            tf.TensorSpec(shape=X_test.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    test_dataset = test_dataset.batch(adjusted_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # モデルの学習
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=50,
        verbose=1
    )

# 評価
y_pred = model.predict(test_dataset).argmax(axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Down', 'Flat', 'Up']))
