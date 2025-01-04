import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
import time

# データ読み込みと前処理
print("Reading and preprocessing data...")
file_path = 'USDJPY.sml_M5_201107210835_202501021155.csv'
try:
    df = pd.read_csv(file_path, sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
except Exception as e:
    print(f"Error reading file: {e}")

# 特徴量計算（簡略化バージョン）
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

# 基準モデル1: 多数派クラス予測
print("Evaluating majority class baseline...")
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)
y_dummy_pred = dummy_model.predict(X_test)
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
print(f"Majority Class Baseline Accuracy: {dummy_accuracy:.4f}")
print(classification_report(y_test, y_dummy_pred, target_names=['Down', 'Flat', 'Up']))

# 基準モデル2: ロジスティック回帰
print("Evaluating logistic regression baseline...")
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
y_logistic_pred = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_logistic_pred)
print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
print(classification_report(y_test, y_logistic_pred, target_names=['Down', 'Flat', 'Up']))
