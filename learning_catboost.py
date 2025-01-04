import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

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

# CatBoostモデルの構築
print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1000,              # 最大反復数
    learning_rate=0.1,            # 学習率
    depth=6,                      # ツリーの深さ
    loss_function='MultiClass',   # 多クラス分類
    class_weights=[0.88, 1.59, 0.81],  # クラスの重み
    random_seed=42,               # 再現性のための乱数シード
    verbose=100                   # 進捗の表示間隔
)

# モデルの学習
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# テストデータでの予測
print("Evaluating CatBoost...")
y_cat_pred = cat_model.predict(X_test)

# 評価結果の出力
cat_accuracy = accuracy_score(y_test, y_cat_pred)
print(f"CatBoost Accuracy: {cat_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_cat_pred, target_names=['Down', 'Flat', 'Up']))
