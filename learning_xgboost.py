import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import DMatrix, train

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
df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
df['BB_Width'] = df['BB_upper'] - df['BB_lower']  # ボリンジャーバンド幅
df['Std_Dev'] = df['Close'].rolling(window=20).std()  # 標準偏差

df['Vol_MA_10'] = df['Vol'].rolling(window=10).mean()  # ボリューム移動平均
df['Vol_MA_50'] = df['Vol'].rolling(window=50).mean()

df['ROC_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)  # 価格変化率（10期間）
df['ROC_50'] = (df['Close'] - df['Close'].shift(50)) / df['Close'].shift(50)
df['Volume_ROC_10'] = (df['Vol'] - df['Vol'].shift(10)) / df['Vol'].shift(10)  # ボリューム変化率

# MACD計算
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# RSI計算
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss))

# ATR計算
df['High_Low'] = df['High'] - df['Low']
df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()

# VWAP計算
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['Cumulative_TP_Vol'] = (df['Typical_Price'] * df['Vol']).cumsum()
df['Cumulative_Vol'] = df['Vol'].cumsum()
df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Vol']

df['Close_to_SMA_10'] = df['Close'] / df['SMA_10']
df['Close_to_SMA_50'] = df['Close'] / df['SMA_50']

df['BB_Width_Ratio'] = df['BB_Width'] / df['Close']

df['Acceleration'] = df['Close'].diff().diff()


# 差分特徴量を追加
print("Calculating difference features...")
original_features = ['SMA_10', 'SMA_50', 'SMA_200', 'MA_Gap_Short_Mid', 'MA_Gap_Mid_Long', 
                     'Price_Diff', 'BB_upper', 'BB_lower', 'BB_Width', 'Std_Dev',
                     'Vol_MA_10', 'Vol_MA_50', 'ROC_10', 'ROC_50', 'Volume_ROC_10', 
                     'MACD', 'MACD_signal', 'RSI', 'ATR', 'VWAP','Close_to_SMA_10','Close_to_SMA_50','BB_Width_Ratio','Acceleration']
shift_value = 288  # 1日前（288本前）との差分
for feature in original_features:
    diff_feature_name = f"{feature}_Diff"
    df[diff_feature_name] = df[feature] - df[feature].shift(shift_value)

# 無限大やNaNを削除
print("Checking for infinite or NaN values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ターゲットの設定
#future_steps = 288
future_steps = 12
threshold = 0.0002
df['Target'] = 1
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] > threshold, 'Target'] = 2
df.loc[(df['Close'].shift(-future_steps) - df['Close']) / df['Close'] < -threshold, 'Target'] = 0
df['Target'] = df['Target'].astype(int)
df.dropna(inplace=True)

# 特徴量とラベル
features = original_features + [f"{feature}_Diff" for feature in original_features]
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

# データ再サンプリング
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"Resampled Data Size: {X_resampled.shape}")

# XGBoost用のデータセット
dtrain = DMatrix(X_resampled, label=y_resampled)
dtest = DMatrix(X_test, label=y_test)

# XGBoostモデル構築（GPU対応）
print("Training XGBoost on GPU...")
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 25,
    'eta': 0.05,
    'eval_metric': 'merror',
    'tree_method': 'gpu_hist',
    'seed': 42
}

# モデルの学習
evals = [(dtest, 'validation')]
bst = train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1500,
    evals=evals,
    verbose_eval=True
)

# テストデータでの予測
print("Evaluating XGBoost...")
y_xgb_pred = bst.predict(dtest)

# 評価結果の出力
xgb_accuracy = accuracy_score(y_test, y_xgb_pred)
print(f"XGBoost Final Accuracy: {xgb_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_xgb_pred, target_names=['Down', 'Flat', 'Up']))
