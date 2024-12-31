import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from ta.volatility import AverageTrueRange

# データの読み込みと前処理
df = pd.read_csv('5min_data.csv', sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['RSI'] = (df['Close'].diff() > 0).rolling(window=14).mean()
df['Price_Change'] = df['Close'].pct_change()
df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['ATR'] = atr.average_true_range()

future_steps = 288
df['Target'] = (df['Close'].shift(-future_steps) > df['Close']).astype(int)
df.dropna(inplace=True)

# 特徴量リスト
features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'BB_upper', 'BB_lower', 'ATR']
X = df[features]
y = df['Target']

# 特徴量名を変更
X.columns = [f'f{i}' for i in range(X.shape[1])]

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost用のDMatrixを作成
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

# モデルの学習
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.01,
    'max_depth': 20,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'n_estimators': 10000
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dtest, 'eval')],
    early_stopping_rounds=10
)

# モデル保存
model.save_model('xgboost_model_fixed.json')
print("Model with fixed feature names saved as 'xgboost_model_fixed.json'")

# y_predの予測とバイナリ化
y_pred = model.predict(dtest)  # XGBoostモデルの予測
y_pred_binary = (y_pred > 0.5).astype(int)  # 閾値を適用してバイナリ化

# 予測結果の確認
print(f"y_test type: {type(y_test)}, shape: {y_test.shape}")
print(f"y_pred_binary type: {type(y_pred_binary)}, shape: {y_pred_binary.shape}")

# 分類レポートの生成と出力
from sklearn.metrics import classification_report

classification_report_output = classification_report(
    y_test,
    y_pred_binary,
    target_names=['Down', 'Up'],
    zero_division=0
)

from sklearn.metrics import classification_report

# 分類レポートの生成
classification_report_output = classification_report(
    y_test,
    y_pred_binary,
    target_names=['Down', 'Up'],
    zero_division=0  # ゼロ割り防止
)

# 分類レポートを表示と保存
print("\nClassification Report:")
print(classification_report_output)  # ターミナルに表示
with open("classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report_output)

print("Classification report saved as 'classification_report.txt'.")

# 学習済みモデルの保存
model.save_model('xgboost_model_fixed.json')
print("Model saved as 'xgboost_model_fixed.json'.")
