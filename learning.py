import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
from ta.volatility import AverageTrueRange

# データの読み込みと前処理
df = pd.read_csv('USDJPY.sml_M5_202001020000_202412312355.csv', sep='\t', header=0, names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])

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
df['BB_Width'] = df['BB_upper'] - df['BB_lower']
from ta.trend import ADXIndicator
adx = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
df['ADX'] = adx.adx()

threshold = 0.001  # 横ばいのしきい値（0.1%）
future_steps = 288
df['Price_Change'] = (df['Close'].shift(-future_steps) - df['Close']) / df['Close']

# ターゲット定義
def classify_target(change, threshold):
    if change > threshold:
        return 2  # 上昇
    elif change < -threshold:
        return 0  # 下降
    else:
        return 1  # 横ばい

df['Target'] = df['Price_Change'].apply(lambda x: classify_target(x, threshold))
df.dropna(inplace=True)

# 特徴量リスト
#features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'BB_upper', 'BB_lower', 'ATR']
#features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR']
features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR','BB_Width','ADX']
X = df[features]
y = df['Target']

# 特徴量名を変更
X.columns = [f'f{i}' for i in range(X.shape[1])]

# データ分割（トレーニング、検証、テスト）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 検証+テストデータを30%
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 検証とテストを50:50に分割

print(f"Training data size: {X_train.shape}, Validation data size: {X_val.shape}, Test data size: {X_test.shape}")

# XGBoost用のDMatrixを作成
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_test.columns.tolist())

# モデルの学習
params = {
    'objective': 'multi:softmax',  # 多クラス分類
    'num_class': 3,  # クラス数
    'eval_metric': 'mlogloss',  # 多クラスのログ損失
    'learning_rate': 0.01,
    'max_depth': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'n_estimators': 10000,
    'tree_method': 'hist',  # GPU非推奨の代替
    'device': 'cuda'  # GPUを使用
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=10000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50
)


# モデル保存
model.save_model('xgboost_model_fixed.json')
print("Model with fixed feature names saved as 'xgboost_model_fixed.json'")

# テストデータでの予測
y_pred = model.predict(dtest)

# 予測結果の確認
print(f"y_test type: {type(y_test)}, shape: {y_test.shape}")
print(f"y_pred type: {type(y_pred)}, shape: {y_pred.shape}")

# 分類レポートの生成と出力
classification_report_output = classification_report(
    y_test,
    y_pred,
    target_names=['Down', 'Sideways', 'Up'],
    zero_division=0  # ゼロ割り防止
)

print("\nClassification Report:")
print(classification_report_output)
with open("classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(classification_report_output)

print("Classification report saved as 'classification_report.txt'.")
