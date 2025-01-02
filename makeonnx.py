import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
from xgboost import Booster

# 学習済みモデルのロード
model = Booster()
model.load_model('xgboost_model_fixed.json')

# 入力データ型を定義
#features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'Price_Change', 'BB_upper', 'BB_lower', 'ATR']
#features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR','BB_Width','ADX']
#features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR','BB_Width','ADX','High_Low_Spread','Momentum_10','Momentum_50']
features = ['Close', 'SMA_10', 'SMA_50', 'RSI', 'BB_upper', 'BB_lower', 'ATR','BB_Width','ADX','High_Low_Spread','Momentum_10','Momentum_50','Momentum_100','Momentum_200','Momentum_300']
input_dim = len(features)
initial_type = [('float_input', FloatTensorType([None, input_dim]))]

# ONNXに変換
onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)

# ONNXモデルを保存
with open("xgboost_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model converted to ONNX format and saved as 'xgboost_model.onnx'.")
