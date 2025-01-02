import onnxruntime as ort
import numpy as np
import onnx

# ONNXモデルの読み込み
session = ort.InferenceSession("xgboost_model.onnx")

# 入力情報を確認
input_name = session.get_inputs()[0].name
print("Input name:", input_name)
print("Input shape:", session.get_inputs()[0].shape)

# 出力情報を確認
output_name = session.get_outputs()[0].name
print("Output name:", output_name)
print("Output shape:", session.get_outputs()[0].shape)

# ダミーデータで推論
input_data = np.random.rand(1, 9).astype(np.float32)  # 特徴量数を8に設定
outputs = session.run([output_name], {input_name: input_data})
print("Outputs:", outputs)

model = onnx.load("xgboost_model.onnx")
for output in model.graph.output:
    print(f"Output name: {output.name}")
    print(f"Output shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")