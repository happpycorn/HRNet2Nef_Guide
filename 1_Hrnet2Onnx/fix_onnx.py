import onnx

# 讀取你剛產出的模型
# 讀取剛產出的模型，注意容器內的路徑
model_path = "/output/pose_hrnet_w32_256x192.onnx" 
# 儲存修復後的模型，也放在 output 方便管理
save_path = "../output/pose_hrnet_fix.onnx"
model = onnx.load(model_path)

# 強制將 IR 版本降到 6 或 7 (這通常對應 Opset 11/12)
model.ir_version = 7 

# 儲存修復後的模型
onnx.save(model, save_path)
print("✅ 模型 IR 版本已強制修改為 7 (相容模式)")