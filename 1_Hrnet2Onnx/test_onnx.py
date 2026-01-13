import onnxruntime as ort
import numpy as np
from PIL import Image

# === 1. 設定路徑 ===
ONNX_PATH = "/output/pose_hrnet_fix.onnx"  # 請改成你的 ONNX 檔名
IMG_PATH = "/input/test.jpg"
OUTPUT_NPY_NAME = "/output/onnx_output_heatmap.npy"

# === 2. 準備標準 HRNet 預處理函數 (需與 KTC 模擬器一致) ===
def preprocess_image(image_path, target_size=(192, 256)):
    # target_size = (Width, Height) -> 配合 HRNet 常見比例
    print(f"預處理: 開啟圖片並調整大小至 {target_size}...")
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size, Image.BILINEAR)
    
    # 轉為 Numpy float32
    img_np = np.array(img_resized).astype(np.float32)
    
    # 正規化 (0-1) - 必須與 KTC 模擬時的處理一致
    img_np /= 255.0
    
    # HWC 轉 CHW
    img_input = img_np.transpose(2, 0, 1)
    
    # 增加 Batch 維度 -> (1, 3, 256, 192)
    img_input = np.expand_dims(img_input, axis=0)
    return img_input

# === 3. 主程式 ===
if __name__ == "__main__":
    # --- 載入模型 ---
    print(f"正在載入 ONNX 模型: {ONNX_PATH}...")
    # 使用 CPU 執行 (相容性最好)
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    # 自動取得模型輸入和輸出的節點名稱
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"模型輸入名稱: {input_name}, 預期形狀: {input_shape}")

    # --- 處理圖片 ---
    # 根據模型輸入形狀動態調整 resize 目標 (假設是 NCHW)
    target_w = input_shape[3] if isinstance(input_shape[3], int) else 192
    target_h = input_shape[2] if isinstance(input_shape[2], int) else 256
    img_tensor = preprocess_image(IMG_PATH, target_size=(target_w, target_h))

    # --- 執行推論 ---
    print("開始 ONNX Runtime 推論...")
    # run 的第一個參數是輸出名稱列表 (None 代表取得所有輸出)
    outputs = ort_session.run([output_name], {input_name: img_tensor})
    
    # ONNX 的輸出是一個 list，我們取第一個
    heatmap_onnx = outputs[0]
    
    print(f"✅ 推論完成！輸出形狀: {heatmap_onnx.shape}")
    
    # --- 存檔 ---
    np.save(OUTPUT_NPY_NAME, heatmap_onnx)
    print(f"基準結果已存至 {OUTPUT_NPY_NAME}")
    
    # --- (選擇性) 簡單驗證 ---
    # 如果你已經有 Kneron 跑出來的 npy，可以取消註解下面這行來立刻對比
    # try:
    #     ktc_res = np.load("output_heatmap.npy")
    #     # 確保兩個形狀一樣才能比，如果 KTC 是 (1,17,64,48)，ONNX 也應該要是
    #     if ktc_res.shape == heatmap_onnx.shape:
    #         diff = np.abs(ktc_res - heatmap_onnx).mean()
    #         print(f"\n[快速對比] ONNX 與 KTC 結果的平均絕對誤差 (MAE): {diff:.6f}")
    #         if diff < 0.01: print("--> 結果非常接近！轉換成功！")
    #         else: print("--> 存在差異，可能預處理或量化造成損失。")
    # except FileNotFoundError:
    #     pass