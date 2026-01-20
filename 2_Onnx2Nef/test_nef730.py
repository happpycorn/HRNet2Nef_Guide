import os
import numpy as np
from PIL import Image
import ktc
import argparse

def preprocess_for_ktc(image_path, in_h=256, in_w=192):
    """將影像讀入並處理成 KTC 模擬器需要的 NCHW 格式 (1, 3, 256, 192)"""
    img = Image.open(image_path).convert('RGB')
    # Resize 使用 BILINEAR 對齊常見的預處理邏輯
    img_resized = img.resize((in_w, in_h), Image.BILINEAR)
    img_data = np.array(img_resized).astype(np.float32)

    img_data /= 255.0
    
    # 增加 Batch 維度並轉置維度: (H, W, C) -> (1, C, H, W)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = img_data.transpose(0, 3, 1, 2)
    
    return img_data

def main():
    parser = argparse.ArgumentParser(description='Save KTC NEF Inference Result to NPY')
    parser.add_argument('-m', '--model', default='/output/models_730.nef', help='Path to NEF model')
    parser.add_argument('-img', '--image', default='/input/test.jpg', help='Path to input image')
    parser.add_argument('-o', '--output', default='/output/nef_sim_output.npy', help='Output .npy path')
    args = parser.parse_args()

    # 1. 影像預處理
    print(f"📸 Loading and preprocessing: {args.image}")
    input_tensor = preprocess_for_ktc(args.image)

    # 2. 執行 KTC 模擬器推論
    print(f"🧠 Running KTC E2E Simulation (Platform: 730)...")
    try:
        # 使用 v0.31.1 的 kneron_inference API
        results = ktc.kneron_inference(
            [input_tensor],
            nef_file=args.model,
            platform=730,
            input_names=['input'] # 需與 ONNX 匯出時的名稱對齊
        )
        
        # 3. 儲存結果
        # results[0] 通常是 HRNet 的 Heatmap，形狀約為 (1, 17, 64, 48)
        heatmap = results[0]
        np.save(args.output, heatmap)
        
        print(f"✅ Success! Heatmap shape: {heatmap.shape}")
        print(f"💾 Data saved to: {args.output}")

    except Exception as e:
        print(f"❌ Inference failed: {e}")

if __name__ == '__main__':
    main()