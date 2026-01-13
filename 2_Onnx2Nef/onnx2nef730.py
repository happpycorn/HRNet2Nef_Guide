#!/usr/bin/env python3
"""
HRNet Pose: ONNX → (opt) → BIE → NEF 一鍵流程（自動偵測輸入尺寸）
- 參考你現有的 STDC 腳本流程並改為 HRNet Pose 版本：優化 → 效能評估 → 量化(BIE) → 編譯NEF。
- 只需要一個 HRNet Pose 的 ONNX 檔（輸出為 heatmap）。

使用方式（範例，一行即可）：
  python onnx2nef_hrnet_pose_auto.py --onnx /docker_mount/pose_hrnet_w32_256x192.onnx --chip 730 --out_dir /docker_mount/work_dirs/hrnet_pose --images /docker_mount/calib_images

必要條件：
  pip install onnx numpy pillow ktc  #（ktc 已在 toolchain docker 內）

注意：
  • 會自動從 ONNX 讀取 input 名稱與 shape（NCHW）。
  • 量化校正的影像會自動 resize 到網路輸入長寬，做簡單 /255.0 正規化（HRNet 常見設定）。
  • 後處理（argmax/解碼）請放在裝置端 CPU；NEF 只輸出 heatmap。
"""
import argparse
import os
import shutil
import onnx
import numpy as np
from PIL import Image
import ktc


def read_onnx_shape(onnx_model):
    inp = onnx_model.graph.input[0]
    name = inp.name
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if len(dims) != 4:
        raise ValueError(f"Unexpected input dims: {dims}, expect NCHW")
    n, c, h, w = dims
    if n == 0:  # dynamic batch -> force 1
        n = 1
    if c != 3:
        raise ValueError(f"Expect 3-channel input, got C={c}")
    return name, (n, c, h, w)


def load_and_preprocess_images(img_dir, size_hw, limit=None):
    H, W = size_hw
    imgs = []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for root, _, fs in os.walk(img_dir):
        for f in fs:
            if f.lower().endswith(exts):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No images found under {img_dir}")
    files.sort()
    if limit:
        files = files[:limit]
    for p in files:
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize((W, H), Image.BILINEAR)
            arr = np.array(im).astype(np.float32) / 255.0  # HRNet 常用 /255
            arr = np.transpose(arr, (2, 0, 1))  # HWC->CHW
            arr = np.expand_dims(arr, 0)        # CHW->NCHW
            imgs.append(arr)
            print(f"  [+] calib: {p}")
        except Exception as e:
            print(f"  [!] skip {p}: {e}")
    if not imgs:
        raise RuntimeError("No valid images after preprocessing.")
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True, help='Path to HRNet Pose ONNX')
    ap.add_argument('--chip', required=True, help='Kneron chip: 520/630/730/830 ...')
    ap.add_argument('--images', required=True, help='Folder of calibration images')
    ap.add_argument('--out_dir', default='./hrnet_pose_out', help='Folder to save optimized ONNX/BIE/NEF')
    ap.add_argument('--model_id', type=int, default=32769, help='Model ID (decimal)')
    ap.add_argument('--version', default='0001', help='Model version string')
    ap.add_argument('--calib_count', type=int, default=50, help='Number of images for calibration (cap)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] Load ONNX")
    m = onnx.load(args.onnx)
    input_name, (n, c, h, w) = read_onnx_shape(m)
    print(f"    input_name={input_name}, shape=({n},{c},{h},{w})")

    print("[2/6] Optimize ONNX via onnx2onnx_flow")
    m_opt = ktc.onnx_optimizer.onnx2onnx_flow(m)
    opt_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.onnx))[0] + '_opt.onnx')
    onnx.save(m_opt, opt_path)
    print(f"    saved: {opt_path}")

    print("[3/6] Build ModelConfig & Evaluate")
    km = ktc.ModelConfig(args.model_id, args.version, str(args.chip), onnx_model=m_opt)
    try:
        eval_res = km.evaluate()
        print("\n[Eval] NPU estimation:\n" + str(eval_res))
    except Exception as e:
        print(f"[warn] evaluate() failed: {e}")

    print("[4/6] Prepare calibration images")
    imgs = load_and_preprocess_images(args.images, (h, w), limit=args.calib_count)

    print("[5/6] Quantization (BIE)")
    try:
        bie_path = km.analysis({input_name: imgs})
    except Exception as e:
        # 某些版本使用 km.quantize(imgs)；若 analysis 失敗，回退。
        print(f"[warn] analysis() failed: {e}; try quantize(imgs) fallback")
        bie_path = km.quantize(imgs)
    bie_copy = os.path.join(args.out_dir, os.path.basename(bie_path))
    try:
        shutil.copy(bie_path, bie_copy)
    except Exception:
        pass
    print(f"    BIE saved: {bie_path if os.path.exists(bie_path) else bie_copy}")

    print("[6/6] Compile NEF")
    nef_path = ktc.compile([km])
    nef_copy = os.path.join(args.out_dir, os.path.basename(nef_path))
    try:
        shutil.copy(nef_path, nef_copy)
    except Exception:
        pass
    print(f"    NEF saved: {nef_path if os.path.exists(nef_path) else nef_copy}")
    print("\n✅ Done.")


if __name__ == '__main__':
    main()
