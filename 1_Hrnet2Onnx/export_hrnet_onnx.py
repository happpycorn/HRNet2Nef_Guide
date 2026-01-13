#!/usr/bin/env python3
"""
Export HRNet (human pose estimation) PyTorch weights to ONNX.

Place this file at the ROOT of the HRNet-Human-Pose-Estimation repo (or any
folder where `lib/models/pose_hrnet.py` is importable). If your copy of the
repo doesn't include `tools/`, this script is a drop‑in replacement.

Requirements:
  pip install yacs onnx onnxsim (optional) pyyaml

Typical usage:
  python export_hrnet_pose_onnx.py \
    --weights pose_hrnet_w32_256x192.pth \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    --input_h 256 --input_w 192 \
    --opset 12 --simplify

For 384x288:
  python export_hrnet_pose_onnx.py \
    --weights pose_hrnet_w32_384x288.pth \
    --cfg experiments/coco/hrnet/w32_384x288_adam_lr1e-3.yaml \
    --input_h 384 --input_w 288 \
    --opset 12 --simplify

Output:
  Saves an ONNX file next to the .pth as <weights_basename>.onnx
  (e.g., pose_hrnet_w32_256x192.onnx)

Notes for Kneron NEF conversion:
  • Keep input size fixed (no dynamic axes) and opset 11–12.
  • Do NOT include argmax/OKS/NMS in the graph. This script exports heatmaps only.
"""
import argparse
import os
import sys
import types

import torch
import torch.onnx

# Try to support both yacs and plain yaml configs
try:
    from yacs.config import CfgNode as CN
    _HAS_YACS = True
except Exception:
    _HAS_YACS = False

try:
    import yaml
except Exception:
    yaml = None


def load_cfg(cfg_path: str):
    """Load HRNet yaml using HRNet's own config utils so MODEL.EXTRA exists."""
    if cfg_path is None:
        raise ValueError("--cfg is required (HRNet experiment yaml)")
    # Use HRNet's config loader (requires yacs)
    from lib.config import cfg as hr_cfg
    from lib.config import update_config as hr_update_config

    class _Args:
        # Mimic the minimal argparse namespace HRNet expects
        def __init__(self, cfg_path: str):
            self.cfg = cfg_path
            self.opts = []
            self.modelDir = ''
            self.logDir = ''
            self.dataDir = ''
            self.prevModelFile = ''

    args = _Args(cfg_path)
    hr_update_config(hr_cfg, args)
    return hr_cfg


def ensure_import_paths():
    """Allow importing HRNet modules such as lib.models.pose_hrnet."""
    root = os.path.abspath(os.getcwd())
    lib_dir = os.path.join(root, 'lib')
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_model(cfg):
    """Instantiate HRNet pose model via get_pose_net(cfg)."""
    from lib.models.pose_hrnet import get_pose_net  # provided by HRNet repo under lib/models
    model = get_pose_net(cfg, is_train=False)
    return model


def load_weights(model: torch.nn.Module, weights_path: str):
    ckpt = torch.load(weights_path, map_location='cpu')
    if isinstance(ckpt, dict):
        # Handle common layouts: {'state_dict': ..., 'model': ...}
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
        else:
            state = ckpt
    else:
        state = ckpt

    # Some checkpoints prefix keys with "module." or "model."; strip if needed
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        elif k.startswith('model.'):
            new_state[k[len('model.'):]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)} (showing first 10) {missing[:10]}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (showing first 10) {unexpected[:10]}")


def export_onnx(model: torch.nn.Module, h: int, w: int, out_path: str, opset: int = 12,
                dynamic: bool = False, simplify: bool = False):
    model.eval()
    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32)
    input_names = ['input']
    output_names = ['heatmap']

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'heatmap': {0: 'batch', 2: 'h_out', 3: 'w_out'},
        }

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    print(f"[ok] ONNX exported to: {out_path}")

    # Optional simplify
    if simplify:
        try:
            import onnx
            import onnxsim
            model_onnx = onnx.load(out_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            if not check:
                print("[warn] onnx-simplifier check failed; keeping original graph.")
            else:
                onnx.save(model_onnx, out_path)
                print("[ok] ONNX simplified.")
        except Exception as e:
            print(f"[warn] Simplify failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export HRNet pose .pth to .onnx')
    parser.add_argument('--weights', required=True, help='Path to HRNet pose .pth file')
    parser.add_argument('--cfg', required=True, help='Path to HRNet YAML (experiments/*/*.yaml)')
    parser.add_argument('--input_h', type=int, required=True, help='Input height (e.g., 256 or 384)')
    parser.add_argument('--input_w', type=int, required=True, help='Input width  (e.g., 192 or 288)')
    parser.add_argument('--opset', type=int, default=12)
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes (not recommended for NEF)')
    parser.add_argument('--simplify', action='store_true', help='Run onnxsim after export')
    parser.add_argument('--output', default=None, help='Output ONNX path (default: weights basename + .onnx)')
    args = parser.parse_args()

    ensure_import_paths()
    cfg = load_cfg(args.cfg)
    model = build_model(cfg)
    load_weights(model, args.weights)

    out = args.output or os.path.splitext(args.weights)[0] + '.onnx'
    export_onnx(model, args.input_h, args.input_w, out, args.opset, args.dynamic, args.simplify)

    print("Done. You can now run your ONNX → NEF conversion tool.")


if __name__ == '__main__':
    main()
