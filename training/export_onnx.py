"""Exports SimpleCNN (rgb regression) to ONNX for in-browser inference,
and verifies the exported graph produces identical outputs to the PyTorch
model on a real test image before it's trusted for deployment.
"""
import json
from pathlib import Path

import numpy as np
import torch

from training.data import ARCH_RESOLUTION, REGRESSION_TARGETS, DATASET_ROOT
from training.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "onnx"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arch, input_type = "SimpleCNN", "rgb"
    resolution = ARCH_RESOLUTION[arch]

    model = build_model(arch, len(REGRESSION_TARGETS))
    model.load_state_dict(torch.load(
        RESULTS_DIR / "regression" / "model_weights" / f"{arch}_{input_type}_best.pth",
        map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, resolution, resolution)
    onnx_path = OUT_DIR / "simplecnn_rgb_regression.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes=None,
    )
    print(f"Exported ONNX to {onnx_path} ({onnx_path.stat().st_size/1024:.1f} KB)")

    # ---- verify against a real test image ----
    import onnxruntime as ort
    from PIL import Image

    with open(RESULTS_DIR / "normalization_params.json") as f:
        norm_params = json.load(f)
    pixel_norm = norm_params["pixel_norm"][f"{input_type}_{resolution}"]
    target_norm = norm_params["targets"]

    test_img_path = DATASET_ROOT / f"RGBImages_{resolution}" / "RGB_4.png"
    img = Image.open(test_img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(pixel_norm["mean"], dtype=torch.float32)
    std = torch.tensor(pixel_norm["std"], dtype=torch.float32)
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    x = tensor.unsqueeze(0)

    with torch.no_grad():
        torch_out = model(x).numpy()[0]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x.numpy()})[0][0]

    print("\nPyTorch vs ONNX raw (normalized) outputs:")
    for i, trait in enumerate(REGRESSION_TARGETS):
        print(f"  {trait}: torch={torch_out[i]:.6f}  onnx={onnx_out[i]:.6f}  diff={abs(torch_out[i]-onnx_out[i]):.2e}")

    max_diff = np.abs(torch_out - onnx_out).max()
    assert max_diff < 1e-4, f"ONNX output diverges from PyTorch by {max_diff}"
    print(f"\nMax diff: {max_diff:.2e} -- ONNX export verified correct.")

    print("\nDenormalized (physical units) for RGB_4.png:")
    for i, trait in enumerate(REGRESSION_TARGETS):
        val = onnx_out[i] * target_norm[trait]["std"] + target_norm[trait]["mean"]
        print(f"  {trait}: {val:.2f}")


if __name__ == "__main__":
    main()
