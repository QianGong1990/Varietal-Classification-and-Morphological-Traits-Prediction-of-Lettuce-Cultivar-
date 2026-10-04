"""Exports SimpleCNN (rgb classification) weights as compact JSON, same format
as export_weights_json.py but for the 4-class cultivar classifier."""
import json
from pathlib import Path

import torch

from training.data import VARIETIES
from training.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = RESULTS_DIR / "web_weights"


def round_list(arr, ndigits=7):
    return [round(float(x), ndigits) for x in arr.flatten().tolist()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model("SimpleCNN", len(VARIETIES))
    model.load_state_dict(torch.load(
        RESULTS_DIR / "classification" / "model_weights" / "SimpleCNN_rgb_best.pth",
        map_location="cpu", weights_only=True))
    model.eval()
    sd = model.state_dict()

    def conv_block(prefix_conv, prefix_bn, in_ch, out_ch):
        w = sd[f"{prefix_conv}.weight"]
        b = sd[f"{prefix_conv}.bias"]
        gamma = sd[f"{prefix_bn}.weight"]
        beta = sd[f"{prefix_bn}.bias"]
        mean = sd[f"{prefix_bn}.running_mean"]
        var = sd[f"{prefix_bn}.running_var"]
        return {
            "in_ch": in_ch, "out_ch": out_ch,
            "w": round_list(w), "b": round_list(b),
            "bn_gamma": round_list(gamma), "bn_beta": round_list(beta),
            "bn_mean": round_list(mean), "bn_var": round_list(var),
            "bn_eps": 1e-5,
        }

    layers = [
        conv_block("features.0", "features.1", 3, 32),
        conv_block("features.4", "features.5", 32, 64),
        conv_block("features.8", "features.9", 64, 128),
        conv_block("features.12", "features.13", 128, 256),
    ]

    fc1_w = sd["fc1.weight"]
    fc1_b = sd["fc1.bias"]
    fc2_w = sd["fc2.weight"]
    fc2_b = sd["fc2.bias"]

    out = {
        "conv_layers": layers,
        "fc1": {"in": 256, "out": 128, "w": round_list(fc1_w), "b": round_list(fc1_b)},
        "fc2": {"in": 128, "out": 4, "w": round_list(fc2_w), "b": round_list(fc2_b)},
        "classes": VARIETIES,
    }

    out_path = OUT_DIR / "simplecnn_rgb_classification_weights.json"
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Saved {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
