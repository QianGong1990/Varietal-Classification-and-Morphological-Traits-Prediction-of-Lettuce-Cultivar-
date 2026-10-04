"""Grad-CAM: architecture-agnostic implementation.

Automatically locates the last spatial (4D, H>1,W>1) feature map produced
during a forward pass, so it works uniformly across SimpleCNN, torchvision
models (EfficientNetB0, MobileNetV2), and timm-wrapped models (Xception,
InceptionResNetV2) without hardcoding per-architecture layer names.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from training.data import (
    ARCH_RESOLUTION, IMAGENET_MEAN, IMAGENET_STD, REGRESSION_TARGETS, TRAIT_DISPLAY, VARIETIES,
)
from training.gpu_utils import get_device
from training.models import build_model


def find_last_spatial_module_name(model, input_tensor):
    """Runs one forward pass, records every leaf module's output, and returns
    the name of the last one that produced a spatial (B,C,H>1,W>1) tensor."""
    shapes = {}
    hooks = []

    def make_hook(name):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[-1] > 1 and out.shape[-2] > 1:
                shapes[name] = True
        return fn

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    for h in hooks:
        h.remove()
    if not shapes:
        raise RuntimeError("No spatial (4D) feature map found in this model.")
    return list(shapes.keys())[-1]


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model
        self.activations = None
        self.gradients = None
        # Hook the parent container of the detected leaf module rather than the
        # leaf itself: leaves are often in-place activations (nn.ReLU(inplace=True)),
        # and register_full_backward_hook on an in-place op raises a RuntimeError
        # ("view is being modified inplace") in recent PyTorch. The parent's forward
        # hook observes the identical output tensor without this restriction.
        modules = dict(model.named_modules())
        hook_name = target_layer_name
        if "." in target_layer_name:
            parent_name = target_layer_name.rsplit(".", 1)[0]
            if parent_name in modules:
                hook_name = parent_name
        target_module = modules[hook_name]
        target_module.register_forward_hook(self._save_activation)
        target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, x: torch.Tensor, target_index: int) -> np.ndarray:
        """x: (1,C,H,W). Returns a (H,W) numpy heatmap in [0,1], resized to input resolution."""
        self.model.zero_grad()
        out = self.model(x)
        score = out[0, target_index]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, out.detach().cpu().numpy()[0]


def load_image_tensor(path: str, resolution: int, pixel_norm: dict | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    if pixel_norm is not None:
        mean = torch.tensor(pixel_norm["mean"], dtype=torch.float32)
        std = torch.tensor(pixel_norm["std"], dtype=torch.float32)
        tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    else:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32)
        tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    return tensor.unsqueeze(0), img


def overlay_heatmap(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[..., :3]  # RGB in [0,1]
    base = np.asarray(pil_img, dtype=np.float32) / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap
    return np.clip(overlay, 0, 1)


def run_gradcam_grid(arch: str, input_type: str, task: str, image_paths: list, titles: list,
                      target_index, target_label: str, out_path: Path,
                      weight_path: Path, pixel_norm: dict | None = None):
    """target_index: either a single int (applied to every image) or a list
    of one int per image (e.g. that image's predicted class)."""
    device = get_device()
    resolution = ARCH_RESOLUTION[arch]
    out_dim = len(VARIETIES) if task == "classification" else len(REGRESSION_TARGETS)

    model = build_model(arch, out_dim).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, resolution, resolution).to(device)
    layer_name = find_last_spatial_module_name(model, dummy)

    cam_engine = GradCAM(model, layer_name)

    n = len(image_paths)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.0),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    if n == 1:
        axes = axes.reshape(2, 1)

    target_indices = target_index if isinstance(target_index, list) else [target_index] * n

    for i, (path, title) in enumerate(zip(image_paths, titles)):
        x, pil_img = load_image_tensor(path, resolution, pixel_norm)
        x = x.to(device)
        cam, raw_out = cam_engine.compute(x, target_indices[i])
        overlay = overlay_heatmap(pil_img, cam)

        axes[0, i].imshow(pil_img)
        axes[0, i].set_title(title, fontsize=17, pad=6)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")

    target_label_display = TRAIT_DISPLAY.get(target_label, target_label)
    fig.suptitle(f"Grad-CAM: {arch} ({input_type}) — {target_label_display} (layer: {layer_name})",
                 fontsize=20, y=0.985)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(out_path.with_suffix(suffix), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path.with_suffix('.png')} and .svg  (layer used: {layer_name})")
