import torch
import torch.nn as nn
import torchvision.models as tvm

from training.data import ARCH_RESOLUTION

ARCHITECTURES = list(ARCH_RESOLUTION.keys())
OUT_DIM = 4  # 4 varieties for classification, 4 traits (LA,D,FW,DW) for regression


class SimpleCNN(nn.Module):
    """Conv(3,32)->BN->ReLU->Pool x4 (32,64,128,256) -> GAP -> FC(256->128)->ReLU->Dropout(0.5)->FC(128->4)."""

    def __init__(self, out_dim: int = OUT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def _efficientnet_b0(out_dim: int = OUT_DIM) -> nn.Module:
    model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, out_dim))
    return model


def _mobilenet_v2(out_dim: int = OUT_DIM) -> nn.Module:
    model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, out_dim))
    return model


def _timm_head_model(model_name: str, expected_features: int, out_dim: int = OUT_DIM) -> nn.Module:
    import timm
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # pooled features
    in_features = backbone.num_features
    assert in_features == expected_features, f"{model_name}: expected {expected_features}, got {in_features}"
    head = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, out_dim))
    return nn.Sequential(backbone, head)


def _xception(out_dim: int = OUT_DIM) -> nn.Module:
    return _timm_head_model("xception", 2048, out_dim)


def _inception_resnet_v2(out_dim: int = OUT_DIM) -> nn.Module:
    return _timm_head_model("inception_resnet_v2", 1536, out_dim)


BUILDERS = {
    "SimpleCNN": SimpleCNN,
    "EfficientNetB0": _efficientnet_b0,
    "MobileNetV2": _mobilenet_v2,
    "Xception": _xception,
    "InceptionResNetV2": _inception_resnet_v2,
}


def build_model(arch: str, out_dim: int = OUT_DIM) -> nn.Module:
    if arch not in BUILDERS:
        raise ValueError(f"Unknown architecture: {arch}")
    return BUILDERS[arch](out_dim)
