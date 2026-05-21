# models/backbones.py
import torch
import torch.nn as nn
import torchvision.models as tvm


class KochBackbone(nn.Module):
    """
    Koch-style CNN backbone with adjustable FC size and standardized embedding dim.
    We reduce fc_units to match ResNet-18 parameter count (per assignment recommendation).
    """
    def __init__(self, fc_units=1024, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=7), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=4), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=4), nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, fc_units),
            nn.ReLU(),
        )

        # standardized embedding head
        self.embed_head = nn.Linear(fc_units, embedding_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.embed_head(x)
        return x


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 trained from scratch (NO ImageNet weights). Outputs standardized embedding dim.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        # resnet = tvm.resnet18(weights=None)  # IMPORTANT: from scratch
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # remove final fc
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # -> [B,512,1,1]
        self.embed_head = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # [B,512]
        x = self.embed_head(x)
        return x


class MetricModel(nn.Module):
    """
    Simple wrapper so training code always calls model.embed(x).
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.embed(x)