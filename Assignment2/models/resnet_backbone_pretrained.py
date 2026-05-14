import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class FrozenResNet18Backbone(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1
        self.net = resnet18(weights=weights)

        # freeze all parameters
        for p in self.net.parameters():
            p.requires_grad = False

        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, embed_dim)

        # freeze the embedding head too
        for p in self.net.fc.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.net(x)