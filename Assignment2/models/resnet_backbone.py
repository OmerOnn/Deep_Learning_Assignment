import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = resnet18(pretrained=False)  # חשוב: from scratch
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        return self.net(x)