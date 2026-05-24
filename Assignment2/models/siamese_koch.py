import torch
import torch.nn as nn
import torch.nn.functional as F


class KochCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseKoch(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = KochCNN()
        self.classifier = nn.Sequential(nn.Linear(4096, 1),)
        
    def embed(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        # L1 distance
        diff = torch.abs(f1 - f2)
        out = self.classifier(diff)

        return torch.sigmoid(out)
