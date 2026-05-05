# Assignment2/models/siamese_koch.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class KochCNN(nn.Module):
    """
    This is the shared CNN backbone from Koch et al. (2015)
    """

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

        # Fully connected embedding layer
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseKoch(nn.Module):
    """
    Siamese network with L1 distance + sigmoid head
    """

    def __init__(self):
        super().__init__()
        self.backbone = KochCNN()

        self.classifier = nn.Linear(4096, 1)

    def forward(self, img1, img2):
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)

        # L1 distance
        diff = torch.abs(emb1 - emb2)

        out = self.classifier(diff)
        out = torch.sigmoid(out)

        return out
