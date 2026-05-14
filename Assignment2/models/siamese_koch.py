import torch
import torch.nn as nn


class KochCNN(nn.Module):
    """
    Koch et al. (2015) CNN backbone with reduced capacity
    """
    def __init__(self, embed_dim=128):
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
            nn.Linear(256 * 6 * 6, 1024),  # ↓ reduced from 4096
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseKoch(nn.Module):
    """
    Siamese wrapper for metric learning (Triplet / Contrastive)
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.backbone = KochCNN(embed_dim=embed_dim)

    def forward(self, img1, img2):
        emb1 = self.backbone(img1)
        emb2 = self.backbone(img2)
        return emb1, emb2

    def embed(self, img):
        return self.backbone(img)