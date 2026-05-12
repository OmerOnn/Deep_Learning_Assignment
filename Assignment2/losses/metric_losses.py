import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Hadsell et al. contrastive loss.
    y=1 for positive (same), y=0 for negative (different).
    Uses Euclidean distance in embedding space.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, y):
        # emb1, emb2: [B, D]
        # y: [B, 1] float in {0,1}
        d = torch.norm(emb1 - emb2, p=2, dim=1)  # [B]
        y = y.squeeze(1)

        pos = y * (d ** 2)
        neg = (1 - y) * (F.relu(self.margin - d) ** 2)

        return (pos + neg).mean()