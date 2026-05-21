# losses/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Hadsell et al. Contrastive Loss:
    y=1 (same):      L = d^2
    y=0 (different): L = max(0, margin - d)^2
    where d = ||e1 - e2||_2
    """
    def __init__(self, margin=1.0, eps=1e-9):
        super().__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, emb1, emb2, label):
        if label.dim() == 2:
            label = label.squeeze(1)
        label = label.float()

        d = torch.sqrt(torch.sum((emb1 - emb2) ** 2, dim=1) + self.eps)  # (B,)
        pos = label * (d ** 2)
        neg = (1 - label) * (F.relu(self.margin - d) ** 2)
        
        return (pos + neg).mean()