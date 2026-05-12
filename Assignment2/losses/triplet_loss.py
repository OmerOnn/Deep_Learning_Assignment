import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = torch.norm(anchor - positive, p=2, dim=1)
        d_an = torch.norm(anchor - negative, p=2, dim=1)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()