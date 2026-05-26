import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, d_ap, d_an):
        return F.relu(d_ap - d_an + self.margin).mean()
