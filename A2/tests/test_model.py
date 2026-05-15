import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.siamese_koch import SiameseKoch

model = SiameseKoch()

x1 = torch.randn(4, 3, 105, 105)
x2 = torch.randn(4, 3, 105, 105)

out = model(x1, x2)

print("Output shape:", out.shape)
print("Output:", out)