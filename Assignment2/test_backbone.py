from models.siamese_koch import SiameseKoch
import torch

model = SiameseKoch(embed_dim=128)

x = torch.randn(4, 3, 105, 105)
e = model.embed(x)

print(e.shape)