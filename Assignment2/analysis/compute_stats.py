import torch
import torch.nn as nn
from thop import profile

from models.siamese_koch import SiameseKoch
from models.resnet_backbone import ResNet18Backbone


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 105
EMBED_DIM = 128

class EmbedWrapper(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, x):
        return self.siamese_model.embed(x)


def count_model(model, name):
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # wrap siamese encoder properly for thop
    if hasattr(model, "embed"):
        wrapped = EmbedWrapper(model).to(DEVICE)
        macs, params = profile(
            wrapped,
            inputs=(dummy,),
            verbose=False
        )
    else:
        macs, params = profile(
            model,
            inputs=(dummy,),
            verbose=False
        )

    print(f"{name}:")
    print(f"  Parameters: {params / 1e6:.2f} M")
    print(f"  MACs: {macs / 1e9:.2f} G")
    print(f"  Approx FLOPs: {2 * macs / 1e9:.2f} G\n")


def main():
    koch = SiameseKoch(embed_dim=EMBED_DIM).to(DEVICE)
    resnet = ResNet18Backbone(embed_dim=EMBED_DIM).to(DEVICE)

    count_model(koch, "Koch CNN (reduced)")
    count_model(resnet, "ResNet-18 (scratch)")


if __name__ == "__main__":
    main()
