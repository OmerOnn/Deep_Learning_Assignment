sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import sys

from thop import profile
from models.backbones import KochBackbone, ResNet18Backbone, MetricModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = (1, 3, 105, 105) 
OUT_DIR = "results/experiment2_backbone/stats"

os.makedirs(OUT_DIR, exist_ok=True)

def count(model, name):
    model = model.to(DEVICE).eval()
    
    dummy = torch.randn(*INPUT_SIZE).to(DEVICE)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs

    with open(os.path.join(OUT_DIR, f"{name}_stats.txt"), "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Input: {INPUT_SIZE}\n")
        f.write(f"Params: {params}\n")
        f.write(f"MACs: {macs}\n")
        f.write(f"FLOPs(=2*MACs): {flops}\n")

    print(f"{name}: params={params:,} | MACs={macs:,} | FLOPs={flops:,}")

def main():
    emb_dim = 128
    koch = MetricModel(KochBackbone(fc_units=1024, embedding_dim=emb_dim))
    res  = MetricModel(ResNet18Backbone(embedding_dim=emb_dim))

    count(koch, f"koch_fc1024_emb{emb_dim}")
    count(res,  f"resnet18_scratch_emb{emb_dim}")

if __name__ == "__main__":
    main()