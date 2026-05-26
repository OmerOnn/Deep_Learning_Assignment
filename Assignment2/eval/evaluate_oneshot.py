sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, json
import numpy as np
import sys
import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from models.backbones import KochBackbone, ResNet18Backbone, MetricModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EP_PATH = "results/oneshot/episodes_seed42.json"
OUT_DIR = "results/oneshot"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor()
])

def embed_image(model, path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        e = model.embed(x)
    return e.squeeze(0)

def l2(e1, e2):
    return torch.sqrt(torch.sum((e1-e2)**2) + 1e-9)

def eval_model(model, name):
    with open(EP_PATH, "r") as f:
        episodes = json.load(f)

    results = {}
    for N in ["2","5","20"]:
        correct = 0
        eps = episodes[N]
        for ep in eps:
            target = ep["target"]
            query = ep["query"]
            candidates = ep["candidates"]
            support = ep["support"]

            eq = embed_image(model, query)
            best_id = None
            best_d = 1e9
            for cid in candidates:
                es = embed_image(model, support[cid])
                d = l2(eq, es).item()
                if d < best_d:
                    best_d = d
                    best_id = cid

            if best_id == target:
                correct += 1

        acc = correct / len(eps)
        results[N] = acc
        print(f"{name} | {N}-way acc: {acc:.4f}")

    out_path = os.path.join(OUT_DIR, f"{name}_oneshot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

def main():
    backbone = ResNet18Backbone(embedding_dim=128)
    model = MetricModel(backbone).to(DEVICE)
    model.eval()
    eval_model(model, "frozen_resnet18")

if __name__ == "__main__":
    main()
