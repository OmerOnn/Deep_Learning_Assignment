import os, json
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision import transforms
from PIL import Image

from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EP_PATH = "results/oneshot/episodes_seed42.json"
OUT_DIR = "results/oneshot"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = "results/experiment1_loss/koch_bce/20260521_132214/model_best.pth"

transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor()
])

def fix_path(p: str) -> str:

    if os.path.exists(p):
        return p
    p2 = p.replace("data/lfw2/", "data/lfwa/aligned_images/lfw2/")
    if os.path.exists(p2):
        return p2
    p3 = p.replace("data/lfwa/aligned_images/lfw2/", "data/lfwa/aligned_images/")
    if os.path.exists(p3):
        return p3
    return p  

@torch.no_grad()
def embed_image(model, path):
    path = fix_path(path)
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    e = model.embed(x)           # [1, D]
    return e.squeeze(0)          # [D]

def l2(e1, e2):
    return torch.sqrt(torch.sum((e1 - e2) ** 2) + 1e-9)

def eval_oneshot(model, episodes, N):
    eps = episodes[str(N)]
    correct = 0

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

    return correct / len(eps)

def load_koch_model():
    model = SiameseKoch().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    return model

def main():
    with open(EP_PATH, "r") as f:
        episodes = json.load(f)

    model = load_koch_model()
    N_LIST = [2, 5, 20]
    results = {}

    for N in N_LIST:
        acc = eval_oneshot(model, episodes, N)
        results[str(N)] = acc
        print(f"koch_bce | {N}-way acc: {acc:.4f}")

    out_path = os.path.join(OUT_DIR, "koch_bce_oneshot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()