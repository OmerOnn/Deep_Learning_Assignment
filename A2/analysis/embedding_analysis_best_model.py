import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from sklearn.manifold import TSNE

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# חשוב: זה צריך להתאים למבנה הדאטה אצלך (עם/בלי lfw2)
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"

# המודל הטוב ביותר שלך (Triplet semihard Exp1)
CKPT = "results/experiment1_loss/triplet_semihard_m0.2/20260519_143300/model_best.pth"

OUT_DIR = "results/embedding"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor()
])

def load_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

def list_identities(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def sample_images(identity, k=10):
    folder = os.path.join(IMAGES_ROOT, identity)
    imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    imgs = sorted(imgs)
    if len(imgs) <= k:
        return imgs
    return random.sample(imgs, k)

@torch.no_grad()
def embed_paths(model, paths):
    xs = torch.stack([load_img(p) for p in paths], dim=0).to(DEVICE)
    emb = model.embed(xs).cpu().numpy()
    return emb

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model
    model = SiameseKoch().to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()

    # ---- t-SNE על 20 זהויות ----
    all_ids = list_identities(IMAGES_ROOT)
    chosen = random.sample(all_ids, 20)

    paths, labs = [], []
    for ident in chosen:
        imgs = sample_images(ident, k=10)
        paths += imgs
        labs += [ident] * len(imgs)

    E = embed_paths(model, paths)  # [N,128]

    Z = TSNE(n_components=2, perplexity=10, init="pca", random_state=42).fit_transform(E)

    plt.figure(figsize=(9,7))
    for ident in chosen:
        idx = [i for i,l in enumerate(labs) if l == ident]
        plt.scatter(Z[idx,0], Z[idx,1], s=10, alpha=0.8)
    plt.title("t-SNE of embeddings (20 identities)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    tsne_path = os.path.join(OUT_DIR, "tsne_best_model.png")
    plt.savefig(tsne_path, dpi=200)
    plt.close()

    # ---- intra/inter distances + histogram ----
    by_id = {}
    for e, lab in zip(E, labs):
        by_id.setdefault(lab, []).append(e)

    intra = []
    for lab, vecs in by_id.items():
        vecs = np.array(vecs)
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                intra.append(np.linalg.norm(vecs[i]-vecs[j]))
    intra = np.array(intra, dtype=float)

    inter = []
    labels = list(by_id.keys())
    for _ in range(min(5000, max(2000, len(intra)*3))):
        a, b = random.sample(labels, 2)
        ea = random.choice(by_id[a])
        eb = random.choice(by_id[b])
        inter.append(np.linalg.norm(ea-eb))
    inter = np.array(inter, dtype=float)

    plt.figure(figsize=(8,5))
    plt.hist(intra, bins=40, alpha=0.6, label="intra-class")
    plt.hist(inter, bins=40, alpha=0.6, label="inter-class")
    plt.title("Embedding L2 distance distributions")
    plt.xlabel("L2 distance")
    plt.ylabel("count")
    plt.legend()
    hist_path = os.path.join(OUT_DIR, "embedding_distance_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()

    stats_path = os.path.join(OUT_DIR, "embedding_distance_stats.txt")
    with open(stats_path, "w") as f:
        f.write(f"Intra mean: {intra.mean():.4f}\n")
        f.write(f"Intra std: {intra.std():.4f}\n")
        f.write(f"Intra median: {np.median(intra):.4f}\n")
        f.write(f"Inter mean: {inter.mean():.4f}\n")
        f.write(f"Inter std: {inter.std():.4f}\n")
        f.write(f"Inter median: {np.median(inter):.4f}\n")

    print("✅ Saved:")
    print(tsne_path)
    print(hist_path)
    print(stats_path)

if __name__ == "__main__":
    main()
