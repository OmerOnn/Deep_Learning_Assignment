import os
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from itertools import combinations

from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LFW_ROOT = "data/lfw2"
CHECKPOINT = "checkpoints/koch_triplet_semihard_m0.2_best.pt"

N_IDENTITIES = 20
MAX_IMAGES_PER_ID = 10


def load_image(path):
    img = Image.open(path).convert("RGB")
    return T.Compose([
        T.Resize((105, 105)),
        T.ToTensor()
    ])(img)


def main():
    # load model
    model = SiameseKoch(embed_dim=128).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    identities = [
        d for d in os.listdir(LFW_ROOT)
        if os.path.isdir(os.path.join(LFW_ROOT, d))
    ]
    identities = random.sample(identities, N_IDENTITIES)

    embeddings_by_id = {}

    with torch.no_grad():
        for identity in identities:
            folder = os.path.join(LFW_ROOT, identity)
            images = os.listdir(folder)
            images = random.sample(images, min(len(images), MAX_IMAGES_PER_ID))

            embs = []
            for img_name in images:
                img = load_image(os.path.join(folder, img_name))
                img = img.unsqueeze(0).to(DEVICE)
                emb = model.embed(img)
                embs.append(emb.cpu())

            embeddings_by_id[identity] = torch.cat(embs, dim=0)

    intra_distances = []
    inter_distances = []

    # intra-class distances
    for embs in embeddings_by_id.values():
        for i, j in combinations(range(len(embs)), 2):
            intra_distances.append(
                F.pairwise_distance(embs[i:i+1], embs[j:j+1]).item()
            )

    # inter-class distances
    ids = list(embeddings_by_id.keys())
    for i, j in combinations(range(len(ids)), 2):
        e1 = embeddings_by_id[ids[i]]
        e2 = embeddings_by_id[ids[j]]
        inter_distances.append(
            F.pairwise_distance(e1.mean(dim=0, keepdim=True),
                                 e2.mean(dim=0, keepdim=True)).item()
        )

    # plot
    plt.figure(figsize=(8, 6))
    plt.hist(intra_distances, bins=40, alpha=0.7, label="Intra-class")
    plt.hist(inter_distances, bins=40, alpha=0.7, label="Inter-class")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Count")
    plt.title("Embedding Distance Distributions (Triplet Semihard)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/results/embedding_distance_hist.png")
    plt.close()


if __name__ == "__main__":
    main()