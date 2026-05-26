import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
from collections import defaultdict

from models.siamese_koch import SiameseKoch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
OUT_DIR = "results/embedding"

MODEL_CKPT = "results/experiment1_loss/triplet_semihard_m0.2/20260519_143300/model_best.pth"

NUM_IDENTITIES = 20
MAX_IMAGES_PER_IDENTITY = 5
MIN_IMAGES_PER_IDENTITY = 2
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def load_model():
    model = SiameseKoch().to(DEVICE)

    checkpoint = torch.load(MODEL_CKPT, map_location=DEVICE)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def collect_test_images():
    """
    Collect images from identities that have at least MIN_IMAGES_PER_IDENTITY images.
    This is needed so t-SNE can show clusters instead of one point per identity.
    """
    identities = {}

    for identity in sorted(os.listdir(IMAGES_ROOT)):
        identity_dir = os.path.join(IMAGES_ROOT, identity)

        if not os.path.isdir(identity_dir):
            continue

        images = [
            os.path.join(identity_dir, f)
            for f in sorted(os.listdir(identity_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(images) >= MIN_IMAGES_PER_IDENTITY:
            identities[identity] = images

    return identities


@torch.no_grad()
def embed_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    emb = model.embed(x)
    return emb.squeeze(0).cpu().numpy()


def compute_embeddings(model, selected_identities):
    embeddings = []
    labels = []

    for identity, image_paths in selected_identities.items():
        for path in image_paths:
            emb = embed_image(model, path)
            embeddings.append(emb)
            labels.append(identity)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def plot_tsne(embeddings, labels):
    """
    Create t-SNE visualization with multiple images per identity.
    """
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    n_samples = len(embeddings)
    perplexity = min(20, max(5, n_samples // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=RANDOM_SEED
    )

    points = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(labels))

    for identity in unique_labels:
        idx = labels == identity
        plt.scatter(
            points[idx, 0],
            points[idx, 1],
            s=35,
            alpha=0.8,
            label=identity
        )

    plt.title(f"t-SNE of embeddings ({len(unique_labels)} identities, multiple images per identity)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")

    # Do not show a huge legend if it becomes too crowded
    if len(unique_labels) <= 20:
        plt.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_best_model.png"), dpi=200)
    plt.close()


def compute_intra_inter_distances(embeddings, labels):
    """
    Compute pairwise L2 distances:
    intra-class: same identity
    inter-class: different identities
    """
    intra = []
    inter = []

    n = len(embeddings)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])

            if labels[i] == labels[j]:
                intra.append(dist)
            else:
                inter.append(dist)

    intra = np.array(intra)
    inter = np.array(inter)

    return intra, inter


def plot_distance_hist(intra, inter):
    """
    Plot normalized histograms so both distributions are visible.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(
        intra,
        bins=30,
        alpha=0.6,
        density=True,
        label="intra-class"
    )

    plt.hist(
        inter,
        bins=30,
        alpha=0.6,
        density=True,
        label="inter-class"
    )

    plt.title("Embedding L2 distance distributions")
    plt.xlabel("L2 distance")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "embedding_distance_hist.png"), dpi=200)
    plt.close()


def save_distance_stats(intra, inter):
    stats_path = os.path.join(OUT_DIR, "embedding_distance_stats.txt")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Embedding Distance Statistics ===\n")
        f.write(f"intra_mean: {np.mean(intra):.4f}\n")
        f.write(f"intra_std: {np.std(intra):.4f}\n")
        f.write(f"intra_median: {np.median(intra):.4f}\n")
        f.write(f"inter_mean: {np.mean(inter):.4f}\n")
        f.write(f"inter_std: {np.std(inter):.4f}\n")
        f.write(f"inter_median: {np.median(inter):.4f}\n")
        f.write(f"num_intra_pairs: {len(intra)}\n")
        f.write(f"num_inter_pairs: {len(inter)}\n")

    print(f"Saved stats to: {stats_path}")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading model...")
    model = load_model()

    print("Collecting identities...")
    identities = collect_test_images()

    print(f"Found {len(identities)} identities with at least {MIN_IMAGES_PER_IDENTITY} images")

    eligible = list(identities.keys())
    random.shuffle(eligible)

    selected_names = eligible[:NUM_IDENTITIES]

    selected_identities = {}

    for identity in selected_names:
        images = identities[identity]
        random.shuffle(images)
        selected_identities[identity] = images[:MAX_IMAGES_PER_IDENTITY]

    total_images = sum(len(v) for v in selected_identities.values())

    print(f"Selected {len(selected_identities)} identities")
    print(f"Total images used: {total_images}")

    print("Computing embeddings...")
    embeddings, labels = compute_embeddings(model, selected_identities)

    print("Creating t-SNE...")
    plot_tsne(embeddings, labels)

    print("Computing intra/inter distances...")
    intra, inter = compute_intra_inter_distances(embeddings, labels)

    print(f"intra mean = {np.mean(intra):.4f}, std = {np.std(intra):.4f}")
    print(f"inter mean = {np.mean(inter):.4f}, std = {np.std(inter):.4f}")

    print("Creating distance histogram...")
    plot_distance_hist(intra, inter)

    save_distance_stats(intra, inter)

    print("Done.")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()