import os
import random
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from PIL import Image
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

    embeddings = []
    labels = []

    with torch.no_grad():
        for idx, identity in enumerate(identities):
            folder = os.path.join(LFW_ROOT, identity)
            images = os.listdir(folder)
            images = random.sample(images, min(len(images), MAX_IMAGES_PER_ID))

            for img_name in images:
                img = load_image(os.path.join(folder, img_name))
                img = img.unsqueeze(0).to(DEVICE)
                emb = model.embed(img).cpu().numpy()[0]
                embeddings.append(emb)
                labels.append(idx)


    embeddings = np.vstack(embeddings)
    perplexity = min(30, embeddings.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)


    plt.figure(figsize=(8, 8))
    for i in range(N_IDENTITIES):
        idxs = [j for j, l in enumerate(labels) if l == i]
        plt.scatter(
            embeddings_2d[idxs, 0],
            embeddings_2d[idxs, 1],
            s=20
        )

    plt.title("t-SNE of Test Embeddings (Triplet Semihard)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("analysis/results/tsne_triplet_embeddings.png")
    plt.close()


if __name__ == "__main__":
    main()