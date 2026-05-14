import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
from torch.nn.functional import cosine_similarity

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.lfw_pairs import LFWDataset
from models.resnet_backbone_pretrained import FrozenResNet18Backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAIRS_TEST = "data/pairsDevTest.txt"
LFW_ROOT = "data/lfw2"

RESULTS_DIR = "results"
ROC_PNG = os.path.join(RESULTS_DIR, "frozen_resnet18_cosine_roc.png")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "frozen_resnet18_cosine_summary.txt")

BATCH_SIZE = 64


@torch.no_grad()
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    model = FrozenResNet18Backbone(embed_dim=128).to(DEVICE)
    model.eval()

    test_ds = LFWDataset(PAIRS_TEST, LFW_ROOT, transform=transform)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    scores, labels = [], []

    for img1, img2, y in loader:
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)

        emb1 = model(img1)
        emb2 = model(img2)

        sim = cosine_similarity(emb1, emb2)
        scores.append(sim.cpu().numpy())
        labels.append(y.numpy())

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)

    fpr, tpr, _ = roc_curve(labels, scores)
    test_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("Frozen ResNet‑18 (ImageNet) – Cosine Similarity")
    plt.savefig(ROC_PNG)
    plt.close()

    with open(SUMMARY_TXT, "w") as f:
        f.write(f"AUC={test_auc:.4f}\n")

    print("✅ FROZEN RESNET EVAL DONE")
    print(f"AUC={test_auc:.4f}")


if __name__ == "__main__":
    main()
