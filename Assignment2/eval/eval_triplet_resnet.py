import os
import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.lfw_pairs import LFWDataset
from models.resnet_backbone import ResNet18Backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAIRS_TRAIN = "data/pairsDevTrain.txt"
PAIRS_TEST = "data/pairsDevTest.txt"
LFW_ROOT = "data/lfw2"

MODE = "semihard"
MARGIN = 0.2
CKPT_PATH = f"checkpoints/resnet18_triplet_{MODE}_m{MARGIN}_best.pt"

RESULTS_DIR = "results"
SCORES_CSV = os.path.join(RESULTS_DIR, f"resnet18_triplet_{MODE}_m{MARGIN}_test_scores.csv")
ROC_PNG = os.path.join(RESULTS_DIR, f"resnet18_triplet_{MODE}_m{MARGIN}_roc.png")
SUMMARY_TXT = os.path.join(RESULTS_DIR, f"resnet18_triplet_{MODE}_m{MARGIN}_summary.txt")
MISCLS_TXT = os.path.join(RESULTS_DIR, f"resnet18_triplet_{MODE}_m{MARGIN}_misclassified.txt")

BATCH_SIZE = 64
VAL_FRAC = 0.1
SEED = 42


@torch.no_grad()
def predict_scores_from_distance(model, loader):
    model.eval()
    scores = []
    labels = []

    for batch in loader:
        if len(batch) == 3:
            img1, img2, y = batch
        else:
            img1, img2, y, _, _ = batch  # ignore paths

        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)

        emb1 = model(img1)
        emb2 = model(img2)

        dist = torch.norm(emb1 - emb2, p=2, dim=1)
        scores.append((-dist).cpu().numpy())
        labels.append(y.numpy())

    return np.concatenate(scores), np.concatenate(labels)


def best_threshold_by_val(scores, labels):
    best_t, best_acc = scores[0], -1
    for t in np.unique(scores):
        acc = ((scores >= t).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ResNet18Backbone(embed_dim=128).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Validation
    full_train = LFWDataset(PAIRS_TRAIN, LFW_ROOT, transform=transform)
    n_val = int(len(full_train) * VAL_FRAC)
    _, val_ds = random_split(full_train, [len(full_train)-n_val, n_val])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    val_scores, val_labels = predict_scores_from_distance(model, val_loader)
    thr, val_acc = best_threshold_by_val(val_scores, val_labels)

    # Test
    test_ds = LFWDataset(PAIRS_TEST, LFW_ROOT, transform=transform, return_paths=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    test_scores, test_labels = predict_scores_from_distance(model, test_loader)
    test_preds = (test_scores >= thr).astype(int)
    test_acc = (test_preds == test_labels).mean()

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    test_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.savefig(ROC_PNG)
    plt.close()

    with open(SUMMARY_TXT, "w") as f:
        f.write(f"ValAcc={val_acc:.4f}\n")
        f.write(f"TestAcc={test_acc:.4f}\n")
        f.write(f"AUC={test_auc:.4f}\n")

    print("✅ RESNET EVAL DONE")
    print(f"TestAcc={test_acc:.4f} | AUC={test_auc:.4f}")


if __name__ == "__main__":
    main()