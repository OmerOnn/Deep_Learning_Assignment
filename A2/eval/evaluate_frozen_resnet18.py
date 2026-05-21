# eval/evaluate_frozen_resnet18.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as tvm

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
OUTPUT_DIR = "results/experiment3_frozen_resnet18"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def cosine_score(e1, e2):
    # normalize then cosine similarity
    e1 = F.normalize(e1, p=2, dim=1)
    e2 = F.normalize(e2, p=2, dim=1)
    return (e1 * e2).sum(dim=1)  # [-1, 1]

def compute_scores(pairs_file, model):
    pairs = parse_pairs_file(pairs_file, IMAGES_ROOT)
    ds = LFWSiameseDataset(pairs, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    labels, scores = [], []

    model.eval()
    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(loader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)

            e1 = model(img1)
            e2 = model(img2)

            s = cosine_score(e1, e2)
            scores.extend(s.cpu().numpy())
            labels.extend(label.numpy())

            if i % 10 == 0:
                print(f"Inference batch {i}/{len(loader)}")

    return np.array(labels).astype(int), np.array(scores).astype(float)

def best_threshold_by_val(labels, scores):
    # predict same if score >= thr
    candidates = np.unique(scores)
    best_acc, best_thr = -1.0, None
    for thr in candidates:
        preds = (scores >= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_thr, best_acc

def save_roc(labels, scores, path_png, title):
    fpr, tpr, _ = roc_curve(labels, scores)  # scores: higher = more similar
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig(path_png)
    plt.close()

    return roc_auc

def main():
    print("Loading frozen ImageNet-pretrained ResNet-18...")
    resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])  # [B,512,1,1]

    class FrozenResnetEmbed(torch.nn.Module):
        def __init__(self, feat):
            super().__init__()
            self.feat = feat
        def forward(self, x):
            x = self.feat(x)
            return x.view(x.size(0), -1)  # [B,512]

    model = FrozenResnetEmbed(backbone).to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    print("\nCompute VAL scores (proxy on pairsDevTrain)...")
    val_labels, val_scores = compute_scores("data/lfwa/pairsDevTrain.txt", model)

    thr, val_acc = best_threshold_by_val(val_labels, val_scores)
    print(f"Best VAL threshold (cosine): {thr:.6f} | VAL Acc: {val_acc:.4f}")

    val_auc = save_roc(val_labels, val_scores, os.path.join(OUTPUT_DIR, "roc_val.png"),
                       "Frozen ResNet-18 ROC (VAL proxy)")
    print(f"VAL AUC: {val_auc:.4f}")

    print("\nCompute TEST scores...")
    test_labels, test_scores = compute_scores("data/lfwa/pairsDevTest.txt", model)

    test_preds = (test_scores >= thr).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)

    test_auc = save_roc(test_labels, test_scores, os.path.join(OUTPUT_DIR, "roc_test.png"),
                        "Frozen ResNet-18 ROC (TEST)")
    print(f"TEST Acc: {test_acc:.4f}")
    print(f"TEST AUC: {test_auc:.4f}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXP3: FROZEN RESNET18 (IMAGENET) ===\n")
        f.write(f"threshold_cosine_from_val: {thr:.6f}\n")
        f.write(f"val_acc: {val_acc:.4f}\n")
        f.write(f"val_auc: {val_auc:.4f}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")
        f.write(f"test_auc: {test_auc:.4f}\n")

    print(f"\n✅ Saved outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
