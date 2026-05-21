# eval/evaluate_contrastive.py
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
MODEL_CKPT = "results/experiment1_loss/contrastive_m0.2/20260518_222913/model_best.pth"
MARGIN_NAME = "m0.2"
OUTPUT_DIR = f"results/experiment1_loss/contrastive_eval_{MARGIN_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

def compute_scores(pairs_file, model):
    pairs = parse_pairs_file(pairs_file, IMAGES_ROOT)
    ds = LFWSiameseDataset(pairs, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    labels, dists = [], []

    model.eval()
    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(loader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            e1 = model.embed(img1)
            e2 = model.embed(img2)

            dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-9)  # L2 distance
            dists.extend(dist.cpu().numpy())
            labels.extend(label.numpy())

            if i % 10 == 0:
                print(f"Inference batch {i}/{len(loader)}")

    labels = np.array(labels).astype(int)
    dists = np.array(dists).astype(float)
    scores = -dists

    return labels, dists, scores

def best_threshold_by_val(labels, dists):
    candidates = np.unique(dists)
    best_acc, best_thr = -1.0, None
    
    for thr in candidates:
        preds = (dists <= thr).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    return best_thr, best_acc

def save_roc(labels, scores, path_png, title):
    fpr, tpr, _ = roc_curve(labels, scores)
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
    print("Loading model...")

    model = SiameseKoch().to(DEVICE)
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)

    print("\nCompute VAL scores...")

    val_labels, val_dists, val_scores = compute_scores("data/lfwa/pairsDevTrain.txt", model)
    thr, val_acc = best_threshold_by_val(val_labels, val_dists)

    print(f"Best VAL threshold (distance): {thr:.6f} | VAL Acc: {val_acc:.4f}")

    val_auc = save_roc(val_labels, val_scores, os.path.join(OUTPUT_DIR, "roc_val.png"),"Contrastive ROC (VAL proxy)")

    print(f"VAL AUC: {val_auc:.4f}")
    print("\nCompute TEST scores...")

    test_labels, test_dists, test_scores = compute_scores("data/lfwa/pairsDevTest.txt", model)
    test_preds = (test_dists <= thr).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = save_roc(test_labels, test_scores, os.path.join(OUTPUT_DIR, "roc_test.png"),"Contrastive ROC (TEST)")

    print(f"TEST Acc: {test_acc:.4f}")
    print(f"TEST AUC: {test_auc:.4f}")

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXP1: CONTRASTIVE EVAL ===\n")
        f.write(f"model_ckpt: {MODEL_CKPT}\n")
        f.write(f"threshold_distance_from_val: {thr:.6f}\n")
        f.write(f"val_acc: {val_acc:.4f}\n")
        f.write(f"val_auc: {val_auc:.4f}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")
        f.write(f"test_auc: {test_auc:.4f}\n")

    print(f"\n✅ Saved eval outputs to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()