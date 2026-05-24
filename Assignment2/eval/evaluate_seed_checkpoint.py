import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch
from models.backbones import KochBackbone, MetricModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
BATCH_SIZE = 32


transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def build_model(model_type):
    if model_type == "exp1_triplet":
        return SiameseKoch().to(DEVICE)

    if model_type == "exp2_koch":
        backbone = KochBackbone(fc_units=1024, embedding_dim=128)
        return MetricModel(backbone).to(DEVICE)

    raise ValueError(f"Unknown model_type: {model_type}")


def compute_scores(pairs_file, model):
    pairs = parse_pairs_file(pairs_file, IMAGES_ROOT)
    dataset = LFWSiameseDataset(pairs, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    labels = []
    dists = []

    model.eval()

    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            e1 = model.embed(img1)
            e2 = model.embed(img2)

            dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-9)

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
    best_acc = -1.0
    best_thr = None

    for thr in candidates:
        preds = (dists <= thr).astype(int)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc


def save_roc(labels, scores, path_png, title):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.savefig(path_png)
    plt.close()

    return roc_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["exp1_triplet", "exp2_koch"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading model...")
    model = build_model(args.model_type)

    checkpoint = torch.load(args.ckpt, map_location=DEVICE)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    print("\nCompute VAL scores...")
    val_labels, val_dists, val_scores = compute_scores("data/lfwa/pairsDevTrain.txt", model)

    threshold, val_acc = best_threshold_by_val(val_labels, val_dists)
    val_auc = save_roc(
        val_labels,
        val_scores,
        os.path.join(args.out_dir, "roc_val.png"),
        "ROC VAL"
    )

    print(f"VAL threshold: {threshold:.6f}")
    print(f"VAL Acc: {val_acc:.4f}")
    print(f"VAL AUC: {val_auc:.4f}")

    print("\nCompute TEST scores...")
    test_labels, test_dists, test_scores = compute_scores("data/lfwa/pairsDevTest.txt", model)

    test_preds = (test_dists <= threshold).astype(int)
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = save_roc(
        test_labels,
        test_scores,
        os.path.join(args.out_dir, "roc_test.png"),
        "ROC TEST"
    )

    print(f"TEST Acc: {test_acc:.4f}")
    print(f"TEST AUC: {test_auc:.4f}")

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== SEED CHECKPOINT EVAL ===\n")
        f.write(f"model_type: {args.model_type}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"model_ckpt: {args.ckpt}\n")
        f.write(f"threshold_distance_from_val: {threshold:.6f}\n")
        f.write(f"val_acc: {val_acc:.4f}\n")
        f.write(f"val_auc: {val_auc:.4f}\n")
        f.write(f"test_acc: {test_acc:.4f}\n")
        f.write(f"test_auc: {test_auc:.4f}\n")

    print(f"\nSaved eval outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
