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
from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAIRS_TRAIN = "data/pairsDevTrain.txt"
PAIRS_TEST = "data/pairsDevTest.txt"
LFW_ROOT = "data/lfw2"

# ⚠️ תעדכן את זה לפי ה-margin שאימנת
MARGIN = 1.0
CKPT_PATH = f"checkpoints/koch_contrastive_m{MARGIN}_best.pt"

RESULTS_DIR = "results"
SCORES_CSV = os.path.join(RESULTS_DIR, f"koch_contrastive_m{MARGIN}_test_scores.csv")
ROC_PNG = os.path.join(RESULTS_DIR, f"koch_contrastive_m{MARGIN}_roc.png")
SUMMARY_TXT = os.path.join(RESULTS_DIR, f"koch_contrastive_m{MARGIN}_summary.txt")
MISCLS_TXT = os.path.join(RESULTS_DIR, f"koch_contrastive_m{MARGIN}_misclassified.txt")

BATCH_SIZE = 64
VAL_FRAC = 0.1
SEED = 42


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_scores_from_distance(model, loader, device="cpu"):
    """
    For contrastive: score = -euclidean_distance(emb1, emb2)
    Higher score => more likely same identity.
    """
    model.eval()
    all_scores, all_labels, all_paths = [], [], []

    for batch in loader:
        if len(batch) == 3:
            img1, img2, label = batch
            paths = None
        else:
            img1, img2, label, p1, p2 = batch
            paths = list(zip(p1, p2))

        img1 = img1.to(device)
        img2 = img2.to(device)

        emb1 = model.embed(img1)
        emb2 = model.embed(img2)

        dist = torch.norm(emb1 - emb2, p=2, dim=1)  # [B]
        score = (-dist).detach().cpu().numpy()      # higher=better (same)

        all_scores.append(score)
        all_labels.append(np.array(label))

        if paths is not None:
            all_paths.extend(paths)

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)
    return all_scores, all_labels, all_paths


def best_threshold_by_val(scores, labels):
    """
    choose threshold t that maximizes accuracy on VAL:
    predict same if score >= t
    """
    uniq = np.unique(scores)
    best_t, best_acc = float(uniq[len(uniq)//2]), -1.0
    for t in uniq:
        preds = (scores >= t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, float(best_acc)


def main():
    ensure_dir(RESULTS_DIR)
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    # ----- load model -----
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = SiameseKoch()
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)

    # ----- val split from train (threshold selection) -----
    full_train = LFWDataset(PAIRS_TRAIN, LFW_ROOT, transform=transform, return_paths=False)
    n_total = len(full_train)
    n_val = int(n_total * VAL_FRAC)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(SEED)
    _, val_ds = random_split(full_train, [n_train, n_val], generator=gen)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_scores, val_labels, _ = predict_scores_from_distance(model, val_loader, device=DEVICE)

    thr, val_acc = best_threshold_by_val(val_scores, val_labels)

    # ----- test -----
    test_ds = LFWDataset(PAIRS_TEST, LFW_ROOT, transform=transform, return_paths=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_scores, test_labels, test_paths = predict_scores_from_distance(model, test_loader, device=DEVICE)

    test_preds = (test_scores >= thr).astype(int)
    test_acc = float((test_preds == test_labels).mean())

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    test_auc = float(auc(fpr, tpr))

    # ----- ROC plot -----
    plt.figure()
    plt.plot(fpr, tpr, label=f"Contrastive m={MARGIN} (AUC={test_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Koch Contrastive (m={MARGIN})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PNG, dpi=200)
    plt.close()

    # ----- scores csv -----
    with open(SCORES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score(-dist)", "label", "pred", "img1_path", "img2_path"])
        for i in range(len(test_scores)):
            p1, p2 = test_paths[i]
            w.writerow([float(test_scores[i]), int(test_labels[i]), int(test_preds[i]), p1, p2])

    # ----- misclassified list -----
    fa_idx = np.where((test_labels == 0) & (test_preds == 1))[0]  # false accept
    fr_idx = np.where((test_labels == 1) & (test_preds == 0))[0]  # false reject

    fa_idx = fa_idx[:10]
    fr_idx = fr_idx[:10]

    with open(MISCLS_TXT, "w", encoding="utf-8") as f:
        f.write("FALSE ACCEPTS (label=0, pred=1)\n")
        for i in fa_idx:
            p1, p2 = test_paths[i]
            f.write(f"score={test_scores[i]:.4f} | {p1} || {p2}\n")

        f.write("\nFALSE REJECTS (label=1, pred=0)\n")
        for i in fr_idx:
            p1, p2 = test_paths[i]
            f.write(f"score={test_scores[i]:.4f} | {p1} || {p2}\n")

    # ----- summary -----
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"Score definition: score = -L2_distance(emb1, emb2)\n")
        f.write(f"Threshold selected on VAL: {thr:.6f}\n")
        f.write(f"Val accuracy at threshold: {val_acc:.4f}\n")
        f.write(f"Test accuracy at threshold: {test_acc:.4f}\n")
        f.write(f"Test AUC: {test_auc:.4f}\n")
        f.write(f"ROC saved to: {ROC_PNG}\n")
        f.write(f"Scores saved to: {SCORES_CSV}\n")
        f.write(f"Misclassified list saved to: {MISCLS_TXT}\n")

    print("✅ EVAL DONE (CONTRASTIVE)")
    print(f"Threshold(val): {thr:.6f} | ValAcc={val_acc:.4f}")
    print(f"TestAcc={test_acc:.4f} | AUC={test_auc:.4f}")
    print(f"Saved ROC: {ROC_PNG}")


if __name__ == "__main__":
    main()