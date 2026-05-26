import os, random, json
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"  
PAIRS_TRAIN = "data/lfwa/pairsDevTrain.txt"
PAIRS_TEST  = "data/lfwa/pairsDevTest.txt"

VAL_RATIO = 0.2
SEED = 42

OUT_DIR = "results/dataset_plots"
os.makedirs(OUT_DIR, exist_ok=True)
# ====================

def list_identities_images(images_root):
    """counts images per identity from folder structure"""
    counts = {}
    for ident in os.listdir(images_root):
        p = os.path.join(images_root, ident)
        if not os.path.isdir(p):
            continue
        imgs = [f for f in os.listdir(p) if f.lower().endswith(".jpg")]
        counts[ident] = len(imgs)
    return counts  # dict: ident -> num_images

def parse_pairs_file(path):
    """
    LFW pairs format:
    - positive line: name  idx1 idx2
    - negative line: name1 idx1 name2 idx2
    We return list of tuples:
      (label, ident1, img1, ident2, img2)
    where img paths are full paths under IMAGES_ROOT
    """
    pairs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    # first line may be header (10 300) in original LFW, but your parser probably ignores it.
    # We'll robustly skip if it has 2 ints.
    start = 0
    toks0 = lines[0].split()
    if len(toks0) == 2 and toks0[0].isdigit() and toks0[1].isdigit():
        start = 1

    for ln in lines[start:]:
        toks = ln.split()
        if len(toks) == 3:
            name, a, b = toks
            img1 = os.path.join(IMAGES_ROOT, name, f"{name}_{int(a):04d}.jpg")
            img2 = os.path.join(IMAGES_ROOT, name, f"{name}_{int(b):04d}.jpg")
            pairs.append((1, name, img1, name, img2))
        elif len(toks) == 4:
            name1, a, name2, b = toks
            img1 = os.path.join(IMAGES_ROOT, name1, f"{name1}_{int(a):04d}.jpg")
            img2 = os.path.join(IMAGES_ROOT, name2, f"{name2}_{int(b):04d}.jpg")
            pairs.append((0, name1, img1, name2, img2))
        else:
            # ignore malformed
            continue
    return pairs

def split_train_val_pairs(pairs, val_ratio=0.2, seed=42):
    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)
    k = int(len(pairs)*(1-val_ratio))
    return pairs[:k], pairs[k:]

def plot_hist_count_of_identities(values_per_identity, title, xlabel, out_png, log_y=True, max_x=None):
    """
    values_per_identity: list of ints, one per identity.
    We plot histogram of counts of identities that have value=k.
    """
    cnt = Counter(values_per_identity)
    xs = sorted(cnt.keys())
    ys = [cnt[x] for x in xs]

    plt.figure(figsize=(12,5))
    plt.bar(xs, ys, width=0.8, edgecolor="black")
    if log_y:
        plt.yscale("log")
    if max_x is not None:
        plt.xlim(0, max_x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of identities (log scale)" if log_y else "Number of identities")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def unique_images_referenced_per_identity(pairs):
    """
    For each identity, count unique image paths referenced in the pairs list.
    """
    refs = defaultdict(set)
    for label, id1, img1, id2, img2 in pairs:
        refs[id1].add(img1)
        refs[id2].add(img2)
    return {k: len(v) for k, v in refs.items()}

def main():
    # ===== 1) FULL dataset images per identity =====
    full_counts = list_identities_images(IMAGES_ROOT)
    values = list(full_counts.values())
    out1 = os.path.join(OUT_DIR, "full_images_per_identity_log.png")
    plot_hist_count_of_identities(
        values,
        title="Full LFW-a dataset - images per identity distribution (full)",
        xlabel="Number of images per identity",
        out_png=out1,
        log_y=True
    )

    # ===== 2) Train/Val/Test: pair-referenced unique images per identity =====
    train_pairs_all = parse_pairs_file(PAIRS_TRAIN)
    test_pairs = parse_pairs_file(PAIRS_TEST)
    train_pairs, val_pairs = split_train_val_pairs(train_pairs_all, VAL_RATIO, SEED)

    for name, pairs in [("train", train_pairs), ("validation", val_pairs), ("test", test_pairs)]:
        ref_counts = unique_images_referenced_per_identity(pairs)
        out = os.path.join(OUT_DIR, f"{name}_pair_referenced_unique_images_per_identity_log.png")
        plot_hist_count_of_identities(
            list(ref_counts.values()),
            title=f"{name.capitalize()} split - pair-referenced images per identity",
            xlabel="Number of unique images referenced in pair file",
            out_png=out,
            log_y=True
        )

    # ===== 3) Triplet-eligible identities in train (>=2 images available) =====
    # Option A (folder availability): use full_counts but restrict to identities that appear in TRAIN pairs
    train_identities = set()
    for label, id1, img1, id2, img2 in train_pairs:
        train_identities.add(id1); train_identities.add(id2)

    triplet_counts = []
    for ident in train_identities:
        k = full_counts.get(ident, 0)
        if k >= 2:
            triplet_counts.append(k)

    out3 = os.path.join(OUT_DIR, "train_triplet_eligible_identities_log.png")
    plot_hist_count_of_identities(
        triplet_counts,
        title="Train split - Triplet-eligible identities (at least 2 images)",
        xlabel="Number of images available for triplet sampling",
        out_png=out3,
        log_y=True,
        max_x=50  # optional: avoids giant tail stretching; remove if you want full
    )

    print("Saved dataset plots to:", OUT_DIR)
    print("  -", out1)
    print("  - train/validation/test pair-referenced plots")
    print("  -", out3)

if __name__ == "__main__":
    main()
