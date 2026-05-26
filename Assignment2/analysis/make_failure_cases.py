

import os
import sys
import csv
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
TEST_PAIRS_FILE = "data/lfwa/pairsDevTest.txt"

DEFAULT_CKPT = "results/experiment1_loss/triplet_semihard_m0.2/20260519_143300/model_best.pth"
DEFAULT_SUMMARY = "results/experiment1_loss/triplet_eval_semihard_m0.2/summary.txt"

OUTPUT_DIR = "results/failure_cases"
BATCH_SIZE = 32


transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def extract_identity(path):
    """
    Extract identity name from image path.
    Example:
    data/lfwa/aligned_images/lfw2/George_Bush/George_Bush_0001.jpg
    -> George_Bush
    """
    return os.path.basename(os.path.dirname(path))


def read_threshold(summary_path):
    """
    Read threshold_distance_from_val from evaluation summary file.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("threshold_distance_from_val"):
                return float(line.strip().split(":")[1])

    raise ValueError(f"Could not find threshold_distance_from_val in {summary_path}")


def compute_test_distances(model):
    """
    Compute L2 distances for all official test pairs.
    """
    pairs = parse_pairs_file(TEST_PAIRS_FILE, IMAGES_ROOT)
    dataset = LFWSiameseDataset(pairs, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_rows = []
    pair_index = 0

    model.eval()

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            e1 = model.embed(img1)
            e2 = model.embed(img2)

            dists = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-9)
            dists = dists.cpu().numpy()
            labels = labels.numpy()

            for dist, label in zip(dists, labels):
                path1, path2, _ = pairs[pair_index]

                all_rows.append({
                    "index": pair_index,
                    "path1": path1,
                    "path2": path2,
                    "identity1": extract_identity(path1),
                    "identity2": extract_identity(path2),
                    "label": int(label),
                    "distance": float(dist)
                })

                pair_index += 1

    return all_rows


def add_predictions(rows, threshold):
    """
    For distance-based verification:
    prediction = 1 if distance <= threshold
    prediction = 0 otherwise
    """
    for row in rows:
        row["prediction"] = 1 if row["distance"] <= threshold else 0

        if row["label"] == 0 and row["prediction"] == 1:
            row["case_type"] = "False Accept"

        elif row["label"] == 1 and row["prediction"] == 0:
            row["case_type"] = "False Reject"

        else:
            row["case_type"] = "Correct"

    return rows


def select_failure_cases(rows, top_k=3):
    """
    Select 3 strong False Accepts and 3 strong False Rejects.

    False Accept:
    label=0 but predicted=1
    We choose the smallest distances, because the model was most confident they are same.

    False Reject:
    label=1 but predicted=0
    We choose the largest distances, because the model was most confident they are different.
    """
    false_accepts = [r for r in rows if r["case_type"] == "False Accept"]
    false_rejects = [r for r in rows if r["case_type"] == "False Reject"]

    false_accepts = sorted(false_accepts, key=lambda r: r["distance"])[:top_k]
    false_rejects = sorted(false_rejects, key=lambda r: r["distance"], reverse=True)[:top_k]

    return false_accepts, false_rejects


def load_image(path):
    return Image.open(path).convert("RGB")


def make_montage(cases, title, output_path, threshold):
    """
    Create a clean montage image with 3 rows and 2 columns.
    Each row shows one misclassified pair.
    Text is placed above each pair without overlapping the images.
    """
    n = len(cases)

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(9, 3.8 * n)
    )

    if n == 1:
        axes = np.array([axes])

    for i, case in enumerate(cases):
        img1 = load_image(case["path1"])
        img2 = load_image(case["path2"])

        axes[i, 0].imshow(img1)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img2)
        axes[i, 1].axis("off")

        axes[i, 0].set_title(
            f"{case['identity1']}",
            fontsize=11,
            pad=8
        )

        axes[i, 1].set_title(
            f"{case['identity2']}",
            fontsize=11,
            pad=8
        )

        if case["case_type"] == "False Accept":
            row_text = (
                f"Prediction: Same | True label: Different | "
                f"distance={case['distance']:.4f} < threshold={threshold:.4f}"
            )
        else:
            row_text = (
                f"Prediction: Different | True label: Same | "
                f"distance={case['distance']:.4f} > threshold={threshold:.4f}"
            )

        axes[i, 0].text(
            1.15,
            1.18,
            row_text,
            transform=axes[i, 0].transAxes,
            ha="center",
            va="bottom",
            fontsize=11
        )

    fig.suptitle(title, fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(
        hspace=0.55,
        wspace=0.25
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_csv(false_accepts, false_rejects, output_path, threshold):
    rows = false_accepts + false_rejects

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_type",
                "index",
                "identity1",
                "identity2",
                "path1",
                "path2",
                "label",
                "prediction",
                "distance",
                "threshold"
            ]
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "case_type": row["case_type"],
                "index": row["index"],
                "identity1": row["identity1"],
                "identity2": row["identity2"],
                "path1": row["path1"],
                "path2": row["path2"],
                "label": row["label"],
                "prediction": row["prediction"],
                "distance": f"{row['distance']:.6f}",
                "threshold": f"{threshold:.6f}"
            })


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY)
    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading threshold...")
    threshold = read_threshold(args.summary)
    print(f"Threshold: {threshold:.6f}")

    print("Loading model...")
    model = SiameseKoch().to(DEVICE)

    checkpoint = torch.load(args.ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    print("Computing test distances...")
    rows = compute_test_distances(model)
    rows = add_predictions(rows, threshold)

    false_accepts, false_rejects = select_failure_cases(rows, top_k=args.top_k)

    print(f"False Accepts found: {len([r for r in rows if r['case_type'] == 'False Accept'])}")
    print(f"False Rejects found: {len([r for r in rows if r['case_type'] == 'False Reject'])}")

    print("\nSelected False Accepts:")
    for r in false_accepts:
        print(f"{r['identity1']} vs {r['identity2']} | distance={r['distance']:.4f}")

    print("\nSelected False Rejects:")
    for r in false_rejects:
        print(f"{r['identity1']} vs {r['identity2']} | distance={r['distance']:.4f}")

    csv_path = os.path.join(OUTPUT_DIR, "failure_cases.csv")
    fa_path = os.path.join(OUTPUT_DIR, "false_accepts.png")
    fr_path = os.path.join(OUTPUT_DIR, "false_rejects.png")

    save_csv(false_accepts, false_rejects, csv_path, threshold)

    make_montage(
        false_accepts,
        "False Accept Examples",
        fa_path,
        threshold
    )

    make_montage(
        false_rejects,
        "False Reject Examples",
        fr_path,
        threshold
    )

    print("\nSaved outputs:")
    print(csv_path)
    print(fa_path)
    print(fr_path)


if __name__ == "__main__":
    main()