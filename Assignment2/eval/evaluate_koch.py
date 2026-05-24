sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import roc_curve, auc, accuracy_score
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch

# ========= CONFIG =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
MODEL_PATH = "results/debug_koch/model.pth"
OUTPUT_DIR = "results/debug_koch_eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
THRESHOLD = 0.5 
# ==========================


transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def main():
    print("Loading test data...")

    test_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTest.txt",
        IMAGES_ROOT
    )

    test_dataset = LFWSiameseDataset(test_pairs, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test samples: {len(test_dataset)}")

    print("Loading model...")
    model = SiameseKoch().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_scores = []

    print("\nRunning inference...")

    with torch.no_grad():
        for i, (img1, img2, label) in enumerate(test_loader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)

            outputs = model(img1, img2).squeeze().cpu().numpy()

            all_scores.extend(outputs)
            all_labels.extend(label.numpy())

            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Accuracy
    preds = (all_scores > THRESHOLD).astype(int)
    acc = accuracy_score(all_labels, preds)

    print(f"\n✅ Accuracy: {acc:.4f}")

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    print(f"✅ AUC: {roc_auc:.4f}")

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"))
    plt.close()

    # Save raw data
    np.save(os.path.join(OUTPUT_DIR, "scores.npy"), all_scores)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), all_labels)

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("=== EVALUATION RESULTS ===\n\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"Threshold: {THRESHOLD}\n")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()