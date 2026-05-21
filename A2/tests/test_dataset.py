import sys
import os
import time
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch


# ========= CONFIG =========
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
# ==========================


# ✅ פונקציה לפיצול train/validation
def split_train_val(pairs, val_ratio=0.2, seed=42):
    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_ratio))

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    return train_pairs, val_pairs


# transforms (כמו Koch)
transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def main():
    print("Loading data...")

    # ✅ טוענים את TRAIN המקורי
    all_train_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTrain.txt",
        IMAGES_ROOT
    )

    # ✅ טוענים TEST (לא נוגעים בו!)
    test_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTest.txt",
        IMAGES_ROOT
    )

    # ✅ פיצול נכון
    train_pairs, val_pairs = split_train_val(all_train_pairs)

    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}\n")

    # ✅ datasets
    train_dataset = LFWSiameseDataset(train_pairs, transform=transform)
    val_dataset   = LFWSiameseDataset(val_pairs, transform=transform)

    # ✅ loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Building model...")
    model = SiameseKoch().to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    print("\nStart training...\n")

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = 0

        # ===== TRAIN =====
        for img1, img2, label in train_loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            label = label.float().to(DEVICE)

            optimizer.zero_grad()

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== VALIDATION =====
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                label = label.float().to(DEVICE)

                outputs = model(img1, img2).squeeze()
                loss = criterion(outputs, label)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Time: {time.time() - t0:.2f}s\n")

    print("✅ Training finished!")

    return train_losses, val_losses


if __name__ == "__main__":
    main()