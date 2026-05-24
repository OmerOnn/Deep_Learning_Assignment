sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
import time
import random
import csv
import matplotlib.pyplot as plt
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

OUTPUT_DIR = "results/debug_koch"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================


def split_train_val(pairs, val_ratio=0.2, seed=42):
    random.seed(seed)
    pairs = pairs.copy()
    random.shuffle(pairs)

    split_idx = int(len(pairs) * (1 - val_ratio))
    return pairs[:split_idx], pairs[split_idx:]


transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def main():
    print("Loading data...")

    all_train_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTrain.txt",
        IMAGES_ROOT
    )

    test_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTest.txt",
        IMAGES_ROOT
    )

    train_pairs, val_pairs = split_train_val(all_train_pairs)

    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}\n")

    train_dataset = LFWSiameseDataset(train_pairs, transform=transform)
    val_dataset   = LFWSiameseDataset(val_pairs, transform=transform)
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

    print("Training finished!")

    # ============================
    # SAVE EVERYTHING
    # ============================

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))

    # Save losses CSV
    csv_path = os.path.join(OUTPUT_DIR, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], val_losses[i]])

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
    plt.close()

    # Save summary TXT
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("=== KOCH BASELINE TRAINING ===\n\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LR}\n\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f}\n")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    

if __name__ == "__main__":
    main()