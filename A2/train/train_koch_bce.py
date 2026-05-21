# train/train_koch_bce.py

import sys, os, time, random, csv
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pairs_parser import parse_pairs_file
from utils.identity_split import split_pairs_by_identity
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch

# ========= CONFIG =========
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
# ==========================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    set_seed(SEED)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"results/experiment1_loss/koch_bce/{run_id}"
    ensure_dir(OUTPUT_DIR)
    log_path = os.path.join(OUTPUT_DIR, "logs.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Loading data...")

    all_train_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTrain.txt",
        IMAGES_ROOT
    )

    test_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTest.txt",
        IMAGES_ROOT
    )

    train_pairs, val_pairs, train_ids, val_ids, discarded_pairs = split_pairs_by_identity(
        all_train_pairs,
        val_ratio=VAL_RATIO,
        seed=SEED
    )

    log(f"Original train pairs: {len(all_train_pairs)}")
    log(f"Train pairs after identity split: {len(train_pairs)}")
    log(f"Validation pairs after identity split: {len(val_pairs)}")
    log(f"Discarded mixed pairs: {len(discarded_pairs)}")
    log(f"Train identities: {len(train_ids)}")
    log(f"Validation identities: {len(val_ids)}")
    log(f"Test pairs (untouched): {len(test_pairs)}")
    log(f"Config: bs={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, seed={SEED}")

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    train_ds = LFWSiameseDataset(train_pairs, transform=transform)
    val_ds   = LFWSiameseDataset(val_pairs, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SiameseKoch().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_path = os.path.join(OUTPUT_DIR, "model_best.pth")

    log("\nStart training (Koch BCE)...")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ---- TRAIN ----
        model.train()
        total = 0.0

        for b, (img1, img2, label) in enumerate(train_loader):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            label = label.float().to(DEVICE)

            optimizer.zero_grad()

            out = model(img1, img2).squeeze()
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            total += loss.item()

            if b % 10 == 0:
                log(
                    f"[TRAIN] epoch {epoch}/{EPOCHS} "
                    f"batch {b}/{len(train_loader)} "
                    f"loss {loss.item():.4f}"
                )

        train_loss = total / len(train_loader)
        train_losses.append(train_loss)

        # ---- VALIDATION ----
        model.eval()
        total = 0.0

        with torch.no_grad():
            for (img1, img2, label) in val_loader:
                img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                label = label.float().to(DEVICE)

                out = model(img1, img2).squeeze()
                loss = criterion(out, label)

                total += loss.item()

        val_loss = total / len(val_loader)
        val_losses.append(val_loss)

        log(
            f"\nEpoch {epoch}/{EPOCHS} DONE | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={time.time()-t0:.1f}s\n"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "seed": SEED,
                    "val_ratio": VAL_RATIO,
                    "split_type": "identity_level",
                    "original_train_pairs": len(all_train_pairs),
                    "train_pairs": len(train_pairs),
                    "val_pairs": len(val_pairs),
                    "discarded_mixed_pairs": len(discarded_pairs),
                    "train_identities": len(train_ids),
                    "validation_identities": len(val_ids)
                }
            }, best_path)

            log(
                f"✅ Saved best checkpoint: {best_path} "
                f"(best_val={best_val:.4f})"
            )

        # losses.csv
        csv_path = os.path.join(OUTPUT_DIR, "losses.csv")

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)

            w.writerow(["epoch", "train_loss", "val_loss"])

            for i in range(len(train_losses)):
                w.writerow([i+1, train_losses[i], val_losses[i]])

    # plot
    plt.figure()

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Koch BCE Loss Curves")

    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
    plt.close()

    # summary
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXP1: KOCH BCE TRAINING ===\n")
        f.write(f"run_id: {run_id}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"epochs: {EPOCHS}\n")
        f.write(f"lr: {LR}\n")
        f.write(f"seed: {SEED}\n")
        f.write("split_type: identity_level\n")
        f.write(f"original_train_pairs: {len(all_train_pairs)}\n")
        f.write(f"train_pairs: {len(train_pairs)}\n")
        f.write(f"val_pairs: {len(val_pairs)}\n")
        f.write(f"discarded_mixed_pairs: {len(discarded_pairs)}\n")
        f.write(f"train_identities: {len(train_ids)}\n")
        f.write(f"validation_identities: {len(val_ids)}\n")
        f.write(f"best_val_loss: {best_val:.6f}\n")
        f.write(f"best_ckpt: {best_path}\n")

    log(f"\n✅ Done. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()