import os
import csv
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from datasets.lfw_pairs import LFWDataset
from models.siamese_koch import SiameseKoch
from losses.metric_losses import ContrastiveLoss

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
VAL_FRAC = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAIRS_TRAIN = "data/pairsDevTrain.txt"
LFW_ROOT = "data/lfw2"

MARGIN = 1.0  # נתחיל עם 1.0 ואז נשווה 0.5/1.0/2.0 במסגרת K-trials
CKPT_PATH = f"checkpoints/koch_contrastive_m{MARGIN}_best.pt"
LOSS_CSV = f"results/koch_contrastive_m{MARGIN}_losses.csv"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu", print_every=50):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_batches = len(loader)

    with torch.set_grad_enabled(is_train):
        for i, (img1, img2, label) in enumerate(loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.float().unsqueeze(1).to(device)

            if is_train:
                optimizer.zero_grad()

            emb1 = model.embed(img1)
            emb2 = model.embed(img2)

            loss = criterion(emb1, emb2, label)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if i % print_every == 0:
                phase = "TRAIN" if is_train else "VAL"
                print(f"{phase} | Batch [{i}/{n_batches}] | Loss: {loss.item():.4f}")

    return total_loss / n_batches


def main():
    ensure_dir("checkpoints")
    ensure_dir("results")
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    full_dataset = LFWDataset(PAIRS_TRAIN, LFW_ROOT, transform=transform)
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_FRAC)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SiameseKoch().to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch [{epoch}/{EPOCHS}] (margin={MARGIN}) ===")
        train_loss = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=DEVICE)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None, device=DEVICE)
        print(f"Epoch DONE | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        history.append((epoch, train_loss, val_loss))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "margin": MARGIN,
                "config": {"batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR, "seed": SEED}
            }, CKPT_PATH)
            print(f"✅ Saved best checkpoint to {CKPT_PATH} (val_loss={val_loss:.4f})")

    with open(LOSS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        w.writerows(history)

    print(f"\n✅ Wrote loss curves to {LOSS_CSV}")
    print(f"✅ Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()