# train/train_triplet.py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os, time, random, csv
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from datetime import datetime
from models.siamese_koch import SiameseKoch
from datasets.lfw_identities import LFWIdentityDataset
from datasets.pk_sampler import PKBatchSampler
from utils.pairs_parser import parse_pairs_file
from utils.identity_split import split_pairs_by_identity
from losses.triplet_loss import TripletLoss
from losses.mining import pairwise_l2, semi_hard_triplets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def random_triplets_from_batch(labels):
    labels = labels.cpu()
    B = labels.shape[0]

    triplets = []

    for a in range(B):
        same = torch.where(labels == labels[a])[0]
        diff = torch.where(labels != labels[a])[0]

        same = same[same != a]

        if same.numel() == 0 or diff.numel() == 0:
            continue

        p = same[torch.randint(0, same.numel(), (1,)).item()].item()
        n = diff[torch.randint(0, diff.numel(), (1,)).item()].item()

        triplets.append((a,p,n))

    return triplets

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="semihard", choices=["semihard","random"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--P", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = f"results/experiment1_loss/triplet_{args.mode}_m{args.margin}/{run_id}"

    os.makedirs(OUT, exist_ok=True)

    def log(msg):
        print(msg)
        with open(os.path.join(OUT, "logs.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(
        f"Config: mode={args.mode}, epochs={args.epochs}, "
        f"lr={args.lr}, margin={args.margin}, "
        f"P={args.P}, K={args.K}, seed={args.seed}"
    )

    transform = transforms.Compose([
        transforms.Resize((105,105)),
        transforms.ToTensor()
    ])

    all_train_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTrain.txt",
        IMAGES_ROOT
    )

    train_pairs, val_pairs, train_ids, val_ids, discarded_pairs = split_pairs_by_identity(
        all_train_pairs,
        val_ratio=0.2,
        seed=args.seed
    )

    log(f"Train identities: {len(train_ids)}")
    log(f"Validation identities: {len(val_ids)}")
    log(f"Train pairs after identity split: {len(train_pairs)}")
    log(f"Validation pairs after identity split: {len(val_pairs)}")
    log(f"Discarded mixed pairs: {len(discarded_pairs)}")

    ds = LFWIdentityDataset(
        IMAGES_ROOT,
        train_ids,
        transform=transform,
        min_images_per_id=2
    )

    sampler = PKBatchSampler(ds.labels, P=args.P, K=args.K, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler)

    model = SiameseKoch().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripletLoss(margin=args.margin)

    history = []

    best_loss = float("inf")
    best_path = os.path.join(OUT, "model_best.pth")

    log("Start training (Triplet)...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()

        total_loss = 0.0
        n_batches = 0

        for b, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            emb = model.embed(imgs)

            dist_mat = pairwise_l2(emb)

            if args.mode == "random":
                triplets = random_triplets_from_batch(labels)
            else:
                triplets = semi_hard_triplets(labels, dist_mat, margin=args.margin)

            if len(triplets) == 0:
                continue

            a_idx = torch.tensor([t[0] for t in triplets], device=DEVICE)
            p_idx = torch.tensor([t[1] for t in triplets], device=DEVICE)
            n_idx = torch.tensor([t[2] for t in triplets], device=DEVICE)

            d_ap = dist_mat[a_idx, p_idx]
            d_an = dist_mat[a_idx, n_idx]

            loss = criterion(d_ap, d_an)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if b % 10 == 0:
                log(
                    f"[{args.mode.upper()}] epoch {epoch}/{args.epochs} "
                    f"batch {b}/{len(loader)} "
                    f"loss {loss.item():.4f} "
                    f"triplets={len(triplets)}"
                )

        avg = total_loss / max(1, n_batches)

        history.append((epoch, avg))

        log(
            f"\nEpoch {epoch}/{args.epochs} DONE | "
            f"avg_loss={avg:.4f} | "
            f"time={time.time()-t0:.1f}s\n"
        )

        if avg < best_loss:
            best_loss = avg

            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "loss": avg,
                "config": vars(args)
            }, best_path)

            log(
                f"✅ Saved best checkpoint: {best_path} "
                f"(best_loss={best_loss:.4f})"
            )

        with open(os.path.join(OUT, "losses.csv"), "w", newline="") as f:
            w = csv.writer(f)

            w.writerow(["epoch","avg_triplet_loss"])
            w.writerows(history)

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXP1: TRIPLET TRAINING ===\n")

        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")

        f.write("split_type: identity_level\n")
        f.write(f"train_identities: {len(train_ids)}\n")
        f.write(f"validation_identities: {len(val_ids)}\n")
        f.write(f"train_pairs: {len(train_pairs)}\n")
        f.write(f"val_pairs: {len(val_pairs)}\n")
        f.write(f"discarded_pairs: {len(discarded_pairs)}\n")
        f.write(f"best_loss: {best_loss:.6f}\n")
        f.write(f"checkpoint: {best_path}\n")

    log(f"✅ Done. Outputs in: {OUT}")

if __name__ == "__main__":
    main()