import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import random
import csv
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.utils.data import DataLoader

from models.backbones import KochBackbone, ResNet18Backbone, MetricModel
from datasets.lfw_identities import LFWIdentityDataset, load_train_identities_from_pairs
from datasets.pk_sampler import PKBatchSampler
from losses.triplet_loss import TripletLoss
from losses.mining import pairwise_l2, semi_hard_triplets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="koch", choices=["koch","resnet18"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)   # fixed from Exp1 best
    parser.add_argument("--P", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--koch_fc", type=int, default=1024)    # match params
    args = parser.parse_args()

    set_seed(args.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT = f"results/experiment2_backbone/{args.backbone}_triplet_semihard_m{args.margin}/emb{args.emb_dim}_{run_id}"
    os.makedirs(OUT, exist_ok=True)

    def log(msg):
        print(msg)
        with open(os.path.join(OUT, "logs.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"Config: backbone={args.backbone}, epochs={args.epochs}, lr={args.lr}, margin={args.margin}, "
        f"P={args.P}, K={args.K}, seed={args.seed}, emb_dim={args.emb_dim}, koch_fc={args.koch_fc}")

    transform = transforms.Compose([
        transforms.Resize((105,105)),
        transforms.ToTensor()
    ])

    train_ids = load_train_identities_from_pairs("data/lfwa/pairsDevTrain.txt")
    ds = LFWIdentityDataset(IMAGES_ROOT, train_ids, transform=transform, min_images_per_id=2)
    sampler = PKBatchSampler(ds.labels, P=args.P, K=args.K, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler)

    # backbone choice
    if args.backbone == "koch":
        backbone = KochBackbone(fc_units=args.koch_fc, embedding_dim=args.emb_dim)
    else:
        backbone = ResNet18Backbone(embedding_dim=args.emb_dim)

    model = MetricModel(backbone).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripletLoss(margin=args.margin)

    history = []
    best_loss = float("inf")
    best_path = os.path.join(OUT, "model_best.pth")

    log("Start training (Exp2 backbone)...")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for b, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            emb = model.embed(imgs)                 # [B,D]
            dist_mat = pairwise_l2(emb)             # [B,B] with grad
            dist_detached = dist_mat.detach()       # mining only

            triplets = semi_hard_triplets(labels, dist_detached, margin=args.margin)
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
                log(f"[{args.backbone.upper()}] epoch {epoch}/{args.epochs} batch {b}/{len(loader)} "
                    f"loss {loss.item():.4f} triplets={len(triplets)}")

        avg = total_loss / max(1, n_batches)
        history.append((epoch, avg))
        log(f"\nEpoch {epoch}/{args.epochs} DONE | avg_loss={avg:.4f} | time={time.time()-t0:.1f}s\n")

        if avg < best_loss:
            best_loss = avg
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "loss": avg, "config": vars(args)}, best_path)
            log(f"Saved best checkpoint: {best_path} (best_loss={best_loss:.4f})")

        with open(os.path.join(OUT, "losses.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","avg_triplet_loss"])
            w.writerows(history)

    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXP2 BACKBONE TRAINING ===\n")
        for k,v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"best_loss: {best_loss:.6f}\n")
        f.write(f"checkpoint: {best_path}\n")

    log(f"Done. Outputs in: {OUT}")

if __name__ == "__main__":
    main()