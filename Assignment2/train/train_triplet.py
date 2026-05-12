# train_triplet.py
import os
import csv
import random
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from models.siamese_koch import SiameseKoch
from datasets.lfw_identities import LFWIdentityDataset, load_train_identities_from_pairs
from losses.triplet_loss import TripletLoss
from losses.mining import semi_hard_triplets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PKBatchSampler(Sampler):
    """
    Samples batches with P identities and K images per identity => batch size = P*K.
    Guarantees positives exist inside the batch.
    """
    def __init__(self, labels, P=16, K=4, seed=42):
        self.labels = labels
        self.P = P
        self.K = K
        self.seed = seed

        # build index list per identity
        self.id_to_indices = {}
        for idx, y in enumerate(labels):
            self.id_to_indices.setdefault(int(y), []).append(idx)

        self.all_ids = list(self.id_to_indices.keys())

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            # choose P identities that have at least 2 images (so we can pick positives)
            valid_ids = [i for i in self.all_ids if len(self.id_to_indices[i]) >= 2]
            if len(valid_ids) < self.P:
                raise RuntimeError("Not enough identities with >=2 images to form a batch.")

            batch_ids = rng.sample(valid_ids, self.P)
            batch = []
            for cid in batch_ids:
                inds = self.id_to_indices[cid]
                if len(inds) >= self.K:
                    chosen = rng.sample(inds, self.K)
                else:
                    # if fewer than K images, sample with replacement
                    chosen = [rng.choice(inds) for _ in range(self.K)]
                batch.extend(chosen)

            yield batch

    def __len__(self):
        # infinite sampler; DataLoader doesn't really use __len__ here
        return 10**9


@torch.no_grad()
def random_triplets_from_batch(labels):
    """
    Random triplets inside a batch: for each anchor pick random positive and random negative.
    labels: [B] int
    returns list of (a,p,n)
    """
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
        triplets.append((a, p, n))
    return triplets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="semihard", choices=["semihard", "random"],
                        help="semihard = required by assignment; random = ablation only")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--P", type=int, default=16)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir("checkpoints")
    ensure_dir("results")

    # paths
    PAIRS_TRAIN = "data/pairsDevTrain.txt"
    LFW_ROOT = "data/lfw2"

    # identities only from TRAIN split (avoid leakage)
    train_ids = load_train_identities_from_pairs(PAIRS_TRAIN)

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    ds = LFWIdentityDataset(lfw_root=LFW_ROOT, transform=transform, identities=train_ids)

    # build labels list for sampler
    labels_list = [y for _, y in ds.samples]  # identity ids (ints)

    sampler = PKBatchSampler(labels_list, P=args.P, K=args.K, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    model = SiameseKoch().to(DEVICE)
    criterion = TripletLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = f"checkpoints/koch_triplet_{args.mode}_m{args.margin}_best.pt"
    loss_csv = f"results/koch_triplet_{args.mode}_m{args.margin}_losses.csv"

    best_epoch_loss = float("inf")
    history = []

    # How many batches per epoch?
    # We'll define a fixed number to make compute budget consistent.
    steps_per_epoch = 200  # you can keep constant for fairness across runs

    print(f"Triplet TRAIN mode={args.mode} margin={args.margin} | batch={args.P*args.K} | steps/epoch={steps_per_epoch}")
    print(f"Checkpoint -> {ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        used_batches = 0
        used_triplets = 0

        for step, (imgs, y) in enumerate(loader):
            if step >= steps_per_epoch:
                break

            imgs = imgs.to(DEVICE)
            y = y.to(DEVICE)

            emb = model.embed(imgs)  # [B, D]

            if args.mode == "semihard":
                triplets = semi_hard_triplets(embeddings=emb, labels=y, margin=args.margin)
            else:
                triplets = random_triplets_from_batch(y)

            if len(triplets) == 0:
                continue

            a_idx = torch.tensor([t[0] for t in triplets], device=DEVICE)
            p_idx = torch.tensor([t[1] for t in triplets], device=DEVICE)
            n_idx = torch.tensor([t[2] for t in triplets], device=DEVICE)

            anchor = emb[a_idx]
            positive = emb[p_idx]
            negative = emb[n_idx]

            loss = criterion(anchor, positive, negative)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            used_batches += 1
            used_triplets += len(triplets)

            if step % 50 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{steps_per_epoch}] "
                      f"Loss={loss.item():.4f} Triplets={len(triplets)}")

        avg_loss = running / max(1, used_batches)
        print(f"✅ Epoch [{epoch}/{args.epochs}] DONE | avg_loss={avg_loss:.4f} "
              f"| batches_used={used_batches} | triplets_used={used_triplets}")

        history.append((epoch, avg_loss, used_batches, used_triplets))

        # save best (lowest epoch avg loss)
        if avg_loss < best_epoch_loss:
            best_epoch_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "avg_loss": avg_loss,
                "mode": args.mode,
                "margin": args.margin,
                "P": args.P,
                "K": args.K,
                "steps_per_epoch": steps_per_epoch,
                "seed": args.seed
            }, ckpt_path)
            print(f"💾 Saved best checkpoint: {ckpt_path} (avg_loss={avg_loss:.4f})")

    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "avg_loss", "batches_used", "triplets_used"])
        w.writerows(history)

    print(f"\n✅ Wrote triplet loss log to {loss_csv}")
    print(f"✅ Best epoch avg loss: {best_epoch_loss:.4f}")


if __name__ == "__main__":
    main()