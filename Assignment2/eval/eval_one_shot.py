import random
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from datasets.lfw_identities import LFWIdentityDataset, load_train_identities_from_pairs
from models.siamese_koch import SiameseKoch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LFW_ROOT = "data/lfw2"
PAIRS_TRAIN = "data/pairsDevTrain.txt"

CKPT_PATH = "checkpoints/koch_triplet_semihard_m0.2_best.pt"

EMBED_DIM = 128
EPISODES = 300   # מספר ניסויים (אפשר 100–500)
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


@torch.no_grad()
def main(N):
    print(f"\n🔍 Running {N}-way one-shot evaluation")

    transform = transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])

    # load identities from TRAIN split (no leakage)
    identities = load_train_identities_from_pairs(PAIRS_TRAIN)

    ds = LFWIdentityDataset(
        lfw_root=LFW_ROOT,
        identities=identities,
        transform=transform
    )

    # build mapping: identity -> list of indices
    id_to_indices = {}
    for idx, (_, y) in enumerate(ds.samples):
        id_to_indices.setdefault(int(y), []).append(idx)

    valid_ids = [i for i, inds in id_to_indices.items() if len(inds) >= 2]

    # load model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = SiameseKoch(embed_dim=EMBED_DIM).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct = 0

    for epi in range(EPISODES):
        # choose N identities
        chosen_ids = random.sample(valid_ids, N)

        # choose true identity
        true_id = random.choice(chosen_ids)

        # support set: 1 image per identity
        support_imgs = []
        for cid in chosen_ids:
            idx = random.choice(id_to_indices[cid])
            img, _ = ds[idx]
            support_imgs.append(img)

        support_imgs = torch.stack(support_imgs).to(DEVICE)

        # query image: different image of true identity
        q_candidates = id_to_indices[true_id]
        q_idx = random.choice(q_candidates)
        query_img, _ = ds[q_idx]
        query_img = query_img.unsqueeze(0).to(DEVICE)

        # embeddings
        emb_support = model.embed(support_imgs)      # [N, D]
        emb_query = model.embed(query_img)           # [1, D]

        # L2 distances
        dists = torch.norm(emb_support - emb_query, p=2, dim=1)
        pred_idx = torch.argmin(dists).item()

        if chosen_ids[pred_idx] == true_id:
            correct += 1

    acc = correct / EPISODES
    print(f"✅ {N}-way one-shot accuracy: {acc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="N-way (e.g. 2, 5, 20)")
    args = parser.parse_args()

    main(args.N)