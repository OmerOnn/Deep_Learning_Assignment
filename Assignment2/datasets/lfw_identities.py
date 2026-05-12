# datasets/lfw_identities.py
import os
from PIL import Image
from torch.utils.data import Dataset


def load_train_identities_from_pairs(pairs_file: str):
    """
    Reads pairsDevTrain.txt and returns sorted unique identity folder names
    that appear in the TRAIN split (first column of each line).
    For negative pairs lines: identities are in col0 and col2.
    """
    ids = set()
    with open(pairs_file, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            # positive: name, i1, i2
            ids.add(parts[0])
        elif len(parts) == 4:
            # negative: name1, i1, name2, i2
            ids.add(parts[0])
            ids.add(parts[2])

    return sorted(ids)


class LFWIdentityDataset(Dataset):
    """
    Dataset of individual images for Triplet training.
    Returns: (image_tensor, identity_id_int)

    - lfw_root: path to lfw folder (e.g. data/lfw2)
    - identities: list of identity folder names to include (recommended: train identities only)
    """
    def __init__(self, lfw_root: str, transform=None, identities=None):
        self.lfw_root = lfw_root
        self.transform = transform

        if identities is None:
            # fallback: all folders under lfw_root
            identities = sorted([
                d for d in os.listdir(lfw_root)
                if os.path.isdir(os.path.join(lfw_root, d))
            ])

        self.identities = identities
        self.id_to_idx = {name: i for i, name in enumerate(self.identities)}

        self.samples = []
        for name in self.identities:
            folder = os.path.join(lfw_root, name)
            if not os.path.isdir(folder):
                continue

            for fn in os.listdir(folder):
                if fn.lower().endswith(".jpg"):
                    path = os.path.join(folder, fn)
                    self.samples.append((path, self.id_to_idx[name]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under lfw_root={lfw_root}. "
                f"Check the path and that folders contain .jpg files."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y
