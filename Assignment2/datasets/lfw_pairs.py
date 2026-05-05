import os
from PIL import Image
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    def __init__(self, pairs_file, lfw_root, transform=None):
        self.lfw_root = lfw_root
        self.transform = transform
        self.pairs = []

        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        # שורה ראשונה = מספר identities (לא צריך בפועל)
        for line in lines[1:]:
            parts = line.strip().split()

            if len(parts) == 3:
                # זוג חיובי
                name = parts[0]
                img1 = int(parts[1])
                img2 = int(parts[2])
                self.pairs.append((name, img1, name, img2, 1))

            elif len(parts) == 4:
                # זוג שלילי
                name1 = parts[0]
                img1 = int(parts[1])
                name2 = parts[2]
                img2 = int(parts[3])
                self.pairs.append((name1, img1, name2, img2, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name1, img1_idx, name2, img2_idx, label = self.pairs[idx]

        img1_path = os.path.join(
            self.lfw_root, name1, f"{name1}_{img1_idx:04d}.jpg"
        )
        img2_path = os.path.join(
            self.lfw_root, name2, f"{name2}_{img2_idx:04d}.jpg"
        )

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label