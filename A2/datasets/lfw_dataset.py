import os
from PIL import Image
from torch.utils.data import Dataset


class LFWSiameseDataset(Dataset):
    def __init__(self, pairs, transform=None):
        """
        pairs: list of (img1_path, img2_path, label)
        transform: torchvision transforms
        """
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
