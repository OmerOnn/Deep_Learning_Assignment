import os
from PIL import Image
from torch.utils.data import Dataset

def load_train_identities_from_pairs(pairs_file):
    """
    קורא pairsDevTrain ומחזיר סט זהויות שמופיעות בו.
    חשוב: כך אנחנו לא 'נוגעים' בזהויות של TEST.
    """
    ids = set()
    with open(pairs_file, "r") as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            ids.add(parts[0])
        elif len(parts) == 4:
            ids.add(parts[0])
            ids.add(parts[2])
    return ids

class LFWIdentityDataset(Dataset):
    """
    מחזיר (image_tensor, label_int) כאשר label הוא אינדקס זהות.
    """
    def __init__(self, images_root, identities, transform=None, min_images_per_id=2):
        self.images_root = images_root
        self.transform = transform

        self.id_to_images = {}
        for name in sorted(list(identities)):
            person_dir = os.path.join(images_root, name)
            if not os.path.isdir(person_dir):
                continue
            imgs = [os.path.join(person_dir, x) for x in os.listdir(person_dir) if x.lower().endswith((".jpg", ".png"))]
            if len(imgs) >= min_images_per_id:
                self.id_to_images[name] = sorted(imgs)

        self.id_names = sorted(self.id_to_images.keys())
        self.id_to_label = {n:i for i,n in enumerate(self.id_names)}

        # flatten
        self.samples = []
        for n in self.id_names:
            lab = self.id_to_label[n]
            for p in self.id_to_images[n]:
                self.samples.append((p, lab))

        self.labels = [lab for _, lab in self.samples]  # שימושי ל-sampler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lab = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lab
