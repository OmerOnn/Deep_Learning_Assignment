import random
from torch.utils.data import Sampler
from collections import defaultdict

class PKBatchSampler(Sampler):
    """
    מחזיר אינדקסים לבאצ'ים בגודל P*K:
    בוחר P זהויות, ומכל אחת K דוגמאות.
    """
    def __init__(self, labels, P=16, K=4, seed=42):
        self.labels = labels
        self.P = P
        self.K = K
        self.seed = seed

        self.label_to_indices = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.label_to_indices[lab].append(idx)

        self.labels_set = sorted(list(self.label_to_indices.keys()))
        self.rng = random.Random(seed)

    def __iter__(self):
        self.rng.shuffle(self.labels_set)
        # generate infinite-like batches; DataLoader will stop by len(self)
        for _ in range(len(self)):
            chosen_labels = self.rng.sample(self.labels_set, k=min(self.P, len(self.labels_set)))
            batch = []
            for lab in chosen_labels:
                inds = self.label_to_indices[lab]
                if len(inds) >= self.K:
                    batch.extend(self.rng.sample(inds, self.K))
                else:
                    # sample with replacement if not enough
                    batch.extend([self.rng.choice(inds) for _ in range(self.K)])
            yield batch

    def __len__(self):
        # מספר באצ'ים "סביר" לאפוק: total_samples / (P*K)
        return max(1, len(self.labels) // (self.P * self.K))