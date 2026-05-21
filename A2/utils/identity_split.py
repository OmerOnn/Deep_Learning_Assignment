import os
import random


def get_identity_from_path(image_path: str) -> str:
    """
    Extract identity name from an LFW image path.

    Example:
    data/lfwa/aligned_images/lfw2/George_Bush/George_Bush_0001.jpg
    -> George_Bush
    """
    return os.path.basename(os.path.dirname(image_path))


def get_pair_identities(pair):
    """
    pair format:
    (path1, path2, label)

    Returns:
    (identity1, identity2)
    """
    path1, path2, _ = pair
    id1 = get_identity_from_path(path1)
    id2 = get_identity_from_path(path2)
    return id1, id2


def split_pairs_by_identity(pairs, val_ratio=0.2, seed=42):
    """
    Splits pairs into train/validation by identity, not by pair.

    A pair is kept in train only if both identities belong to train identities.
    A pair is kept in validation only if both identities belong to validation identities.
    Mixed pairs are discarded to prevent identity leakage.

    Returns:
    train_pairs, val_pairs, train_ids, val_ids, discarded_pairs
    """

    # Collect all identities from all pairs
    identities = set()
    for pair in pairs:
        id1, id2 = get_pair_identities(pair)
        identities.add(id1)
        identities.add(id2)

    identities = sorted(list(identities))

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(identities)

    split_idx = int(len(identities) * (1 - val_ratio))

    train_ids = set(identities[:split_idx])
    val_ids = set(identities[split_idx:])

    train_pairs = []
    val_pairs = []
    discarded_pairs = []

    for pair in pairs:
        id1, id2 = get_pair_identities(pair)

        if id1 in train_ids and id2 in train_ids:
            train_pairs.append(pair)

        elif id1 in val_ids and id2 in val_ids:
            val_pairs.append(pair)

        else:
            # One identity is in train and the other is in validation.
            # We discard this pair to avoid leakage.
            discarded_pairs.append(pair)

    return train_pairs, val_pairs, train_ids, val_ids, discarded_pairs