import torch

@torch.no_grad()
def semi_hard_triplets(embeddings, labels, margin):
    """
    embeddings: [B, D]
    labels: [B] int identity label
    returns indices (a,p,n) for semi-hard triplets within the batch:
      d(ap) < d(an) < d(ap)+margin
    If none exist, falls back to hard negatives (closest negative) for each anchor if possible.
    """
    device = embeddings.device
    B = embeddings.size(0)

    # pairwise distances [B,B]
    dist = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    for a in range(B):
        same = (labels == labels[a])
        diff = ~same
        same[a] = False  # exclude self

        pos_idx = torch.where(same)[0]
        neg_idx = torch.where(diff)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        for p in pos_idx.tolist():
            d_ap = dist[a, p]
            # semi-hard: d_ap < d_an < d_ap + margin
            d_an_all = dist[a, neg_idx]
            mask = (d_an_all > d_ap) & (d_an_all < d_ap + margin)
            semi = neg_idx[mask]
            if semi.numel() > 0:
                # choose one semi-hard negative (closest among semi-hard)
                n = semi[torch.argmin(dist[a, semi])].item()
                triplets.append((a, p, n))
            else:
                # fallback: hardest negative (closest)
                n = neg_idx[torch.argmin(d_an_all)].item()
                triplets.append((a, p, n))

    return triplets
