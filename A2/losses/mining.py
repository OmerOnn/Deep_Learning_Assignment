import torch

def pairwise_l2(emb, eps=1e-9):
    sq = (emb ** 2).sum(dim=1, keepdim=True)             # [B,1]
    dist2 = sq + sq.t() - 2.0 * (emb @ emb.t())          # [B,B]
    dist2 = torch.clamp(dist2, min=0.0)
    dist = torch.sqrt(dist2 + eps)
    return dist

@torch.no_grad()
def semi_hard_triplets(labels, dist_detached, margin):
    B = labels.shape[0]
    triplets = []

    for a in range(B):
        pos = torch.where(labels == labels[a])[0]
        neg = torch.where(labels != labels[a])[0]
        pos = pos[pos != a]
        if pos.numel() == 0 or neg.numel() == 0:
            continue

        p = pos[torch.randint(0, pos.numel(), (1,)).item()].item()
        d_ap = dist_detached[a, p]
        d_an_all = dist_detached[a, neg]
        mask = (d_an_all > d_ap) & (d_an_all < d_ap + margin)
        candidates = neg[mask]

        if candidates.numel() > 0:
            cand_d = dist_detached[a, candidates]
            n = candidates[torch.argmin(cand_d)].item()
        else:
            cand_d = dist_detached[a, neg]
            n = neg[torch.argmin(cand_d)].item()

        triplets.append((a, p, n))

    return triplets