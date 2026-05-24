sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models as tvm

from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pairs_parser import parse_pairs_file
from datasets.lfw_dataset import LFWSiameseDataset
from models.siamese_koch import SiameseKoch
from models.backbones import KochBackbone, ResNet18Backbone, MetricModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

PAIRS_TEST = "data/lfwa/pairsDevTest.txt"
OUT_DIR = "results/summary"
os.makedirs(OUT_DIR, exist_ok=True)

def dataloader_from_pairs(pairs_file):
    pairs = parse_pairs_file(pairs_file, IMAGES_ROOT)
    ds = LFWSiameseDataset(pairs, transform=transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

@torch.no_grad()
def scores_koch_bce(model, loader):
    labels, scores = [], []
    model.eval()
    for img1, img2, lab in loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        s = model(img1, img2).squeeze(1).detach().cpu().numpy()
        scores.extend(s)
        labels.extend(lab.numpy())
    return np.array(labels).astype(int), np.array(scores).astype(float)

@torch.no_grad()
def scores_l2_metric(model, loader):
    labels, scores = [], []
    model.eval()
    for img1, img2, lab in loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        e1 = model.embed(img1)
        e2 = model.embed(img2)
        dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-9)
        s = (-dist).detach().cpu().numpy()
        scores.extend(s)
        labels.extend(lab.numpy())
    return np.array(labels).astype(int), np.array(scores).astype(float)

@torch.no_grad()
def scores_frozen_cosine(model, loader):
    labels, scores = [], []
    model.eval()
    for img1, img2, lab in loader:
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        e1 = model(img1)
        e2 = model(img2)
        e1 = F.normalize(e1, p=2, dim=1)
        e2 = F.normalize(e2, p=2, dim=1)
        s = (e1 * e2).sum(dim=1).detach().cpu().numpy()
        scores.extend(s)
        labels.extend(lab.numpy())
    return np.array(labels).astype(int), np.array(scores).astype(float)

def plot_one(ax, labels, scores, name):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    return roc_auc

def main():
    loader = dataloader_from_pairs(PAIRS_TEST)

    # ====== paths ======
    CKPT_KOCH_BCE = "results/debug_koch/model.pth"
    CKPT_CONTRASTIVE = "results/experiment1_loss/contrastive_m0.2/20260518_222913/model_best.pth"
    CKPT_TRIPLET_RANDOM = "results/experiment1_loss/triplet_random_m0.2/20260519_130331/model_best.pth"
    CKPT_TRIPLET_SEMIHARD = "results/experiment1_loss/triplet_semihard_m0.2/20260519_143300/model_best.pth"
    CKPT_EXP2_KOCH = "results/experiment2_backbone/koch_triplet_semihard_m0.2/emb128_20260519_152115/model_best.pth"
    CKPT_EXP2_RESNET = "results/experiment2_backbone/resnet18_triplet_semihard_m0.2/emb128_20260519_154440/model_best.pth"
    # ============================================

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)

    # 1) Koch BCE
    koch_bce = SiameseKoch().to(DEVICE)
    koch_bce.load_state_dict(torch.load(CKPT_KOCH_BCE, map_location=DEVICE), strict=True)
    y, s = scores_koch_bce(koch_bce, loader)
    plot_one(ax, y, s, "E1 Koch BCE")

    # 2) Contrastive best
    m = torch.load(CKPT_CONTRASTIVE, map_location=DEVICE)["model_state"]

    contrastive = SiameseKoch().to(DEVICE)
    contrastive.load_state_dict(m, strict=True)

    y, dists = [], []
    contrastive.eval()
    with torch.no_grad():
        for img1, img2, lab in loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            e1 = contrastive.embed(img1)
            e2 = contrastive.embed(img2)

            dist = torch.sqrt(torch.sum((e1 - e2) ** 2, dim=1) + 1e-9)
            s = (-dist).cpu().numpy()

            y.extend(lab.numpy())
            dists.extend(s)

    y, s = np.array(y), np.array(dists)
    plot_one(ax, y, s, "E1 Contrastive (m=0.2)")

    # 3) Triplet random
    m = torch.load(CKPT_TRIPLET_RANDOM, map_location=DEVICE)["model_state"]    
    trip_r = SiameseKoch().to(DEVICE)
    trip_r.load_state_dict(m, strict=True)
    y, s = scores_l2_metric(trip_r, loader)
    plot_one(ax, y, s, "E1 Triplet random (m=0.2)")
    
    # 4) Triplet semihard
    m = torch.load(CKPT_TRIPLET_SEMIHARD, map_location=DEVICE)["model_state"]
    trip_s = SiameseKoch().to(DEVICE)
    trip_s.load_state_dict(m, strict=True)
    y, s = scores_l2_metric(trip_s, loader)
    plot_one(ax, y, s, "E1 Triplet semihard (m=0.2)")

    # 5) Exp2 Koch matched (fc1024 emb128)
    m = torch.load(CKPT_EXP2_KOCH, map_location=DEVICE)["model_state"]
    exp2k = MetricModel(KochBackbone(fc_units=1024, embedding_dim=128)).to(DEVICE)
    exp2k.load_state_dict(m, strict=True)
    y, s = scores_l2_metric(exp2k, loader)
    plot_one(ax, y, s, "E2 Koch matched")

    # 6) Exp2 ResNet scratch
    m = torch.load(CKPT_EXP2_RESNET, map_location=DEVICE)["model_state"]
    exp2r = MetricModel(ResNet18Backbone(embedding_dim=128)).to(DEVICE)
    exp2r.load_state_dict(m, strict=True)
    y, s = scores_l2_metric(exp2r, loader)
    plot_one(ax, y, s, "E2 ResNet18 scratch")

    # 7) Exp3 Frozen pretrained cosine
    resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])

    class Frozen(torch.nn.Module):
        def __init__(self, feat): super().__init__(); self.feat = feat
        def forward(self, x): return self.feat(x).view(x.size(0), -1)

    frozen = Frozen(backbone).to(DEVICE)

    for p in frozen.parameters(): p.requires_grad = False
    
    y, s = scores_frozen_cosine(frozen, loader)

    plot_one(ax, y, s, "E3 Frozen ResNet18 (cosine)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models (pairsDevTest)")
    ax.legend(fontsize=8)

    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "roc_all_models.png")

    fig.savefig(out_path, dpi=200)

    plt.close(fig)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()