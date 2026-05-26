import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import argparse
import torch

from torchvision import transforms
from PIL import Image

from models.siamese_koch import SiameseKoch
from models.backbones import KochBackbone, MetricModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EP_PATH = "results/oneshot/episodes_seed42.json"


transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])


def fix_path(path):
    if os.path.exists(path):
        return path

    path2 = path.replace("data/lfw2/", "data/lfwa/aligned_images/lfw2/")
    if os.path.exists(path2):
        return path2

    path3 = path.replace("data/lfwa/aligned_images/lfw2/", "data/lfwa/aligned_images/")
    if os.path.exists(path3):
        return path3

    return path


def build_model(model_type):
    if model_type == "exp1_triplet":
        return SiameseKoch().to(DEVICE)

    if model_type == "exp2_koch":
        backbone = KochBackbone(fc_units=1024, embedding_dim=128)
        return MetricModel(backbone).to(DEVICE)

    raise ValueError(f"Unknown model_type: {model_type}")


@torch.no_grad()
def embed_image(model, path):
    path = fix_path(path)
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    e = model.embed(x)
    return e.squeeze(0)


def l2(e1, e2):
    return torch.sqrt(torch.sum((e1 - e2) ** 2) + 1e-9)


def evaluate_oneshot(model):
    with open(EP_PATH, "r", encoding="utf-8") as f:
        episodes = json.load(f)

    results = {}

    for n_value in ["2", "5", "20"]:
        correct = 0
        eps = episodes[n_value]

        for ep in eps:
            target = ep["target"]
            query = ep["query"]
            candidates = ep["candidates"]
            support = ep["support"]

            query_embedding = embed_image(model, query)

            best_id = None
            best_distance = float("inf")

            for candidate_id in candidates:
                support_embedding = embed_image(model, support[candidate_id])
                distance = l2(query_embedding, support_embedding).item()

                if distance < best_distance:
                    best_distance = distance
                    best_id = candidate_id

            if best_id == target:
                correct += 1

        acc = correct / len(eps)
        results[n_value] = acc
        print(f"{n_value}-way accuracy: {acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["exp1_triplet", "exp2_koch"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--seed", required=True)

    args = parser.parse_args()

    model = build_model(args.model_type)

    checkpoint = torch.load(args.ckpt, map_location=DEVICE)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    results = evaluate_oneshot(model)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    output = {
        "seed": args.seed,
        "model_type": args.model_type,
        "checkpoint": args.ckpt,
        "results": results
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved one-shot results to: {args.out_json}")


if __name__ == "__main__":
    main()
