sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os, json, random
import sys

from datasets.lfw_identities import load_train_identities_from_pairs

IMAGES_ROOT = "data/lfwa/aligned_images/lfw2"
OUT_DIR = "results/oneshot"
SEED = 42
EPISODES_PER_N = 300

os.makedirs(OUT_DIR, exist_ok=True)

def list_images(identity):
    p = os.path.join(IMAGES_ROOT, identity)
    imgs = [os.path.join(p, x) for x in os.listdir(p) if x.lower().endswith(".jpg")]
    return sorted(imgs)

def main():
    random.seed(SEED)
    ids = sorted(list(load_train_identities_from_pairs("data/lfwa/pairsDevTrain.txt")))
    ids = [i for i in ids if len(list_images(i)) >= 2]

    episodes = {}
    for N in [2,5,20]:
        eps = []
        for _ in range(EPISODES_PER_N):
            target = random.choice(ids)
            distractors = random.sample([x for x in ids if x != target], N-1)
            candidates = [target] + distractors

            random.shuffle(candidates)

            # query + support for each candidate
            query_img = random.choice(list_images(target))
            support = {}
            for cid in candidates:
                imgs = list_images(cid)
                if cid == target:
                    imgs2 = [x for x in imgs if x != query_img]
                    support[cid] = random.choice(imgs2) if imgs2 else random.choice(imgs)
                else:
                    support[cid] = random.choice(imgs)

            eps.append({
                "N": N,
                "target": target,
                "query": query_img,
                "candidates": candidates,
                "support": support
            })
        episodes[str(N)] = eps

    out_path = os.path.join(OUT_DIR, "episodes_seed42.json")
    
    with open(out_path, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Saved episodes: {out_path}")

if __name__ == "__main__":
    main()
