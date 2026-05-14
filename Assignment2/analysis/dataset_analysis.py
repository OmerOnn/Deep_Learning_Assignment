import os
from collections import Counter
import matplotlib.pyplot as plt

LFW_ROOT = "data/lfw2"
PAIRS_TRAIN = "data/pairsDevTrain.txt"
PAIRS_TEST = "data/pairsDevTest.txt"


def count_identities_in_pairs(pairs_file):
    ids = set()
    with open(pairs_file, "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split()
            ids.add(parts[0])
    return ids


def count_images_per_identity(lfw_root):
    counts = {}
    for name in os.listdir(lfw_root):
        person_dir = os.path.join(lfw_root, name)
        if os.path.isdir(person_dir):
            counts[name] = len([
                f for f in os.listdir(person_dir)
                if f.endswith(".jpg")
            ])
    return counts


def main():
    train_ids = count_identities_in_pairs(PAIRS_TRAIN)
    test_ids = count_identities_in_pairs(PAIRS_TEST)

    print(f"Train identities: {len(train_ids)}")
    print(f"Test identities: {len(test_ids)}")
    print(f"Overlap: {len(train_ids & test_ids)}")

    img_counts = count_images_per_identity(LFW_ROOT)

    train_imgs = [img_counts[i] for i in train_ids if i in img_counts]
    test_imgs = [img_counts[i] for i in test_ids if i in img_counts]

    print(f"Avg images per identity (train): {sum(train_imgs)/len(train_imgs):.2f}")
    print(f"Avg images per identity (test): {sum(test_imgs)/len(test_imgs):.2f}")

    plt.figure()
    plt.hist(train_imgs, bins=20)
    plt.title("Images per identity (Train)")
    plt.xlabel("Images")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("train_identity_image_hist.png")

    plt.figure()
    plt.hist(test_imgs, bins=20)
    plt.title("Images per identity (Test)")
    plt.xlabel("Images")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("test_identity_image_hist.png")

    print("Saved histograms.")


if __name__ == "__main__":
    main()