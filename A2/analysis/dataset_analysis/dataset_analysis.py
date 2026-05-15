import sys
import os
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.pairs_parser import parse_pairs_file

IMAGES_ROOT = "data/lfwa/aligned_images"


def analyze_dataset(images_root):
    identity_counts = defaultdict(int)
    total_images = 0

    for person in os.listdir(images_root):
        person_path = os.path.join(images_root, person)

        if not os.path.isdir(person_path):
            continue

        images = os.listdir(person_path)
        count = len(images)

        identity_counts[person] = count
        total_images += count

    num_identities = len(identity_counts)
    all_counts = list(identity_counts.values())

    avg_images = np.mean(all_counts)
    std_images = np.std(all_counts)

    return {
        "identity_counts": identity_counts,
        "num_identities": num_identities,
        "total_images": total_images,
        "all_counts": all_counts,
        "avg": avg_images,
        "std": std_images,
        "min": np.min(all_counts),
        "max": np.max(all_counts),
        "median": np.median(all_counts),
    }


def create_plots(stats):
    counts = stats["all_counts"]

    os.makedirs("analysis/plots", exist_ok=True)

    # ✅ 1. Histogram רגיל
    plt.figure()
    plt.hist(counts, bins=50)
    plt.title("Images per Identity Distribution")
    plt.xlabel("Number of images")
    plt.ylabel("Frequency")
    plt.savefig("analysis/plots/hist_regular.png")
    plt.close()

    # ✅ 2. Log scale
    plt.figure()
    plt.hist(counts, bins=50)
    plt.yscale("log")
    plt.title("Images per Identity Distribution (log)")
    plt.xlabel("Number of images")
    plt.ylabel("Frequency (log)")
    plt.savefig("analysis/plots/hist_log.png")
    plt.close()

    # ✅ 3. Zoom (<=20)
    plt.figure()
    small = [x for x in counts if x <= 20]
    plt.hist(small, bins=20)
    plt.title("Images per Identity (<=20 images)")
    plt.xlabel("Number of images")
    plt.ylabel("Frequency")
    plt.savefig("analysis/plots/hist_zoom.png")
    plt.close()

    # ✅ 4. CDF (מאוד חזק לדו"ח)
    plt.figure()
    sorted_counts = np.sort(counts)
    cdf = np.arange(len(sorted_counts)) / len(sorted_counts)

    plt.plot(sorted_counts, cdf)
    plt.title("CDF of Images per Identity")
    plt.xlabel("Number of images")
    plt.ylabel("Cumulative probability")
    plt.savefig("analysis/plots/cdf.png")
    plt.close()

    # ✅ 5. Top identities
    counter = Counter(counts)
    most_common = counter.most_common(10)

    values = [v[0] for v in most_common]
    freqs = [v[1] for v in most_common]

    plt.figure()
    plt.bar(values, freqs)
    plt.title("Most Common Image Counts per Identity")
    plt.xlabel("Images per identity")
    plt.ylabel("Number of identities")
    plt.savefig("analysis/plots/top_counts.png")
    plt.close()


def save_txt(stats, train_pairs, test_pairs):
    with open("analysis/dataset_summary.txt", "w", encoding="utf-8") as f:

        f.write("===== DATASET SUMMARY =====\n\n")

        f.write(f"Total number of identities: {stats['num_identities']}\n")
        f.write(f"Total number of images: {stats['total_images']}\n")
        f.write(f"Average images per identity: {stats['avg']:.2f}\n")
        f.write(f"Std images per identity: {stats['std']:.2f}\n")
        f.write(f"Median images per identity: {stats['median']}\n")
        f.write(f"Min images per identity: {stats['min']}\n")
        f.write(f"Max images per identity: {stats['max']}\n\n")

        f.write("===== PAIRS =====\n\n")
        f.write(f"Train pairs: {len(train_pairs)}\n")
        f.write(f"Test pairs: {len(test_pairs)}\n\n")

        total_pairs = len(train_pairs) + len(test_pairs)
        f.write(f"Total pairs: {total_pairs}\n\n")

        f.write("===== CLASS BALANCE =====\n\n")

        train_labels = [p[2] for p in train_pairs]
        test_labels = [p[2] for p in test_pairs]

        f.write(f"Train positives: {sum(train_labels)}\n")
        f.write(f"Train negatives: {len(train_labels) - sum(train_labels)}\n\n")

        f.write(f"Test positives: {sum(test_labels)}\n")
        f.write(f"Test negatives: {len(test_labels) - sum(test_labels)}\n\n")

        f.write("===== NOTES =====\n")
        f.write("Dataset is highly imbalanced.\n")
        f.write("Most identities have very few images.\n")
        f.write("Long tail distribution exists.\n")


def main():
    print("Loading pairs...")

    train_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTrain.txt",
        IMAGES_ROOT
    )

    test_pairs = parse_pairs_file(
        "data/lfwa/pairsDevTest.txt",
        IMAGES_ROOT
    )

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    print("\nAnalyzing dataset...")
    stats = analyze_dataset(IMAGES_ROOT)

    print("\nCreating plots...")
    create_plots(stats)

    print("Saving summary...")
    save_txt(stats, train_pairs, test_pairs)

    print("\n✅ DONE!")
    print("Check folder: analysis/")


if __name__ == "__main__":
    main()
