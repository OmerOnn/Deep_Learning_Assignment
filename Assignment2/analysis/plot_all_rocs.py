import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

MODELS = [
    ("Koch BCE", "results/koch_bce_test_scores.csv"),
    ("Koch Contrastive", "results/koch_contrastive_m1.0_test_scores.csv"),
    ("Koch Triplet (semihard)", "results/koch_triplet_semihard_m0.2_test_scores.csv"),
]


def main():
    plt.figure(figsize=(8, 6))

    for name, path in MODELS:
        df = pd.read_csv(path)

        # label column (always exists)
        y_true = df["label"]

        # automatically find score column
        score_cols = [
            c for c in df.columns
            if c not in ["label", "pred", "img1_path", "img2_path"]
        ]
        assert len(score_cols) == 1, f"Ambiguous score column in {path}: {score_cols}"
        y_score = df[score_cols[0]]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Face Verification on LFW")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis/results/roc_all_models.png")
    plt.show()



if __name__ == "__main__":
    main()