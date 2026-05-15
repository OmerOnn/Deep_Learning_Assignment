import pandas as pd
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

files = glob.glob("results/*test_scores.csv")

rows = []

for f in files:
    df = pd.read_csv(f)

    if "score" in df.columns:
        scores = df["score"]
    elif "score(-dist)" in df.columns:
        scores = df["score(-dist)"]
    elif "distance" in df.columns:
        scores = -df["distance"]
    else:
        raise ValueError(f"Unknown format in {f}")

    labels = df["label"]

    # ✅ accuracy עם threshold פשוט
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    acc = (preds == labels).mean()

    # ✅ AUC
    auc = roc_auc_score(labels, scores)

    rows.append({
        "file": f.split("/")[-1],
        "accuracy": acc,
        "auc": auc
    })

summary = pd.DataFrame(rows)

if len(summary) == 0:
    print("❌ No files found!")
else:
    summary.sort_values(by="accuracy", ascending=False, inplace=True)

    summary.to_csv("analysis/results/summary_table.csv", index=False)

    print(summary)