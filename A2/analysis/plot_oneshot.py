import os
import json
import matplotlib.pyplot as plt

DIR = "results/oneshot"

files = [f for f in os.listdir(DIR) if f.endswith("_oneshot.json")]

data = {}

for f in files:
    name = f.replace("_oneshot.json", "")
    with open(os.path.join(DIR, f)) as fp:
        data[name] = json.load(fp)

# X axis
N_values = [2, 5, 20]

plt.figure(figsize=(8,5))

for model, res in data.items():
    y = []
    for n in N_values:
        v = res[str(n)]

        if isinstance(v, list):
            v = v[0]
        if isinstance(v, dict):
            v = list(v.values())[0]

        y.append(float(v))

    plt.plot(N_values, y, marker='o', label=model)


plt.xlabel("N-way")
plt.ylabel("Accuracy")
plt.title("One-shot Accuracy Comparison")
plt.legend()
plt.grid()

out_path = os.path.join(DIR, "oneshot_comparison.png")

plt.savefig(out_path)

print("✅ Saved:", out_path)