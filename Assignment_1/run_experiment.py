import numpy as np
import sys
import time
import os
import matplotlib.pyplot as plt
from mnist_preprocess import load_and_preprocess_mnist, validation_split
from train_and_predict import l_layer_model, predict


# All output files will be saved inside this folder.
# The folder will be created automatically if it does not exist.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_Ass1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment(batch_size, use_batchnorm=False, l2_lambda=0.0):

    # Load data
    X_train, Y_train, X_test, Y_test = load_and_preprocess_mnist()

    # Required configuration
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    # Redirect all print statements to a file inside output_Ass1
    filename = f"training_report_batch_{batch_size}_use_batchnorm_{use_batchnorm}_l2_{l2_lambda}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    output_file = open(filepath, "w", buffering=1)
    sys.stdout = output_file

    desired_epochs = 30

    # l_layer_model uses only 80% of X_train after internal validation split
    m_full = X_train.shape[1]
    m_effective_train = int((1 - 0.2) * m_full)

    batches_per_epoch = (m_effective_train + batch_size - 1) // batch_size
    num_iterations = desired_epochs * batches_per_epoch

    is_l2 = l2_lambda > 0.0

    print(f"\n ================ Running training with num_iterations = {num_iterations} ================")

    parameters, costs = l_layer_model(
        X_train,
        Y_train,
        layers_dims,
        learning_rate,
        num_iterations,
        batch_size,
        use_batchnorm,
        l2_lambda
    )

    # Evaluate model performance
    test_accuracy = predict(X_test, Y_test, parameters, use_batchnorm)

    print(f"| Final Test accuracy: {test_accuracy:.4f}")
    if use_batchnorm:
        print("| Batch normalization: True")
    if is_l2:
        print(f"| L2 regularization: True (lambda={l2_lambda})")
    print("======================================================")

    print("\n" * 3)

    # Print cost and index every 100 steps
    print("Costs every 100 iterations:")
    for i, cost in enumerate(costs):
        step_index = (i + 1) * 100
        print(f"Step Index: {step_index} | Cost: {cost:.6f}")

    output_file.close()
    sys.stdout = sys.__stdout__

    # Plot cost graph after training is finished
    steps = [(i + 1) * 100 for i in range(len(costs))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, costs, marker="o", markersize=3)
    plt.xlabel("Training step")
    plt.ylabel("Cost")

    title = f"Cost vs Training Step | batch={batch_size}, BN={use_batchnorm}, L2={l2_lambda}"
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()

    graph_filename = f"cost_graph_batch_{batch_size}_use_batchnorm_{use_batchnorm}_l2_{l2_lambda}.png"
    graph_path = os.path.join(OUTPUT_DIR, graph_filename)

    plt.savefig(graph_path)
    plt.close()


# =========================================
# |             Section 4                 |
# =========================================
def section_4():
    batch_size = 16
    run_experiment(batch_size)


# =========================================
# |             Section 5                 |
# =========================================
def section_5():
    batch_size = 16
    run_experiment(batch_size, True)


# =========================================
# |             Section 6                 |
# =========================================
def section_6():
    batch_size = 16
    l2_value = 0.001
    run_experiment(batch_size, True, l2_lambda=l2_value)


def plot_compare_running_time():
    labels = ["Regular", "BN", "R, L2=0.001", "BN, L2=0.01"]
    times = [12.86, 10.56, 12.24, 16.29]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, times)
    plt.xlabel("Experiment")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "training_time_comparison.png")
    plt.savefig(output_path)

    plt.close()


def plot_compare_final_accuracies():
    # Model names
    models = ["Regular", "BN", "R, L2=0.001", "BN, L2=0.001"]

    # Replace these with your real results
    train_acc = [0.9667, 0.9239, 0.9643, 0.9443]
    val_acc = [0.9491, 0.918, 0.9478, 0.9317]
    test_acc = [0.9489, 0.9197, 0.9476, 0.9379]

    x = np.arange(len(models))   # positions of model groups
    width = 0.25                 # width of each bar

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, train_acc, width, label="Train")
    plt.bar(x, val_acc, width, label="Validation")
    plt.bar(x + width, test_acc, width, label="Test")

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Train, Validation, and Test Accuracy Comparison")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "accuracy_comparison_all_models.png")
    plt.savefig(output_path)

    plt.close()


if __name__ == "__main__":
    # section_4()
    # section_5()
    # section_6()

    # plot_compare_running_time()
    plot_compare_final_accuracies()