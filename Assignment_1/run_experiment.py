import numpy as np
import sys
import time
from mnist_preprocess import load_and_preprocess_mnist, validation_split
from train_and_predict import l_layer_model, predict


def run_experiment(batch_size, use_batchnorm, l2_lambda=0.0):

    # Load data
    X_train, Y_train, X_test, Y_test = load_and_preprocess_mnist()

    # Required configuration
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009


    # Redirect all print statements to a file
    filename = f"training_report_batch_{batch_size}_use_batchnorm_{use_batchnorm}_l2_{l2_lambda}.txt"
    output_file = open(filename, "w", buffering=1)
    sys.stdout = output_file

        
    desired_epochs = 20

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
    print("======================================================")

    print("\n" * 3)
    # Print cost and index every 100 steps
    print(f'Costs every 100 iterations:')
    for i, cost in enumerate(costs):
        step_index = (i + 1) * 100
        print(f"Step Index: {step_index} | Cost: {cost:.6f}")

    output_file.close()
    sys.stdout = sys.__stdout__
    
    
if __name__ == "__main__":
    # batch_size = [16, 24, 32, 64, 128]
    # use_batchnorm_options = [False, True]
    # l2_values = [0, 0.001]
    # for batch in batch_size:
    #     for use_batchnorm in use_batchnorm_options:
    #         for l2 in l2_values:
    #             run_experiment(batch, use_batchnorm, l2_lambda=l2)
    
    
    
    batch_size = [16, 24, 32, 64, 128]
    for batch in batch_size:
            run_experiment(batch, False)