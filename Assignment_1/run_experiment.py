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

    # # Early stopping settings
    # step_jump = 100
    # max_no_improvement_steps = 100
    # min_improvement = 1e-4

    # best_val_accuracy = 0
    # best_parameters = None
    # best_iteration = 0
    # best_costs = []
    # best_training_time = 0 # Time of the specific best run
    # steps_without_improvement = 0

        
    desired_epochs = 10

    # l_layer_model uses only 80% of X_train after internal validation split
    m_full = X_train.shape[1]
    m_effective_train = int((1 - 0.2) * m_full)

    batches_per_epoch = (m_effective_train + batch_size - 1) // batch_size

    # Calculate samples and batches per epoch
    m = X_train.shape[1]
    batches_per_epoch = int(np.ceil(m / batch_size))
    num_iterations = desired_epochs * batches_per_epoch
    
    is_l2 = l2_lambda > 0.0

    print(f"\n ================ Running training with num_iterations = {num_iterations} ================")

    # START TIMER
    training_start_time = time.time()

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

    # END TIMER
    training_end_time = time.time()
    pure_training_duration = training_end_time - training_start_time

    # Evaluate model performance
    test_accuracy = predict(X_test, Y_test, parameters, use_batchnorm)

    print(f"Final Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {pure_training_duration:.2f} seconds")

    print("\n" * 4)
    # Print cost and index every 100 steps
    for i, cost in enumerate(costs):
        step_index = (i + 1) * 100
        print(f"Step Index: {step_index} | Cost: {cost:.6f}")


    # # Final report summary
    # print("\n" + "="*40)
    # print("             FINAL REPORT")
    # print("="*40)
    # print(f"Layers dimensions: {layers_dims}")
    # print(f"Learning rate:     {learning_rate}")
    # print(f"Batch size:        {batch_size}")
    # print(f"Batchnorm used:    {use_batchnorm}")
    # print(f"L2 Regularization: {is_l2}")
    # print("-" * 40)
    # print(f"Best Iterations:       {best_iteration}")
    # print(f"Best Epochs:           {best_iteration / batches_per_epoch:.2f}")
    # print(f"Best Run Training Time: {best_training_time:.2f} seconds") 
    # print("-" * 40)
    # print(f"Final Train Accuracy:      {train_accuracy:.4f}")
    # print(f"Final Validation Accuracy: {best_val_accuracy:.4f}")
    # print(f"Final Test Accuracy:       {test_accuracy:.4f}")

    # if len(best_costs) > 0:
    #     overall_min_cost = min(best_costs)
    #     overall_min_idx = best_costs.index(overall_min_cost) * 100
    #     print(f"Lowest overall Cost:       {overall_min_cost:.6f} at step {overall_min_idx}")
    # print("="*40)

    output_file.close()
    sys.stdout = sys.__stdout__
    
    
if __name__ == "__main__":
    batch_size = [16, 24, 32, 64, 128]
    use_batchnorm_options = [False]
    l2_values = [0]
    for batch in batch_size:
        for l2 in l2_values:
            run_experiment(batch,False, l2_lambda=l2)