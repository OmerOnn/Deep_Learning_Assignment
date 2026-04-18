# import numpy as np
# import sys
# import time
# from mnist_preprocess import load_and_preprocess_mnist
# from train_and_predict import l_layer_model, predict


# def run_experiment(batch_size, use_batchnorm, l2_lambda=0.0):

#     # Load data
#     X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_preprocess_mnist()

#     # Required configuration
#     layers_dims = [784, 20, 7, 5, 10]
#     learning_rate = 0.009


#     # Redirect all print statements to a file
#     filename = f"training_report_batch_{batch_size}_use_batchnorm_{use_batchnorm}_l2_{l2_lambda}.txt"
#     output_file = open(filename, "w", buffering=1)
#     sys.stdout = output_file

#     # Early stopping settings
#     step_jump = 100
#     max_no_improvement_steps = 100
#     min_improvement = 1e-4

#     best_val_accuracy = 0
#     best_parameters = None
#     best_iteration = 0
#     best_costs = []
#     best_training_time = 0 # Time of the specific best run
#     steps_without_improvement = 0
#     num_iterations = step_jump

#     # Calculate samples and batches per epoch
#     m = X_train.shape[1]
#     batches_per_epoch = int(np.ceil(m / batch_size))
    
#     is_l2 = l2_lambda > 0.0

#     while True:
#         print(f"\n ================ Running training with num_iterations = {num_iterations} ================")
        
#         current_epochs = num_iterations / batches_per_epoch
#         print(f"Current Epochs: {current_epochs:.2f}")

#         # START TIMER
#         training_start_time = time.time()

#         parameters, costs = l_layer_model(
#             X_train,
#             Y_train,
#             layers_dims,
#             learning_rate,
#             num_iterations,
#             batch_size,
#             use_batchnorm,
#             l2_lambda
#         )

#         # END TIMER
#         training_end_time = time.time()
#         pure_training_duration = training_end_time - training_start_time

#         # Print cost and index every 100 steps
#         for i, cost in enumerate(costs):
#             step_index = (i + 1) * 100
#             print(f"Step Index: {step_index} | Cost: {cost:.6f}")

#         # Evaluate model performance
#         train_accuracy = predict(X_train, Y_train, parameters, use_batchnorm)
#         val_accuracy = predict(X_val, Y_val, parameters, use_batchnorm)
#         test_accuracy = predict(X_test, Y_test, parameters, use_batchnorm)

#         print("\n" * 4)
#         print(f"Train accuracy: {train_accuracy:.4f}")
#         print(f"Validation accuracy: {val_accuracy:.4f}")
#         print(f"Test accuracy: {test_accuracy:.4f}")
#         print(f"training time: {pure_training_duration:.2f} seconds")

#         # Check for improvement on validation set
#         if val_accuracy - best_val_accuracy > min_improvement:
#             best_val_accuracy = val_accuracy
#             best_parameters = parameters
#             best_iteration = num_iterations
#             best_costs = costs[:]
#             best_training_time = pure_training_duration # Capture time of the improved model
#             steps_without_improvement = 0
#         else:
#             steps_without_improvement += step_jump

#         # Stopping criterion
#         if steps_without_improvement >= max_no_improvement_steps:
#             print(f"\nStopping criterion met: No improvement for {max_no_improvement_steps} steps.")
#             break
        
#         num_iterations += step_jump
#         print(f"\n" * 10)

#     # Final evaluation using the best parameters found
#     train_accuracy = predict(X_train, Y_train, best_parameters, use_batchnorm)
#     test_accuracy = predict(X_test, Y_test, best_parameters, use_batchnorm)

#     # Final report summary
#     print("\n" + "="*40)
#     print("             FINAL REPORT")
#     print("="*40)
#     print(f"Layers dimensions: {layers_dims}")
#     print(f"Learning rate:     {learning_rate}")
#     print(f"Batch size:        {batch_size}")
#     print(f"Batchnorm used:    {use_batchnorm}")
#     print(f"L2 Regularization: {is_l2}")
#     print("-" * 40)
#     print(f"Best Iterations:       {best_iteration}")
#     print(f"Best Epochs:           {best_iteration / batches_per_epoch:.2f}")
#     print(f"Best Run Training Time: {best_training_time:.2f} seconds") 
#     print("-" * 40)
#     print(f"Final Train Accuracy:      {train_accuracy:.4f}")
#     print(f"Final Validation Accuracy: {best_val_accuracy:.4f}")
#     print(f"Final Test Accuracy:       {test_accuracy:.4f}")

#     if len(best_costs) > 0:
#         overall_min_cost = min(best_costs)
#         overall_min_idx = best_costs.index(overall_min_cost) * 100
#         print(f"Lowest overall Cost:       {overall_min_cost:.6f} at step {overall_min_idx}")
#     print("="*40)

#     output_file.close()
#     sys.stdout = sys.__stdout__
    
    
# if __name__ == "__main__":
#     batch_size = [16, 32, 64]
#     use_batchnorm_options = [False]
#     l2_values = [0.001]
#     for batch in batch_size:
#         for l2 in l2_values:
#             run_experiment(batch,False, l2_lambda=l2)













import numpy as np
import sys
import time
from mnist_preprocess import load_and_preprocess_mnist
from train_and_predict import l_layer_model, predict


def split_train_validation(X, Y, validation_split=0.2):
    """
    Reproduce the same split used inside l_layer_model.
    """
    m_total = X.shape[1]
    val_size = int(validation_split * m_total)

    X_val = X[:, :val_size]
    Y_val = Y[:, :val_size]

    X_train = X[:, val_size:]
    Y_train = Y[:, val_size:]

    return X_train, Y_train, X_val, Y_val


def run_experiment(batch_size, use_batchnorm=False, l2_lambda=0.0):
    X_full, Y_full, X_test, Y_test = load_and_preprocess_mnist()

    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009
    num_iterations = 100000

    filename = f"training_report_batch_{batch_size}_use_batchnorm_{use_batchnorm}_l2_{l2_lambda}.txt"
    output_file = open(filename, "w", buffering=1)
    original_stdout = sys.stdout
    sys.stdout = output_file

    training_start_time = time.time()

    parameters, costs = l_layer_model(
        X_full,
        Y_full,
        layers_dims,
        learning_rate,
        num_iterations,
        batch_size,
        use_batchnorm,
        l2_lambda
    )

    training_end_time = time.time()
    pure_training_duration = training_end_time - training_start_time

    # recreate same split used inside l_layer_model
    X_train, Y_train, X_val, Y_val = split_train_validation(X_full, Y_full, validation_split=0.2)

    train_accuracy = predict(X_train, Y_train, parameters, use_batchnorm)
    val_accuracy = predict(X_val, Y_val, parameters, use_batchnorm)
    test_accuracy = predict(X_test, Y_test, parameters, use_batchnorm)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50)
    print(f"Layers dimensions:         {layers_dims}")
    print(f"Learning rate:             {learning_rate}")
    print(f"Batch size:                {batch_size}")
    print(f"Use batchnorm:             {use_batchnorm}")
    print(f"L2 lambda:                 {l2_lambda}")
    print(f"Training Time:             {pure_training_duration:.2f} seconds")
    print("-" * 50)
    print(f"Final Train Accuracy:      {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Test Accuracy:       {test_accuracy:.4f}")

    if len(costs) > 0:
        overall_min_cost = min(costs)
        overall_min_idx = (costs.index(overall_min_cost) + 1) * 100
        print(f"Lowest overall Cost:       {overall_min_cost:.6f} at step {overall_min_idx}")

    print("=" * 50)

    sys.stdout = original_stdout
    output_file.close()


if __name__ == "__main__":
    batch_sizes = [16, 24, 32, 64]
    use_batchnorm_options = [False]
    l2_values = [0.0]

    for batch in batch_sizes:
        for use_batchnorm in use_batchnorm_options:
            for l2 in l2_values:
                run_experiment(batch, use_batchnorm, l2_lambda=l2)