from mnist_preprocess import validation_split
import forward_propagation as fp
import backward_propagation as bp
import numpy as np
import copy
import time


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, l2_lambda=0.0):
    """
    Args:
        X (vector): input data, a numpy array of shape (height*width , number_of_examples)
        Y (vector): “real” labels of the data, a vector of shape (num_of_classes, number of examples)
        layers_dims (list): list containing the dimensions of each layer, including the input
        learning_rate (float): learning rate for gradient descent
        num_iterations (int): number of iterations for gradient descent
        batch_size (int): number of examples in a single training batch

    Returns:
        parameters (dict): dictionary containing the learned parameters
        costs (list): list of costs computed during training, used for plotting the learning curve
    """

    # Split into train / validation
    X_train, Y_train, X_val, Y_val = validation_split(X, Y, validation_split=0.2)

    parameters = fp.initialize_parameters(layers_dims)
    costs = []

    m = X_train.shape[1]
    batches_per_epoch = (m + batch_size - 1) // batch_size
    start_index = 0

    best_val = 0
    num_of_batches = 0
    
    best_parameters = copy.deepcopy(parameters)
    epsilon = 1e-6
    small_improve_counter = 0
    
    # START TIMER
    training_start_time = time.time()

    for i in range(num_iterations):
        
        num_of_batches += 1
        epoch_progress = num_of_batches / batches_per_epoch

        print(
            f"\n================= Epoch number = {(i // batches_per_epoch) + 1} ,"
            f"Size: {epoch_progress:.4f} ================="
        )

        end_index = min(start_index + batch_size, m)

        X_batch = X_train[:, start_index:end_index]
        Y_batch = Y_train[:, start_index:end_index]

        # Forward propagation
        AL, caches = fp.l_model_forward(X_batch, parameters, use_batchnorm)

        # Compute cost
        cost = fp.compute_cost(AL, Y_batch, parameters, l2_lambda)

        # Backward propagation
        grads = bp.l_model_backward(AL, Y_batch, caches, l2_lambda)

        # Update parameters
        parameters = bp.update_parameters(parameters, grads, learning_rate)

        # Check validation every 50 iterations
        if i % 5 == 0 and i != 0:

            cur_val = predict(X_val, Y_val, parameters, use_batchnorm)

            # Check for improvement
            if cur_val - best_val > epsilon:
                best_val = cur_val
                small_improve_counter = 0  # Reset counter on improvement
                best_parameters = copy.deepcopy(parameters)  # Save the best parameters
                print(f"New best validation accuracy: {best_val:.4f} at iteration {i}")
                print("\n" * 4)
            else:
                small_improve_counter += 1
                if small_improve_counter >= 100:
                    print(f"Validation accuracy: {cur_val:.4f} at iteration {i}")
                    print(f"Early Stopping: No improvement for 100 training steps.")
                    break

        # Save cost every 100 iterations for plotting
        if i % 100 == 0 and i != 0:
            costs.append(cost)

        # Move to the next batch
        start_index = end_index

        # End of epoch
        if (i + 1) % batches_per_epoch == 0:
            print(f"Epoch {(i + 1) // batches_per_epoch} completed. Cost: {cost:.6f}")

            permutation = np.random.permutation(m)
            X_train = X_train[:, permutation]
            Y_train = Y_train[:, permutation]

            print(f"Epoch {(i + 1) // batches_per_epoch}: Cost = {cost}")

            start_index = 0
            num_of_batches = 0
            print("\n" * 4)
            
            
    # END TIMER
    training_end_time = time.time()
    pure_training_duration = training_end_time - training_start_time

    train_accuracy = predict(X_train, Y_train, best_parameters, use_batchnorm)
    val_accuracy = predict(X_val, Y_val, best_parameters, use_batchnorm)

    print("\n" * 4)
    print("================= Training Completed =================")
    print(f"| Training time: {pure_training_duration:.2f} seconds")
    print(f'| Total epochs completed: {(i // batches_per_epoch) + 1}')
    print(f"\n| Final Train accuracy: {train_accuracy:.4f}")
    print(f"| Final Validation accuracy: {val_accuracy:.4f}")

    return best_parameters, costs
    


def predict(X, Y, parameters, use_batchnorm=False):
    """
    Args:
        X (vector): input data, a numpy array of shape (height*width , number_of_examples)
        Y (vector): “real” labels of the data, a vector of shape (num_of_classes, number of examples)
        parameters: learned parameters

    Returns:
        accuracy (float): the accuracy of the model on the given data
    """

    AL, _ = fp.l_model_forward(X, parameters, use_batchnorm)

    predictions = np.argmax(AL, axis=0)
    true_labels = np.argmax(Y, axis=0)

    correct_predictions = 0
    for pred, true_label in zip(predictions, true_labels):
        if pred == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions)
    return accuracy