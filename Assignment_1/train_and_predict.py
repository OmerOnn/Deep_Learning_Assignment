import forward_propagation as fp
import backward_propagation as bp
import numpy as np
import copy


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
    
    
    
    # parameters = fp.initialize_parameters(layers_dims)  # Initialize parameters for the network
    # costs = []
    # m = X.shape[1]  # Number of examples in the training set
    
    # for i in range(num_iterations):
    #     for batch_start in range(0, m, batch_size):
            
    #         batch_end = min(batch_start + batch_size, m) # End index for the current batch, ensuring it does not exceed the total number of examples
    #         X_batch = X[:, batch_start : batch_end]
    #         Y_batch = Y[:, batch_start : batch_end]
            
    #         # Forward propagation, cost computation, backward propagation, and parameter update for the current batch
    #         AL, caches = fp.l_model_forward(X_batch, parameters, use_batchnorm)  # Forward propagation
    #         cost = fp.compute_cost(AL, Y_batch, l2_lambda)  # Compute the cost
    #         grads = bp.l_model_backward(AL, Y_batch, caches, l2_lambda)  # Backward propagation
    #         parameters = bp.update_parameters(parameters, grads, learning_rate)  # Update parameters
            
            
    #         if i % 100 == 0:
    #             costs.append(cost)  # Store the cost every 100 iterations of the outer loop
                
    
    # return parameters, costs
    
    
    
   # --------------------------
    # Internal train / validation split
    # --------------------------
    validation_split = 0.2
    min_improvement = 1e-4
    check_every = 100
    early_stopping_patience = 100

    m_total = X.shape[1]
    val_size = int(validation_split * m_total)

    X_val = X[:, :val_size]
    Y_val = Y[:, :val_size]

    X_train = X[:, val_size:]
    Y_train = Y[:, val_size:]

    # --------------------------
    # Initialization
    # --------------------------
    parameters = fp.initialize_parameters(layers_dims)
    best_parameters = copy.deepcopy(parameters)

    costs = []
    training_step = 0

    best_val_accuracy = -1
    best_step = 0
    steps_since_improvement = 0

    m_train = X_train.shape[1]
    batches_per_epoch = int(np.ceil(m_train / batch_size))

    stop_training = False

    print("\n" + "=" * 50)
    print("TRAINING STARTED")
    print("=" * 50)
    print(f"Layers dimensions: {layers_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Use batchnorm: {use_batchnorm}")
    print(f"L2 lambda: {l2_lambda}")
    print(f"Validation split: {validation_split}")
    print("=" * 50)

    # --------------------------
    # Training loop
    # --------------------------
    while not stop_training and training_step < num_iterations:
        permutation = np.random.permutation(m_train)
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        for batch_start in range(0, m_train, batch_size):
            if stop_training or training_step >= num_iterations:
                break

            batch_end = min(batch_start + batch_size, m_train)

            X_batch = X_train_shuffled[:, batch_start:batch_end]
            Y_batch = Y_train_shuffled[:, batch_start:batch_end]

            # initialize -> forward -> cost -> backward -> update
            AL, caches = fp.l_model_forward(X_batch, parameters, use_batchnorm)
            cost = fp.compute_cost(AL, Y_batch, parameters, l2_lambda)
            grads = bp.l_model_backward(AL, Y_batch, caches, l2_lambda)
            parameters = bp.update_parameters(parameters, grads, learning_rate)

            training_step += 1

            if training_step % check_every == 0:
                costs.append(cost)

                val_accuracy = predict(X_val, Y_val, parameters, use_batchnorm)

                print(f"Step Index: {training_step} | Cost: {cost:.6f}")
                print(f"Validation Accuracy: {val_accuracy:.4f}")

                if val_accuracy - best_val_accuracy > min_improvement:
                    best_val_accuracy = val_accuracy
                    best_parameters = copy.deepcopy(parameters)
                    best_step = training_step
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += check_every

                if steps_since_improvement >= early_stopping_patience:
                    print("\nEarly stopping triggered.")
                    print(f"No significant improvement for {early_stopping_patience} training steps.")
                    stop_training = True
                    break

    best_epochs = best_step / batches_per_epoch if batches_per_epoch > 0 else 0

    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Best Iterations: {best_step}")
    print(f"Best Epochs: {best_epochs:.2f}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print("=" * 50)

    return best_parameters, costs
    
    
    
    
def predict(X, Y, parameters, use_batchnorm=False):  
    """

    Args:
        X (vector): input data, a numpy array of shape (height*width , number_of_examples)
        Y (vector): “real” labels of the data, a vector of shape (num_of_classes, number of examples)
        parameters (_type_): _description_
        
    Returns:
        accuracay (float): the accuracy of the model on the given data
        
    Note:
        using the softmax function to normalize the output values
    """
    
    AL, _ = fp.l_model_forward(X, parameters, use_batchnorm)  # Forward propagation to get the output probabilities
    
    predictions = np.argmax(AL, axis=0) # Get the predicted class for each example by taking the index of the maximum probability
    true_labels = np.argmax(Y, axis=0) #
    
    correct_predictions = 0
    for pred, true_label in zip(predictions, true_labels):
        if pred == true_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(predictions)  # Calculate the accuracy
    return accuracy