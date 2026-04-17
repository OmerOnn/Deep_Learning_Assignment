from mnist_preprocess import validation_split
import forward_propagation as fp
import backward_propagation as bp
import numpy as np


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

    X_train, Y_train, X_val, Y_val = validation_split(X, Y, validation_split=0.2)

    parameters = fp.initialize_parameters(layers_dims)
    costs = []
    m = X_train.shape[1]
    start_index = 0

    best_val = 0

    for i in range(num_iterations):

        current_epochs = m / batch_size
        print(f"\n================= Current Epochs {current_epochs:.2f} =================")

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

        if i % 100 == 0 and i != 0:

            costs.append(cost)
            
            val = predict(X_val, Y_val, parameters, use_batchnorm)
            
            if val >= best_val:
                best_val = val
                print(f"New best validation accuracy: {best_val:.4f} at iteration {i}")
            
            else:
                print(f"Validation accuracy: {val:.4f} at iteration {i}")
                break
                
        start_index = end_index + 1

        if i % (m // batch_size) == 0 and i != 0:

            print(f'Epoch {i // (m // batch_size)} completed. Cost: {cost:.6f}')

            permutation = np.random.permutation(m)

            X_train = X_train[:, permutation]
            Y_train = Y_train[:, permutation]

            print(f"Epoch {i // (m // batch_size)}: Cost = {cost}")

            start_index = 0
    
    train_accuracy = predict(X_train, Y_train, parameters, use_batchnorm)
    val_accuracy = predict(X_val, Y_val, parameters, use_batchnorm)
    
    print("\n" * 4)
    print(f"Final Train accuracy: {train_accuracy:.4f}")
    print(f"Final Validation accuracy: {val_accuracy:.4f}")

    return parameters, costs
    
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