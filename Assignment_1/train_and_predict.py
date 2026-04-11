import forward_propagation as fp
import backward_propagation as bp
import numpy as np


def l_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
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
    
    
    
    parameters = fp.initialize_parameters(layers_dims)  # Initialize parameters for the network
    costs = []
    m = X.shape[1]  # Number of examples in the training set
    
    for i in range(num_iterations):
        for batch_start in range(0, m, batch_size):
            
            batch_end = min(batch_start + batch_size, m) # End index for the current batch, ensuring it does not exceed the total number of examples
            X_batch = X[:, batch_start : batch_end]
            Y_batch = Y[:, batch_start : batch_end]
            
            # Forward propagation, cost computation, backward propagation, and parameter update for the current batch
            AL, caches = fp.l_model_forward(X_batch, parameters, use_batchnorm=False)  # Forward propagation
            costs = fp.compute_cost(AL, Y_batch)  # Compute the cost
            grads = bp.l_model_backward(AL, Y_batch, caches)  # Backward propagation
            parameters = bp.update_parameters(parameters, grads, learning_rate)  # Update parameters
            
            if i%100 == 0:
                costs.append(costs)  # Store the cost every 100 iterations for plotting the learning curve
                
    
    return parameters, costs
    
    
    
def predict(X, Y, parameters):  
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
    
    AL, _ = fp.l_model_forward(X, parameters, use_batchnorm=False)  # Forward propagation to get the output probabilities
    
    predictions = np.argmax(AL, axis=0) # Get the predicted class for each example by taking the index of the maximum probability
    true_labels = np.argmax(Y, axis=0) #
    
    correct_predictions = 0
    for pred, true_label in zip(predictions, true_labels):
        if pred == true_label:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(predictions)  # Calculate the accuracy
    return accuracy