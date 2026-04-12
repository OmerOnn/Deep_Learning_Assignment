import numpy as np
import forward_propagation as fp

def linear_backward(dZ, cache):
    """
    linear part of the backward propagation process for a single layer
    
    Args:
        dZ (vector): Gradient of the cost with respect to the linear output (of current layer l)
        cache (tuple): tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        
    Returns:
        dA_prev (vector): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (matrix): Gradient of the cost with respect to W (current layer l), same shape as W
        db (vector): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]  # Number of examples in the batch
    
    dW = (1 / m) * np.dot(dZ, A_prev.T) # Compute the gradient with respect to W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True) # Compute the gradient with respect to b
    dA_prev = np.dot(W.T, dZ) # Compute the gradient with respect to the activation of the previous layer
    
    return dA_prev, dW, db
    
    
    
    
def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer
    
    Args:
        dA (vector): post-activation gradient for current layer l
        cache (tuple): tuple contains both the linear cache (cache[0]) and the activation cache (cache[1])
        activation (string): the activation function used in this layer, stored as a text string: "softmax" or "relu"
        
    Returns:
        dA_prev (vector): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (matrix): Gradient of the cost with respect to W (current layer l), same shape as W
        db (vector): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]
    
    
    dZ = softmax_backward(dA, activation_cache) if activation == "softmax" else relu_backward(dA, activation_cache)  # Compute dZ based on the activation function
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db
    
    
    
def relu_backward(dA, activation_cache):
    """
    Backward propagation for a single RELU unit
    
    Args:
        dA (vector): post-activation gradient for current layer l
        activation_cache (vector): cache containing Z (stored in forward) for backpropagation
        
    Returns:
        dZ : gradient of the cost with respect to Z 
    """
    
    Z = activation_cache
    dZ = np.array(dA, copy=True)  # copy for not modifying dA while computing dZ 
    dZ[Z <= 0] = 0  # When z <= 0
    return dZ
    
    
    
    
def softmax_backward (dA, activation_cache):
    """
    Backward propagation for a softmax unit
    
    Args:
        dA (vector): ground truth labels Y
        activation_cache (vector): cache containing Z (stored in forward) for backpropagation
        
    Returns:
        dZ : gradient of the cost with respect to Z 
    """
    
    Z = activation_cache
    A, _ = fp.softmax(Z)  # Compute the softmax output using the cached Z
    dZ = A - dA  # Compute the gradient with respect to Z
    return dZ
    
    
    
    
    
def l_model_backward(AL, Y, caches):
    """
    Backward propagation process for the entire network.

    Args:
        AL (vector): probabilities vector, the output of the forward propagation (L_model_forward)
        Y (vector): the true labels vector
        caches (list): list of tuples caches containing for each layer:
                    a) the linear cache
                    b) the activation cache

    Returns:
        grads (dictionary): A dictionary with the gradients
                    grads["dA" + str(l)] = ...
                    grads["dW" + str(l)] = ...
                    grads["db" + str(l)] = ...

    Note:
        dA_prev is stored as layer l-1 because it is the gradient with respect to the
        input of the current layer, and that input is the activation of the previous layer.
    """
    
    grads = {}
    L = len(caches)  # Number of layers in the network
    m = AL.shape[1]  # Number of examples in the batch
    
    # Initializing the backpropagation for the last layer
    dA_prev_last, dW_last, db_last = linear_activation_backward(Y, caches[L - 1], activation="softmax")  # Backpropagation for the last layer
    grads["dA" + str(L - 1)] = dA_prev_last
    grads["dW" + str(L)] = dW_last
    grads["db" + str(L)] = db_last

    
    # Initializing the backpropagation for the hidden layer
    for l in range(L - 1, 0, -1):
        dA_prev, dW, db = linear_activation_backward(grads["dA" + str(l)], caches[l - 1], activation="relu")  # Backpropagation for the hidden layer
        grads["dA" + str(l - 1)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db
        
    return grads




    
def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent (lecture 1 page 21)
    
    Args:
        parameters (dictionary): containing the DNN architectures parameters
        grads (dictionary): containing the gradients (generated by L_model_backward)
        learning_rate (float): the learning rate used to update the parameters (the “alpha”)
        
    Returns:
        parameters : the updated values of the parameters object provided as input
    """
    
    L = len(parameters) // 2  # Number of layers in the network (there are 2 parameters for each layer: W and b)
    
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]  # Update W
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]  # Update b
        
    return parameters