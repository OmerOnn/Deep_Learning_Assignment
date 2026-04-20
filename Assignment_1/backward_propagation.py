import numpy as np
import forward_propagation as fp


def linear_backward(dZ, cache, l2_lambda=0.0):
    """
    linear part of the backward propagation process for a single layer
    
    Args:
        dZ (vector): Gradient of the cost with respect to the linear output (of current layer l)
        cache (dict): dictionary of values (A_prev, W, b) coming from the forward propagation in the current layer
        l2_lambda (float): L2 regularization coefficient
        
    Returns:
        dA_prev (vector): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (matrix): Gradient of the cost with respect to W (current layer l), same shape as W
        db (vector): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache["A"], cache["W"], cache["b"]
    m = A_prev.shape[1]  # Number of examples in the batch
    
    # Compute the gradients
    dW = (1 / m) * np.dot(dZ, A_prev.T) + (l2_lambda / m) * W # Compute the gradient with respect to W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True) # Compute the gradient with respect to b
    dA_prev = np.dot(W.T, dZ) # Compute the gradient with respect to the activation of the previous layer
    
    return dA_prev, dW, db
    
     
def linear_activation_backward(dA, cache, activation, l2_lambda=0.0):
    """
    Backward propagation for the LINEAR->ACTIVATION layer
    
    Args:
        dA (vector): post-activation gradient for current layer l
        cache (dict): dictionary containing both the linear cache (cache["linear_cache"]) and the activation cache (cache["activation_cache"])
        activation (string): the activation function used in this layer, stored as a text string: "softmax" or "relu"
        l2_lambda (float): L2 regularization coefficient
        
    Returns:
        dA_prev (vector): Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW (matrix): Gradient of the cost with respect to W (current layer l), same shape as W
        db (vector): Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]
    
    
    dZ = softmax_backward(dA, activation_cache) if activation == "softmax" else relu_backward(dA, activation_cache)  # Compute dZ based on the activation function
    dA_prev, dW, db = linear_backward(dZ, linear_cache, l2_lambda)
    
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
    dZ = dA * (Z > 0)
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

    dZ = dA
    return dZ

    
    
def l_model_backward(AL, Y, caches, l2_lambda=0.0):
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
    last_cache = caches[L - 1]
    dA = AL - Y
    dZ = softmax_backward(dA, last_cache["activation_cache"])
    dA_prev, dW, db = linear_backward(dZ, last_cache["linear_cache"])

    grads["dA" + str(L - 1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    
    # Initializing the backpropagation for the hidden layer
    for l in range(L - 1, 0, -1):
        
        cur_cache = caches[l - 1]
        cur_dA = grads["dA" + str(l)]
        
        # if using batchnorm
        batch_norm_cache = cur_cache.get("cache_batchnorm", None)
        if batch_norm_cache is not None:
            cur_dA = batchnorm_backward(cur_dA, batch_norm_cache)
        
        dA_prev, dW, db = linear_activation_backward(cur_dA, cur_cache, activation="relu", l2_lambda=l2_lambda)  # Backpropagation for the hidden layer
        
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


def batchnorm_backward(dNA, batch_norm_cache):
    """_summary_

    Args:
        dNA (_type_): _description_
        batch_norm_cache (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    A = batch_norm_cache["A"]
    mean = batch_norm_cache["mean"]
    variance = batch_norm_cache["variance"]
    standard_div = batch_norm_cache["standard_div"]
    
    epsilon = 1e-8
    
    m = A.shape[1]
    
    center_A = A - mean
    
    dVariance = np.sum(dNA * center_A * -0.5 * (variance + epsilon)**(-1.5), axis=1, keepdims=True)
    
    dMean = np.sum(dNA * -standard_div, axis=1, keepdims=True) + dVariance * np.sum(-2.0 * center_A, axis=1, keepdims=True)
    
    dA = dNA * standard_div + dVariance * 2.0 * center_A / m + dMean / m
    return dA