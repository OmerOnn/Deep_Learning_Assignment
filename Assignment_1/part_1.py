import numpy as np


def initialize_parameters(layer_dims):
    """
    Initialize parameters for a fully connected neural network.
    Args:
        layer_dims (list): dimensions of each layer in the network.
    Returns:
        dict: dictionary containing initialized weights and biases
        keys: W1, W2,..., WL for weights
        values: b1, b2,..., bL for biases
    """
    
    np.random.seed(42)
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01  # Initialization the weights
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))  # Initialization the bias
        
    return parameters
    
    
    
def linear_forward(A, W, b):
    """
    Compute the linear part of the forward propagation
    Args:
        A (vector): the activations of the previous layer
        W (matrix): the weight matrix of the current layer
        b (vector): the bias vector of the current layer

    Returns:
        Z (vector): linear component of the activation function
    """
    if W.shape[1] == A.shape[0]:  # Check if the dimensions are compatible for matrix multiplication
        Z = np.dot(W, A) + b   # Compute the linear part of the forward propagation
        return Z
    
def softmax(Z):
    """
    Compute the softmax activation function
    Args:
        Z (vector): linear component of the activation function
        
    Returns:
        A (vector): activations of the layer
        activations_cache (vector): cache containing Z for backpropagation
    """
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)  # Shift Z for numerical stability -> for long numbers, we wont get overflow
    exp_z = np.exp(shifted_Z)  # Compute the exponential of each element in shifted_Z
    sum_exp_z = np.sum(exp_z, axis=0, keepdims=True)  # Compute the sum of the exponentials
    A = exp_z / sum_exp_z  # Compute the softmax output
    activations_cache = Z
    return A, activations_cache
    
    
def relu(Z):
    """
    Compute the ReLU activation function
    Args:
        Z (vector): linear component of the activation function
        
    Returns:
        A (vector): activations of the layer
        activations_cache (vector): cache containing Z for backpropagation
    """
    A = np.maximum(0, Z)  # Apply ReLU activation function -> for each dimension, if Z is greater than 0, keep it; otherwise, set it to 0
    activations_cache = Z
    return A, activations_cache
    

def linear_activation_forward(A_prev, W, B, activation):
    """
    Compute the forward propagation for the LINEAR->ACTIVATION layer
    Args:
        A_prev (vector): activations of the previous layer
        W (matrix): weights matrix of the current layer
        B (vector):  bias vector of the current layer
        activation (string): activation function to be used in this layer, "relu" or "softmax"
        
    Returns:
        A (vector): activations of the current layer
        cache (dictionary): joint dictionary containing both linear_cache and activation_cache

    """
    Z = linear_forward(A_prev, W, B)  # Compute the linear part of the forward propagation
    
    A, activation_cache = softmax(Z) if activation == "softmax" else relu(Z)  # Compute the activation function
    linear_cache = (A_prev, W, B)  # Cache for the linear part of the forward propagation
    
    #cache = (linear_cache, activation_cache)  # Cache for the entire layer
    cache = {
        "linear_cache": linear_cache,
        "activation_cache": activation_cache
    }
    return A, cache
    
    
def l_model_forward(X, parameters, use_batchnorm):
    
    
def compute_cost(AL, Y):
    
    
def apply_batchnorm(A):