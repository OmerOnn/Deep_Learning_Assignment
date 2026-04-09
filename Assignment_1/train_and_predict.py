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