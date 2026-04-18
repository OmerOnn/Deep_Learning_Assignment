# import numpy as np
# from keras.datasets import mnist


# def one_hot_encode(labels, num_classes=10):
#     """
#     Convert labels to one-hot encoding.

#     Args:
#         labels (numpy.ndarray): shape (number_of_examples,)
#         num_classes (int): number of classes

#     Returns:
#         one_hot (numpy.ndarray): shape (num_classes, number_of_examples)
#     """
#     m = labels.shape[0]
#     one_hot = np.zeros((num_classes, m))

#     for i in range(m):
#         one_hot[labels[i], i] = 1

#     return one_hot


# def load_and_preprocess_mnist(validation_split=0.2, seed=42):
#     """
#     Load MNIST, preprocess it, and split train into train/validation.

#     Args:
#         validation_split (float): percentage of training data used for validation
#         seed (int): random seed for reproducibility

#     Returns:
#         X_train, Y_train, X_val, Y_val, X_test, Y_test
#     """

#     # Load MNIST
#     (train_X, train_y), (test_X, test_y) = mnist.load_data()

#     # Normalize pixel values to [0, 1]
#     train_X = train_X / 255.0
#     test_X = test_X / 255.0

#     # Flatten each image from (28, 28) to (784,)
#     train_X = train_X.reshape(train_X.shape[0], 28 * 28)
#     test_X = test_X.reshape(test_X.shape[0], 28 * 28)

#     # Transpose to shape (input_size, number_of_examples)
#     train_X = train_X.T
#     test_X = test_X.T

#     # Convert labels to one-hot
#     train_Y = one_hot_encode(train_y, 10)
#     test_Y = one_hot_encode(test_y, 10)

#     # Shuffle training data
#     np.random.seed(seed)
#     m = train_X.shape[1]
#     permutation = np.random.permutation(m)

#     train_X = train_X[:, permutation]
#     train_Y = train_Y[:, permutation]

#     # Split train / validation
#     val_size = int(validation_split * m)

#     X_val = train_X[:, :val_size]
#     Y_val = train_Y[:, :val_size]

#     X_train = train_X[:, val_size:]
#     Y_train = train_Y[:, val_size:]

#     X_test = test_X
#     Y_test = test_Y

#     return X_train, Y_train, X_val, Y_val, X_test, Y_test









import numpy as np
from keras.datasets import mnist


def one_hot_encode(labels, num_classes=10):
    m = labels.shape[0]
    one_hot = np.zeros((num_classes, m))

    for i in range(m):
        one_hot[labels[i], i] = 1

    return one_hot


def load_and_preprocess_mnist(seed=42):
    """
    Load MNIST, preprocess it, and return full training set and test set.
    """

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = train_X / 255.0
    test_X = test_X / 255.0

    train_X = train_X.reshape(train_X.shape[0], 28 * 28)
    test_X = test_X.reshape(test_X.shape[0], 28 * 28)

    train_X = train_X.T
    test_X = test_X.T

    train_Y = one_hot_encode(train_y, 10)
    test_Y = one_hot_encode(test_y, 10)

    np.random.seed(seed)
    m = train_X.shape[1]
    permutation = np.random.permutation(m)

    train_X = train_X[:, permutation]
    train_Y = train_Y[:, permutation]

    return train_X, train_Y, test_X, test_Y