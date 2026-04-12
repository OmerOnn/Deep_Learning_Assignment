from mnist_preprocess import load_and_preprocess_mnist
from train_and_predict import l_layer_model, predict


# Load data
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_and_preprocess_mnist()

# Required configuration
layers_dims = [784, 20, 7, 5, 10]
learning_rate = 0.009
batch_size = 64
use_batchnorm = False    # for section 4 -> For section 5 -> True

# Early stopping settings
max_no_improvement = 100
min_improvement = 1e-4

best_val_accuracy = 0
steps_without_improvement = 0

best_parameters = None

num_iterations = 1
epochs = 0

while steps_without_improvement < max_no_improvement:
    
    # Train the model using the required function
    parameters, costs = l_layer_model(
        X_train,
        Y_train,
        layers_dims,
        learning_rate,
        num_iterations,
        batch_size,
        use_batchnorm
    )

    # Evaluate on validation set
    val_accuracy = predict(X_val, Y_val, parameters)


    # Check improvement
    if val_accuracy - best_val_accuracy > min_improvement:
        best_val_accuracy = val_accuracy
        best_parameters = parameters
        steps_without_improvement = 0
    else:
        steps_without_improvement += 1

    epochs += num_iterations

    # print(
    #     f"Iterations: {num_iterations}, "
    #     f"Epochs so far: {epochs}, "
    #     f"Validation Accuracy: {val_accuracy:.4f}, "
    #     f"Steps without improvement: {steps_without_improvement}"
    # )

    num_iterations += 1

# Use best parameters
parameters = best_parameters

# Final evaluation
train_accuracy = predict(X_train, Y_train, parameters)
val_accuracy = predict(X_val, Y_val, parameters)
test_accuracy = predict(X_test, Y_test, parameters)

print("\n========= Training finished =========")
print(f"Layers dimensions: {layers_dims}")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Batchnorm used: {use_batchnorm}")
print(f"Total iterations checked: {num_iterations - 1}")
print(f"Total epochs counted: {epochs}")
print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")