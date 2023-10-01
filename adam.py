import numpy as np
from tensorflow.keras.datasets import mnist

# Loading MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preparing data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train_onehot = np.zeros((len(y_train), 10))
y_train_onehot[np.arange(len(y_train)), y_train] = 1
y_test_onehot = np.zeros((len(y_test), 10))
y_test_onehot[np.arange(len(y_test)), y_test] = 1

# Parameteres
learning_rate = 0.001
epochs = 10
batch_size = 64

# Network structure
input_size = 28 * 28  # input layer
hidden_size = 128  # hidden layer
output_size = 10  # output layer

# Randomizing weights of network
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))


# Activation function and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Training CNN
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        # Forward propagation
        batch_X = X_train[i:i + batch_size].reshape(-1, input_size)
        batch_y = y_train_onehot[i:i + batch_size]

        # Hidden layer
        hidden_input = np.dot(batch_X, weights_input_hidden) + bias_hidden
        hidden_output = sigmoid(hidden_input)

        # Output layer
        output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        output_output = softmax(output_input)

        # Error function
        error = batch_y - output_output

        # Backpropagation
        d_output = error
        d_hidden = np.dot(d_output, weights_hidden_output.T) * sigmoid_derivative(hidden_output)

        # Updating weights
        weights_input_hidden += learning_rate * np.dot(batch_X.T, d_hidden)
        weights_hidden_output += learning_rate * np.dot(hidden_output.T, d_output)

        bias_hidden += learning_rate * np.sum(d_hidden, axis=0, keepdims=True)
        bias_output += learning_rate * np.sum(d_output, axis=0, keepdims=True)

    # Calculating accuracy on a new data
    test_input = X_test.reshape(-1, input_size)
    hidden_input = np.dot(test_input, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output_output = softmax(output_input)

    predicted_labels = np.argmax(output_output, axis=1)
    accuracy = np.mean(predicted_labels == y_test)
    print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy * 100:.2f}%')

print("Finished training!")
