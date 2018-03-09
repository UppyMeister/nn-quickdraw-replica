import math
import random
import numpy as np

class ActivationFunction:
    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

def softmax(x):
    e = np.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:  
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

sigmoid = ActivationFunction(
    lambda x: 1 / (1 + math.exp(-x)),
    lambda y: y * (1-y))

tanh = ActivationFunction(
    lambda x: math.tanh(x),
    lambda y: 1 - (y**2))

ReLU = ActivationFunction(
    lambda x: x * (x > 0),
    lambda y: 1 * (y > 0))

class NeuralNetwork:
    def __init__(self, a = 0, b = 0, c = 0):
        if (isinstance(a, dict)):
            for key in a:
                print(key)
                setattr(self, key, a[key])

            self.setActivationFunction()
        else:
            self.input_nodes = a
            self.hidden_nodes = b
            self.output_nodes = c

            # Weights between input nodes and hidden nodes
            self.weights_ih = np.random.uniform(low=-1, high=1, size=(self.hidden_nodes, self.input_nodes))
            
            # Weights between hidden nodes and output nodes
            self.weights_ho = np.random.uniform(low=-1, high=1, size=(self.output_nodes, self.hidden_nodes))

            # Biasses
            self.bias_h = np.random.uniform(low=-1, high=1, size=(self.hidden_nodes, 1))
            self.bias_o = np.random.uniform(low=-1, high=1, size=(self.output_nodes, 1))
            
            self.setLearningRate()
            self.setActivationFunction()

    def setActivationFunction(self, func = sigmoid):
        self.activation_function = func

    def setLearningRate(self, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def predict(self, inp_array):
        # Generate hidden layer output
        inputs = np.array(inp_array).reshape((len(inp_array), 1)) # Make a one-column matrix (2D) from a 1D array.
        hidden = np.dot(self.weights_ih, inputs) # Multiply weights and inputs.
        hidden = np.add(hidden, self.bias_h) # Add bias to prevent '0' results.
        hidden = np.vectorize(self.activation_function.activation)(hidden) # Apply activation function to the matrix.

        # Generate output layer output
        outputs = np.dot(self.weights_ho, hidden) # Multiply weights and inputs (previous layer outputs = current layer inputs).
        outputs = np.add(outputs, self.bias_o) # Add bias.
        outputs = np.vectorize(self.activation_function.activation)(outputs) # Apply activation function.

        return np.squeeze(np.asarray(outputs)).tolist() # Return a 1D array.

    def train(self, inp_array, target_array):
        # Generate hidden layer output
        inputs = np.array(inp_array).reshape((len(inp_array), 1)) # Make a one-column matrix (2D) from a 1D array.
        hidden = np.dot(self.weights_ih, inputs) # Multiply weights and inputs.
        hidden = np.add(hidden, self.bias_h) # Add bias to prevent '0' results.
        hidden = np.vectorize(self.activation_function.activation)(hidden) # Apply activation function to the matrix.

        # Generate output layer output
        outputs = np.dot(self.weights_ho, hidden) # Multiply weights and inputs (previous layer outputs = current layer inputs).
        outputs = np.add(outputs, self.bias_o) # Add bias.
        outputs = np.vectorize(self.activation_function.activation)(outputs) # Apply activation function.
        
        # Convert arrays to matrix object
        targets = np.array(target_array).reshape((len(target_array), 1))

        # Calculate total error
        output_errors = np.subtract(targets, outputs)

        # Calculate gradient
        gradients = np.vectorize(self.activation_function.dactivation)(outputs)
        gradients *= output_errors
        gradients *= self.learning_rate

        # Calculate deltas
        hidden_transposed = hidden.transpose()
        weights_ho_deltas = np.dot(gradients, hidden_transposed)

        # Apply deltas to weights
        self.weights_ho = np.add(self.weights_ho, weights_ho_deltas)

        # Adjust the bias by it's deltas (the gradients)
        self.bias_o = np.add(self.bias_o, gradients)
        
        # Calculate hidden layer errors (error per perceptron)
        weights_ho_transposed = self.weights_ho.transpose()
        hidden_errors = np.dot(weights_ho_transposed, output_errors)

        # Calculate hidden layer gradients
        hidden_gradients = np.vectorize(self.activation_function.dactivation)(hidden)
        hidden_gradients *= hidden_errors
        hidden_gradients *= self.learning_rate

        # Calculate input layer -> hidden layer deltas
        inputs_transposed = inputs.transpose()
        weights_ih_deltas = np.dot(hidden_gradients, inputs_transposed)

        # Apply deltas to weights
        self.weights_ih = np.add(self.weights_ih, weights_ih_deltas)

        # Adjust the hidden bias by it's deltas (the hidden gradients)
        self.bias_h = np.add(self.bias_h, hidden_gradients)
