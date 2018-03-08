import math

import random
import numpy as np
from Matrix import Matrix

class ActivationFunction:
    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

sigmoid = ActivationFunction(
    lambda x: 1 / (1 + math.exp(-x)),
    lambda y: y * (1-y))

tanh = ActivationFunction(
    lambda x: math.tanh(x),
    lambda y: 1 - (y**2))

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Weights between input nodes and hidden nodes
        self.weights_ih = np.random.random((self.hidden_nodes, self.input_nodes))
        # Weights between hidden nodes and output nodes
        self.weights_ho = np.random.random((self.output_nodes, self.hidden_nodes))

        self.bias_h = np.random.random((self.hidden_nodes, 1))
        self.bias_o = np.random.random((self.output_nodes, 1))

        self.setLearningRate()
        self.setActivationFunction()

    def setActivationFunction(self, func = sigmoid):
        self.activation_function = func

    def setLearningRate(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def predict(self, inp_array):
        # Generate hidden layer output
        inputs = np.matrix(inp_array) # Make a one-column matrix (2D) from a 1D array
        inputs2 = Matrix.fromArray(inp_array)
        hidden = Matrix.multiply(self.weights_ih, inputs) # Matrix product of input and weights
        hidden.add(self.bias_h) # Add bias to prevent '0' results
        hidden.map(self.activation_function.activation) # Activation Function - Applies activation function to the matrix.

        # Generate output layer output
        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(self.activation_function.activation)

        return output.toArray()

    def train(self, inp_array, target_array):
        # Generate hidden layer output
        inputs = Matrix.fromArray(inp_array) # Make a one-column matrix (2D) from a 1D array
        hidden = Matrix.multiply(self.weights_ih, inputs) # Matrix product of input and weights
        hidden.add(self.bias_h) # Add bias to prevent '0' results
        hidden.map(self.activation_function.activation) # Activation Function - Applies activation function to the matrix.

        # Generate output layer output
        outputs = Matrix.multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(self.activation_function.activation)
        
        # Convert arrays to matrix object
        targets = Matrix.fromArray(target_array)

        # Calculate total error
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        gradients = Matrix.mapMatrix(outputs, self.activation_function.dactivation)
        gradients.mult(output_errors)
        gradients.mult(self.learning_rate)

        # Calculate deltas
        hidden_transposed = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.multiply(gradients, hidden_transposed)

        # Apply deltas to weights
        self.weights_ho.add(weights_ho_deltas)

        # Adjust the bias by it's deltas (the gradients)
        self.bias_o.add(gradients)
        
        # Calculate hidden layer errors (error per perceptron)
        weights_ho_transposed = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiply(weights_ho_transposed, output_errors)

        # Calculate hidden layer gradients
        hidden_gradients = Matrix.mapMatrix(hidden, self.activation_function.dactivation)
        hidden_gradients.mult(hidden_errors)
        hidden_gradients.mult(self.learning_rate)

        # Calculate input layer -> hidden layer deltas
        inputs_transposed = Matrix.transpose(inputs)
        weights_ih_deltas = Matrix.multiply(hidden_gradients, inputs_transposed)

        # Apply deltas to weights
        self.weights_ih.add(weights_ih_deltas)

        # Adjust the hidden bias by it's deltas (the hidden gradients)
        self.bias_h.add(hidden_gradients)
